import numpy as np
from scipy.stats import chi2


def cond_entr_obj(binning):
    return binning.mean_cond_entr


from collections import defaultdict
from scipy.stats import entropy 

def incremental_entropy(h_old, n, c_old, c_new):
    delta = c_new - c_old
    if n == 0 or n == -delta: # old or new histogram empty
        return 0.0
    else:
        new_term = c_new*np.log(c_new) if c_new > 0 else 0
        old_term = c_old*np.log(c_old) if c_old > 0 else 0
        return np.log(n+delta)-(new_term + n*(np.log(n)-h_old) - old_term)/(n+delta)

class Binning:

    @staticmethod
    def trivial(x, y):
        n, _ = x.shape
        k = len(np.unique(y))
        # print('k', k)
        bins = np.zeros(n, dtype=int)
        max_bin = 0
        counts =  defaultdict(int) # np.zeros(n, dtype=int)
        counts[0] = n
        y_counts = defaultdict(lambda : np.zeros(k, int)) # np.zeros(shape=(n, k), dtype=int)
        _, y_counts[0] = np.unique(y, return_counts=True)
        cond_entr = defaultdict(float) # np.zeros(n)
        cond_entr[0] = entropy(y_counts[0], base=np.e)
        return Binning(x, y, bins, max_bin, counts, y_counts, cond_entr, cond_entr[0], 1)
    
    @staticmethod
    def from_assignment(x, y, bins):
        binning = Binning.trivial(x, y)
        for i, dest in enumerate(bins):
            binning.move(i, dest)
        return binning

    def __init__(self, x, y, bins, max_bin, counts, y_counts, cond_entr, mean_cond_entr, non_empty_bin_count):
        n, _ = x.shape
        self.n = n
        self.x = x
        self.y = y
        self.bins = bins
        self.max_bin = max_bin
        self.counts = counts
        self.y_counts = y_counts
        self.cond_entr = cond_entr
        self.mean_cond_entr = mean_cond_entr
        self.non_empty_bin_count = non_empty_bin_count

    def move(self, i, dest):
        orig = self.bins[i]
        if orig == dest: 
            return
        
        self.bins[i] = dest
        c = self.y[i]
        self.mean_cond_entr = self.mean_cond_entr - self.counts[orig]*self.cond_entr[orig]/self.n - self.counts[dest]*self.cond_entr[dest]/self.n
        self.counts[orig] -= 1
        self.counts[dest] += 1
        self.non_empty_bin_count = self.non_empty_bin_count + (self.counts[dest] == 1) - (self.counts[orig] == 0)
        self.y_counts[orig][c] -= 1
        self.y_counts[dest][c] += 1
        self.cond_entr[orig] = incremental_entropy(self.cond_entr[orig], self.counts[orig]+1, self.y_counts[orig][c]+1, self.y_counts[orig][c])
        self.cond_entr[dest] = incremental_entropy(self.cond_entr[dest], self.counts[dest]-1, self.y_counts[dest][c]-1, self.y_counts[dest][c])
        self.mean_cond_entr = self.mean_cond_entr + self.counts[orig]*self.cond_entr[orig]/self.n + self.counts[dest]*self.cond_entr[dest]/self.n

    def apply_cut_off(self, l, order):
        # TODO: remove code duplication
        split_off_bins = {}

        for i in range(l+1):
            j = order[i]
            b = self.bins[j]
            if b not in split_off_bins:
                self.max_bin += 1
                split_off_bins[b] = self.max_bin
            _b = split_off_bins[b]
            self.move(j, _b)

    def best_cut_off(self, order, objective, candidate_mask):
        _max_bin = self.max_bin
        split_off_bins = {}
        origins = np.zeros(self.n, dtype=int)

        obj_star = float('inf')
        i_star = -1
        # forward
        for i in range(self.n):
            j = order[i]
            b = self.bins[j]
            if b not in split_off_bins:
                self.max_bin += 1
                split_off_bins[b] = self.max_bin
            _b = split_off_bins[b]
            origins[i] = b
            self.move(j, _b)
            if candidate_mask[i]:
                obj_value = objective(self)
                if obj_value < obj_star:
                    i_star, obj_star = i, obj_value
        
        # rewind
        for i in range(self.n-1, -1, -1):
            j = order[i]
            self.move(j, origins[i])
        self.max_bin = _max_bin

        return i_star, obj_star
        # return i_star, mean_cond_entr_star


class VariableSelection:

    def __init__(self, delta=0.05, gamma='0.5', objective='cond_entr', multiple_cuts_per_var=True, verbose=False):
        self.delta = delta
        self.gamma = gamma
        self.objective = objective
        self.multiple_cuts_per_var = multiple_cuts_per_var
        self.verbose = verbose

    def p_value(self, binning):
        scaled_mi_diff = 2*self.n_*(self.cond_entr_-binning.mean_cond_entr)
        df = (self.k_-1)*(binning.non_empty_bin_count-self.non_empty_bin_count_)
        return 1 - chi2.cdf(scaled_mi_diff, df)
    
    @staticmethod
    def cond_entr(binning):
        return binning.mean_cond_entr

    def fit(self, x, y):
        obj_func = getattr(self, self.objective)
        binning = Binning.trivial(x, y)
        orders = np.argsort(x, axis=0)

        self.n_, self.p_ = x.shape
        self.k_ = len(np.unique(y))
        num_cand = int(self.n_**self.gamma)
        cand_indx = int((self.n_/self.n_**self.gamma) / 2) + np.linspace(0, self.n_, num_cand, endpoint=False).astype(int)
        cand_mask = np.zeros(self.n_, bool)
        cand_mask[cand_indx] = True

        self.non_empty_bin_count_ = binning.non_empty_bin_count
        self.cond_entr_ = binning.mean_cond_entr
        self.num_cuts_selected_ = np.zeros(self.p_, int)
        t = 0

        while True:
            j_star, i_star, obj_star = -1, -1, float('inf')
            for j in range(self.p_):
                if self.num_cuts_selected_[j]>0 and not self.multiple_cuts_per_var:
                    continue
                i_j_star, obj_j_star = binning.best_cut_off(orders[:, j], obj_func, cand_mask)
                if obj_j_star < obj_star:
                    j_star, i_star, obj_star = j, i_j_star, obj_j_star
            
            binning.apply_cut_off(i_star, orders[:, j_star])
            correction = (num_cand*self.p_-t) if self.multiple_cuts_per_var else (num_cand*(self.p_-t))
            if self.p_value(binning) <= self.delta / correction:
                self.num_cuts_selected_[j_star] += 1
                self.cond_entr_ = binning.mean_cond_entr
                self.non_empty_bin_count_ = binning.non_empty_bin_count
                if self.verbose: print('.', end='')
            else:
                break
            t += 1
        
        self.selected_ = np.flatnonzero(self.num_cuts_selected_)
        self.num_cuts_selected_ = self.num_cuts_selected_[self.selected_]
        return self

    def transform(self, x, y):
        return x[:, self.selected_], y
    
    def fit_transform(self, x, y):
        return self.fit(x, y).transform(x, y)
    