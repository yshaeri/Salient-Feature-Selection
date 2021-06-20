# Module containing impurity metrics.

from collections import Counter
import numpy as np
import scipy as sp
import sys

def Probabilty(xx):
    if len(xx) == 0:        return [];
    try:
        freq = Counter(xx); #frequencies
        n_records = float(len(xx))
        return np.array([v/n_records for v in freq.values()])
    except TypeError:
        sys.stderr.write("Please use Numpy arrays!")
        sys.exit()

#_train = np.concatenate((train_data,np.transpose([train_label])),axis=1)
#impurity.gini(_train[:, -1])
#xx = _train[:, -1]
def gini(xx):
    Pr = Probabilty(xx);
    return 1 - np.sum(np.square(Pr));

def entropy(xx):
    Pr = Probabilty(xx);
    return sp.stats.entropy(Pr,base=2); #np.sum([-p*np.log2(p) if p != 0 else None for p in Pr])

def impurity(split,metric):
    ipm = 0;
    if(metric==gini or metric=='gini'):
        imp = sum([gini(x) for x in split])
    elif(metric==entropy or metric=='entropy' or metric=='info gain' or metric=='information gain'):
        imp = sum([entropy(x)*len(x) for x in split])/sum([len(x) for x in split]);
    return imp;