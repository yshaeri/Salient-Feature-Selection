import numpy as np
import itertools
from Spike_Isolation.Event_Classification.Yang_NSP.dtree_spikes_master.perturb import perturb

def axis_p(data=None,labels=None):
    Ndim = len(data[0]);
    N    = len(data);
    #coeff = np.eye(Ndim);    coeff[:,Ndim]=0;
    coeff = np.concatenate([np.eye(Ndim), [[0]]*Ndim ], axis=1)
    Label_set = np.unique(labels)
    Nclass = len(Label_set);
    x = np.floor(Nclass / 2);
    impurity_a_min = x/Nclass*np.log2(x)+(Nclass-x)/Nclass*np.log2(Nclass-x);

    impurity_plane = []
    for i in range(len(coeff)):
        coeff[i,:], _impurity_plane = perturb(data, labels, coeff[i,:], 0);
        impurity_plane.append(_impurity_plane);

    idx = np.where(impurity_plane==min(impurity_plane))[0];

    V=[];
    #for i in range(len(idx)):
    for i in idx: #??
        tmp = data*coeff[i,:Ndim];
        tmp = np.abs(sum(tmp.T) +coeff[i,-1]);

        V2 = [];
        for l,label in enumerate(Label_set):
            V2.append(sum(tmp[np.where(labels == label)[0]])); 
        V2 = np.array(list(itertools.combinations(V2, 2)));
        V2 = sum(V2.T);
        V.append(np.min(V2));

    coeff = coeff[idx,:];
    idx   = np.where(V == max(V))[0];
    idx   = idx[0];
    coeff = coeff[idx,:];
    return coeff;
