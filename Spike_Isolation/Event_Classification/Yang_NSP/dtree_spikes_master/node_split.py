import numpy as np;
from Spike_Isolation.Event_Classification.Yang_NSP.dtree_spikes_master.impurity import impurity;
from Spike_Isolation.Event_Classification.Yang_NSP.dtree_spikes_master.perturb import perturb;
from Spike_Isolation.Event_Classification.Yang_NSP.dtree_spikes_master.axis_p import axis_p;
from Spike_Isolation.Event_Classification.Yang_NSP.dtree_spikes_master.randomization import randomization;
#import time

#data,labels = data_tree[inode],labels_tree[inode]

def node_split(data=None,labels=None,opt=None,bits=None,repeats=50):
    Ndim=len(data[0]);
    N=len(data);
    coeff_a = np.zeros(Ndim + 1);
    coeff_a = axis_p(data,labels);
    impurity_axis = impurity(data,labels,coeff_a)[0];
    if opt == 0:    return coeff_a
    
    coeff_opt,impurity_plane = perturb(data,labels,coeff_a,1);
    j=1;
    J=repeats;#No of repeats
    #T0=time.time()
    while (j <= J):
        coeff_r,impurity_plane_r=randomization(data,labels,coeff_opt);
        
        if impurity_plane_r < impurity_plane:
            coeff_opt = coeff_r;
            coeff_opt,impurity_plane = perturb(data,labels,coeff_opt,1);
        j=j + 1;
        #print(j,time.time()-T0)

    coeff_one = max(abs(coeff_opt[:Ndim]));
    idx = np.where( abs(coeff_opt)==coeff_one )[0][0];
    coeff_opt = coeff_opt/ coeff_opt[idx];#coeff_one

    coeff_opt[:Ndim] = np.round( coeff_opt[:Ndim] * 2**(bits-1) ) / 2**(bits-1);
    coeff_opt[np.where(coeff_opt[:Ndim]==1 )]= 1-1/2**(bits-1);
    coeff_opt[np.where(coeff_opt[:Ndim]==-1)]=-1+1/2**(bits-1);
    coeff_opt[idx]=1;
    coeff_opt,impurity_plane = perturb(data,labels,coeff_opt,0);

    if impurity_axis - impurity_plane <= 0:        coeff = coeff_a;
    else:                                          coeff = coeff_opt;

    return coeff;