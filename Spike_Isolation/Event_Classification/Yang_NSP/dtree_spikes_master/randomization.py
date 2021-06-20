#from oct2py import Oct2Py as oc
#import random
import numpy as np;
from Spike_Isolation.Event_Classification.Yang_NSP.dtree_spikes_master.impurity import impurity;

'''
data=np.array([[1,2],[3,4],[5,6],[7,8]])
labels=[1,2,1,2]
coeff=[1,2,.7]
'''
def randomization(data=None,labels=None,coeff=None):

    labels = np.array(labels)
    Label_set = np.unique(labels)
    data   = np.array(data)
    
    N=len(data);
    Ndim=len(data[0]);

    rvector=np.random.uniform(-1,1,Ndim+1);#(np.random.uniform(0,1,Ndim+1)-0.5)*2

    V = data*coeff[:Ndim];
    V = sum(V.T).T + coeff[-1];

    R = data*rvector[:Ndim];
    R = sum(R.T).T + rvector[-1];

    candidates = -V/R;

    candidates,idx = np.sort(candidates), np.argsort(candidates)
    labels_sort    = labels[idx];

    impurity_alpha = np.ones(N)*np.Inf;#[];
    Pr1,Pr2      = np.ones(len(Label_set)),np.ones(len(Label_set));
    for i in range(1,N):
        for l,label in enumerate(Label_set):  #??
        #for l in range(Ndim):
            #label = Label_set[l];

            #Low Speed
#            if sum(labels_sort[:i]==label): Pr1[l]=sum(labels_sort[:i]==label)/i;
#            else:                           Pr1[l]=1e-05;
#            if sum(labels_sort[i:]==label): Pr2[l]=sum(labels_sort[i:]==label)/(N-i);
#            else:                           Pr2[l]=1e-05;
            #High Speed
            if len(np.where(labels_sort[:i]==label)[0])!=0: Pr1[l]=len(np.where(labels_sort[:i]==label)[0])/i;
            else:                                           Pr1[l]=1e-05;
            if len(np.where(labels_sort[i:]==label)[0])!=0: Pr2[l]=len(np.where(labels_sort[i:]==label)[0])/(N-i);
            else:                                           Pr2[l]=1e-05;

        #impurity_alpha.append( -i*sum(Pr1*np.log2(Pr1)) -(N-i)*sum(Pr2*np.log2(Pr2)));
        impurity_alpha[i-1] = -i*sum(Pr1*np.log2(Pr1)) -(N-i)*sum(Pr2*np.log2(Pr2));
    impurity_alpha = np.array(impurity_alpha)/N;


    min_impurity_alpha,idx = min(impurity_alpha),np.argmin(impurity_alpha);
    alpha = (candidates[idx] + candidates[idx + 1]) / 2;
    coeff = coeff + rvector*alpha;
    impurity_plane = impurity(data,labels,coeff)[0];

    return coeff,impurity_plane;
