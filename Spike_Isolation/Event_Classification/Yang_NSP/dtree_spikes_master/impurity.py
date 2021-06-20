import numpy as np

def impurity(data=None,labels=None,coeff=None):
    labels    = np.array(labels);
    Label_set = np.unique(labels);
    data      = np.array(data);

    N   = len(data);
    Ndim=len(data[0]);


    V=data*coeff[:Ndim];
    V=sum(V.T).T + coeff[-1];

    idxp=np.where(V >  0);
    idxn=np.where(V <= 0);
    labels_p=labels[idxp];
    labels_n=labels[idxn];

    PrL,PrR      = np.ones(len(Label_set)),np.ones(len(Label_set));
    for l,label in enumerate(Label_set):
        if(sum(labels_p==label)>0):   PrR[l]=sum(labels_p==label)/len(labels_p);
        else:                         PrR[l]=1e-05;
        if(sum(labels_n==label)>0):   PrL[l]=sum(labels_n==label)/len(labels_n);
        else:                         PrL[l]=1e-05;


    info_R=-sum(PrR*np.log2(PrR));
    info_L=-sum(PrL*np.log2(PrL));
    info=+info_R*len(labels_p)/N +info_L*len(labels_n)/N;

    return info,info_R,info_L;