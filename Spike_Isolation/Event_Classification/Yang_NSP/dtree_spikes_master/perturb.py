import numpy as np
from Spike_Isolation.Event_Classification.Yang_NSP.dtree_spikes_master.impurity import impurity;
#coeff=coeff[i,:]; opt=0

def perturb(data=None,labels=None,coeff=None,opt=None):

    labels = np.array(labels)
    Label_set = np.unique(labels)
    data   = np.array(data)

    N=len(data);
    Ndim=len(data[0]);
    Nclass=len(Label_set);

    V=data*coeff[:Ndim];
    V=sum(V.T).T + coeff[-1];

    Pstag=1;
    if opt == 0: dim_start=Ndim;
    else:        dim_start=0;

    for dim in range(dim_start,Ndim + 1):
        if dim == Ndim: U = V;
        else:           U = V/data[:,dim];

        candidates = coeff[dim] - U;
        candidates,idx = np.sort(candidates), np.argsort(candidates);
        labels_sort =labels[idx];

        class_logic = [];
        for l in range(Nclass):  class_logic.append( labels_sort==Label_set[l] );
        class_sum = np.cumsum(class_logic, axis=1);
        Pr1 = class_sum[:,:-1]/[np.r_[1:N]];
        Pr2 = (np.transpose([class_sum[:,-1]]*(N-1)) -class_sum[:,:-1])/(N-np.r_[1:N])
        idx = np.where(Pr1 == 0);  Pr1[idx] = 0.00001;
        idx = np.where(Pr2 == 0);  Pr2[idx] = 0.00001;
        impurity_coeff = -np.r_[1:N]/N*sum(Pr1*np.log2(Pr1)) - np.r_[N-1:0:-1]/N*sum(Pr2*np.log2(Pr2));

#        impurity_coeff = [];
#        Pr1,Pr2      = np.ones(Nclass),np.ones(Nclass);
#        for i in range(1, N):
#            for j in range(Nclass):
#                if sum(labels_sort[:i] == Label_set[j]):  Pr1[j]=sum(labels_sort[:i] == Label_set[j]) / i;
#                else:                                     Pr1[j]=1e-05;
#                if sum(labels_sort[i:] == Label_set[j]):  Pr2[j]=sum(labels_sort[i:] == Label_set[j]) /(N - i);
#                else:                                     Pr2[j]=1e-05;
#            impurity_coeff.append( -i*sum(Pr1*np.log2(Pr1)) -(N-i)*sum(Pr2*np.log2(Pr2)) );
#            #impurity_coeff[-1]=impurity_coeff[-1] / N;
#        impurity_coeff = np.array(impurity_coeff)/N;

        min_impurity_coeff,idx=min(impurity_coeff),np.argmin(impurity_coeff);
        coeff_new=coeff;
        coeff_new[dim]=(candidates[idx] + candidates[idx+1]) / 2;

        impurity_plane    = impurity(data,labels,coeff)[0];
        impurity_plane_new= impurity(data,labels,coeff_new)[0];
        if impurity_plane - impurity_plane_new > 0.001:
            coeff =coeff_new;
            impurity_plane =impurity_plane_new;
            Pstag =1;
        else:
            if Pstag > np.random.uniform(): #np.random.uniform(0,1):
                coeff=coeff_new;
                impurity_plane=impurity_plane_new;
            Pstag = Pstag - 0.1*Pstag;

    return coeff,impurity_plane;