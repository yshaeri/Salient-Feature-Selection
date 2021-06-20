#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 14:32:39 2018

@author: m-ali
"""

#%% Definitions (parameters, variables, functions)

FE_mode = (' ',1); 
FE_Feat, FE_Clus = FE.Feature_Extraxtion( Clus, FE_mode );

O_No = 2; # No of optimal features
F_No = len(FE_Feat.Coef.label)

CD   = [[[[] for i in range(4)] for d in range(5)] for s in range(S_No)];
T    = [[[[] for i in range(4)] for d in range(5)] for s in range(S_No)];

CAG = [[[[] for iC in range(len(FE_Clus[s]))] for iFS in range(F_No)] for s in range(S_No)];#1-PeG

CD_GT = [[] for s in range(S_No)];

#%% 4. Salient Feature Selection (FS)

if(IMode=='Generate'):
    # iS is Recording Site index and S_No is No of Feature Sets
    # iF is Feature Set index and F_No is No of Feature Sets
    for iS in range(S_No):
        C_No = len(FE_Clus[iS]); # No of Clusters (units)
        FS_Clus = [[FE_Clus[iS][iC].feature[iF] for iF in range(F_No)] for iC in range(C_No)]

        r, c = 0, 0;
        T0=time.time()
        (dw, d_withins, d_betweens), CDt, CDc, FSM = FS.Class_Discrimination( FS_Clus, );
        CD[iS][r][c] = CDc;
        T[iS][r][c]  = time.time()-T0;
    
        for iFS in range(F_No):                                 # iFS is Feature Set index and F_No is No of Feature Sets
            for iF in range(len(FE_Feat.Coef.ticks[iFS])):      # iFS is Feature Set index and F_No is No of Feature Sets
                for iC in range(C_No):
                    data =[];
                    label=[];
                    for ic in range(C_No):
                        data +=list(FS_Clus[ic][iFS][iF]);
                        label+= [1 if iC==ic else 0]*len(FS_Clus[ic][iFS][iF]);
                    data = np.array(data);
                    x_range = [data.min(), data.max()];
                    i1 = np.where(label)[0];
                    i0 = np.where(np.array(label)!=1)[0];
    
                    data = np.array([data]).T;
                    GNB = sk.naive_bayes.GaussianNB().fit(data, label);
                    CAG[iS][iFS][iC].append(1-sum(GNB.predict(data)!=np.array(label))/len(label));
    
        CD_GT[iS] = [CD[iS][4][0],CD[iS][4][2],CAG[iS],CD[iS][4][1],CD[iS][4][3]];  #Ground Truth = Bayes decision accuracy
    CD_GT_name = ["Bayes Decision accuracy using Histogram","Bayes Decision accuracy with normal estimation","Naive Normal Bayes Classifier Accuracy", "Bayes Decision accuracy using smoothed Histogram","Bayes Decision accuracy with normal estimation (fast)","Bayes Classifier Accuracy","KNN  Classifier Accuracy"];

    with open('FS_Data - r%d.dat'%(_run+1), "wb") as file:
        pickle.dump({'CD':CD, 'T':T, 'CAG':CAG,'CD_GT':CD_GT,'CD_GT_name':CD_GT_name,'O_No':O_No, 'F_No':F_No, }, file);


elif(IMode=='Load'):
    ##%% Load data clusters to file
    with open('FS_Data - r%d.dat'%_run, "rb") as file:	dic= pickle.load(file)
    CD=dic['CD'];
    T=dic['T'];
    CAG=dic['CAG'];
    CD_GT=dic['CD_GT'];
    CD_GT_name=dic['CD_GT_name']
    O_No=dic['O_No']
    F_No=dic['F_No']
    CD_Average=dic['CD_Average']
    CD_Norm=dic['CD_Norm']

elif(IMode=='Batch-load'):
    CD, T, CAB, CAG, CD_GT, = [],[],[],[],[]
    for r in range(len(_Runs)):
        with open('FS_Data - r%d.dat'%_Runs[r], "rb") as file:	dic= pickle.load(file)
        CD    +=dic['CD']   [(5 if r else 0):]
        T     +=dic['T']    [(5 if r else 0):]
        CAG   +=dic['CAG']  [(5 if r else 0):]
        CD_GT +=dic['CD_GT'][(5 if r else 0):]
    
    CD_GT_name=dic['CD_GT_name']
    O_No=dic['O_No']
    F_No=dic['F_No']
    CD_Average=dic['CD_Average']
    CD_Norm=dic['CD_Norm']


#%%
