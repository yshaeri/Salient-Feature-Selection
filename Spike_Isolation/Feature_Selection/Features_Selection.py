# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:58:09 2015

@author: mali
"""

#%% #####################
import numpy as np, scipy as sp, scipy.stats as st
import sklearn.metrics;
import sys, os;
#%%
def Class_Discrimination( FS_Clus, ):
    C_No = len(FS_Clus);
    F_No = len(FS_Clus[0]);

    avg=np.mean;

    dw          = [[] for iFS in range(F_No)];
    d_withins   = [[] for iFS in range(F_No)];
    d_betweens  = [[] for iFS in range(F_No)];
    CDt         = [[] for iFS in range(F_No)];
    CDc         = [[] for iFS in range(F_No)];
    FSM         = [[] for iFS in range(F_No)];

    Pw = np.array([FS_Clus[iC][0].shape[1] for iC in range(C_No)]);   Pw = Pw/Pw.sum();

    for iFS in range(F_No):      # iFS is Feature Set index and F_No is No of Feature Sets
        mu      = np.array([avg(FS_Clus[iC][iFS],1)  for iC in range(C_No)]);

        dw[iFS] = np.array([np.std(FS_Clus[iC][iFS],axis=1) for iC in range(C_No)]);
        d_withins[iFS]  = np.array([[np.sqrt((Pw[iC]*dw[iFS][iC]**2+Pw[ic]*dw[iFS][ic]**2)/(Pw[ic]+Pw[iC])) for ic in range(C_No)] for iC in range(C_No)]);
        d_betweens[iFS] = np.array([[mu[iC]]*C_No for iC in range(C_No)])\
                         -np.array([[mu[iC]       for iC in range(C_No)]]*C_No);

        CDt[iFS] =              np.exp(np.abs(d_betweens[iFS]/d_withins[iFS]));

        CDc[iFS]  = np.array([np.product([CDt[iFS][iC,ic]**Pw[ic] for ic in range(C_No) if ic!=iC],axis=0)**(2/(1-Pw[iC]))\
                       /(np.sum([CDt[iFS][iC,ic]*Pw[ic]      for ic in range(C_No) if ic!=iC],axis=0)/(1-Pw[iC]))\
                       for iC in range(C_No)]);

        FSM[iFS] = np.sum([CDc[iFS][iC]*Pw[iC] for iC in range(C_No)],axis=0);

    return (dw, d_withins, d_betweens), CDt, CDc, FSM;

#%% Cluster based on salient features
def Optimal_Feature_Selection( FS_Clus, FE_Feat, O_No, ):
    C_No = len(FS_Clus);
    F_No = len(FS_Clus[0]);

    (dw, d_withins, d_betweens), CDt, CDc, FSM = Class_Discrimination( FS_Clus, );

    SF = [[] for iC in range(C_No)];
    OF = [[] for iC in range(C_No)];
    FD = [[[] for iO in range(O_No)] for iC in range(C_No)];

    for iC in range(C_No):
        for iS in range(F_No):
            for iF in range(len(CDc[iS][iC])):
                SF[iC].append((CDc[iS][iC][iF],iS,iF));
        SF[iC] = sorted(SF[iC], key=lambda row: row[0])[::-1]; #Sorted feature set based on Class discrimination for each channel
        for iO in range(O_No):
            _FD = [1]*len(SF[iC]);#Feature dissimilarity
            for iSF in range(len(SF[iC])):
                iS = SF[iC][iSF][1];
                iF = SF[iC][iSF][2];

                for io in range(iO):
                    iOS = OF[iC][io][0];
                    iOF = OF[iC][io][1];
                    _FD[iSF]*= 1-np.abs(np.corrcoef(FS_Clus[iC][iS][iF],FS_Clus[iC][iOS][iOF])[0,1]);#Feature dissimilarity

            FD[iC][iO] = _FD;#*(np.array(_FD)>0);     #Feature dissimilarity
            iP = np.argmax(np.array(SF[iC])[:,0]*_FD);
            iOS = SF[iC][iP][1];
            iOF = SF[iC][iP][2];
            OF[iC].append((iOS,iOF));

    return OF, SF, FD;

def Peak_Detector( FE_data ):
    C_No = len(FS_Clus[FE_mode]);
    F_No = len(FS_Clus[FE_mode][0]);
    PF = [[] for iC in range(C_No)]

    for iC in range(C_No):
        for iF in range(F_No):
            PF[iC].append( np.array([[np.max(FS_Clus[FE_mode][iC][iF],axis=0), np.argmax(FS_Clus[FE_mode][iC][iF],axis=0)],
                                     [np.min(FS_Clus[FE_mode][iC][iF],axis=0), np.argmin(FS_Clus[FE_mode][iC][iF],axis=0)]]).T );
        PF[iC] = np.array(PF[iC]);
    return  PF;

