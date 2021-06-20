# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 18:19:39 2016

@author: m-ali
"""
import numpy as np;
import pywt;
#%%
        
class Feature_Extractor(object):
    Transform= [];#Type, Xform
    Function = "";
    Level    = [];
    Coef = [];

    def __init__(self, Transform=None, Function=None, Level=None, ):
        class coef(object):
            def __init__(self, ):
                self.vector=[]; self.label=[]; self.ticks = [];
        if(Transform!=None): self.Transform = Transform;
        if(Function!=None):  self.Function  = Function;
        if(Level!=None):     self.Level     = Level;
        self.Coef = coef();

class Feature_Set:
    feature = [];
    def __init__(self, F_No=None, ):
        self.feature = [[] for f in range(F_No)];


#%%
def Feature_Extraxtion( Clus, FE_mode):
    S_No = len(Clus);
    Du = len(Clus[0].Clusters[0]);

    if(  len(FE_mode)==3): FE_Feat = Feature_Extractor(FE_mode[0],FE_mode[1],FE_mode[2]);#FE_mode, FEC_name[FE_mode], FEC_vector[FE_mode]);
    elif(len(FE_mode)==2): FE_Feat = Feature_Extractor(FE_mode[0],None,FE_mode[1]);#FE_mode, FEC_name[FE_mode], FEC_vector[FE_mode]);


    if(FE_mode[0][:3].lower()=='dwt'):
        F_No = FE_mode[2]+1;
        phi, psi = pywt.Wavelet(FE_mode[1]).wavefun(level=FE_mode[2])[:2];
        FE_Feat.Coef.vector.append(phi);
        FE_Feat.Coef.ticks.append(np.r_[0:Du:phi.shape[0]-2]);
        FE_Feat.Coef.label.append('A%d'%(FE_mode[2]));
        for f in range(FE_mode[2],0,-1):
            phi, psi = pywt.Wavelet(FE_mode[1]).wavefun(level=f)[:2];
            FE_Feat.Coef.vector.append(psi);
            FE_Feat.Coef.ticks.append(np.r_[0:Du:psi.shape[0]-2]);
            FE_Feat.Coef.label.append('D%d'%f);
    elif(FE_mode[0][:11].lower()=='derivatives'):
        F_No = FE_mode[1];
        for f in range(F_No,0,-1):
            FE_Feat.Coef.vector.append(np.array([1]+[0]*f+[-1]));
            FE_Feat.Coef.label.append('dd%d'%(f+1));
            FE_Feat.Coef.ticks.append(np.r_[0:Du-f-1]);
    else:
        F_No=1;
        FE_Feat.Coef.vector.append(np.array([1]));
        FE_Feat.Coef.label.append('s');
        FE_Feat.Coef.ticks.append(np.r_[0:Du]);
        pass;

    FE_Clus = [[Feature_Set(F_No)   for c in range(len(Clus[s].Clusters))] for s in range(S_No)];

    for s in range(S_No):
        C_No = len(Clus[s].Clusters);#Clus[s].Clus_No
        for c in range(C_No):
            for f in range(F_No):
                if(FE_mode[0][:3].lower() == 'dwt'):  #DWT with subsampling
                    FE_Clus[s][c].feature[f] = np.apply_along_axis(lambda x: pywt.wavedec(x, FE_mode[1], level=FE_mode[2])[f], axis=0, arr= Clus[s].Clusters[c]);
                elif(FE_mode[0][:11].lower()=='derivatives'):
                    if f==0:    FE_Clus[s][c].feature[f] = np.diff(Clus[s].Clusters[c],       axis=0);
                    else:       FE_Clus[s][c].feature[f] = np.diff(FE_Clus[s][c].feature[f-1],axis=0);
                else:
                    FE_Clus[s][c].feature[f] = Clus[s].Clusters[c];

    return FE_Feat, FE_Clus;
