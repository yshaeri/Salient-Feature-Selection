# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:44:14 2016

@author: m-ali
"""
import numpy as np;
import scipy as sp;
import scipy.io
import sklearn.cluster;
from sklearn.metrics import silhouette_score;
import os, glob, platform, distro;

import pickle
class ARRAY(object):
    Clusters=[];
    def __init__(self, *args, **kwargs):
      self.Clusters     = args[0];

Unit_No=0;
SR=0;
Rs=0;
DR=0;
#%%
def Load_Data( Data_mode, Data_Set, ix ):
    DIR = os.getenv("HOME")+'/Public/Dataset/CRCNS.org/Visual Cortex/pvc1/crcns-ringach-data/neurodata/';

    File_Lists=[];
    for dr in sorted([d for d in glob.glob( DIR+'*') if os.path.isdir(d)]):
        try:
            File_Lists.append( sorted(glob.glob(dr+'/'+'*.mat')));
        except Exception as e:
            print(e);
            raise;
            return -1;

    if ix[0] == 'all':
        File_List=[];
        for d in range(len(File_Lists)):
            for f in range(len(File_Lists[d])):
                File_List.append(File_Lists[d][f]);
    else:
        try:
            File_List = [File_Lists[ix[0][0]][ix[0][1]]];
        except IndexError as e:
            print("IndexError: File list is index out of range.", e.args);
            return -1;
        except Exception as e:
            print(e);
            return -1;

    SR = 30000;
    properties = (SR, Rs, DR);

    data = [];
    S_No = 0;

    for file in File_List:
        try:
            Spikes = sp.io.loadmat( file )['pepANA']['listOfResults'][0,0];
            print("File", os.path.basename(file), "is inserted.\n");
        except IndexError as e:
            print("IndexError: File list is index out of range.", e.args);
            return -1;
        except Exception as e:
            print(e);
            return -1;

        if(str(ix[0]).lower() == 'all' or str(ix[1]).lower()=='all'): L_No = len(Spikes[0]);
        else:                                       L_No = len(ix[1])
        for l in range(L_No): #List Index
            if(str(ix[0]).lower() == 'all' or str(ix[1]).lower()=='all'):
                iD = 'all';
                il = l;
            else:
                iD = ix[1][l][1];
                il = ix[1][l][0];
            ir = 0;
            R_No = len(Spikes[0,il]['repeat'][0,0][0]);
            D_No = len(Spikes[0,il]['repeat'][0,0][0,ir]['data'][0,0][0]);

            if  str(iD).lower()=='all':  iD = np.r_[0:D_No];
            s_no = len(iD);
            for s in range(s_no): #Site Index
                data.append(ARRAY([],0));

                if(Data_mode[1]=='whitened'):  MUA = sp.cluster.vq.whiten(Spikes[0,il]['repeat'][0,0][0,0]['data'][0,0][0,iD[s]][0,1]);
                else:                          MUA = Spikes[0,il]['repeat'][0,0][0,0]['data'][0,0][0,iD[s]][0,1];

                K = np.min((6,len(MUA.T)-1));
                if(K>2):
                    kmean = [sklearn.cluster.KMeans(n_clusters=k) for k in range(2,K)];
                    for k in range(0,K-2):  kmean[k].fit_transform(MUA.T);
                    Clus_No = 2 + np.argmax([silhouette_score(MUA.T, kmean[k].labels_, metric='sqeuclidean') for k in range(0,K-2)]);
                    MUA = Spikes[0,il]['repeat'][0,0][0,0]['data'][0,0][0,iD[s]][0,1];
                    Clus_temp = [];
    
                    for c in set(kmean[Clus_No-2].labels_):
                        IDX = np.where(kmean[Clus_No-2].labels_==c)[0];
                        Clus_temp.append(MUA[:,IDX]);
    
                    Ix = np.argsort([np.var(np.mean(Clus_temp[c],1)) for c in range(Clus_No)])[::-1];
    
                    for c in range(Clus_No):
                        data[-1].Clusters.append(Clus_temp[Ix[c]]);
                else:   
                    MUA = Spikes[0,il]['repeat'][0,0][0,0]['data'][0,0][0,iD[s]][0,1];
                    data[-1].Clusters.append(MUA)
    
            print("%d channels of data #%d is clustered successfully." %(s_no,il));
            S_No = S_No+s_no;
    return data, properties;

def Unit_Gen(Data_Set, iCs):
    Dist = distro.linux_distribution()[0];
    if(Dist.lower()   == 'ubuntu' and Data_Set=='pvc1'):      DIR = os.getenv("HOME")+'/Public/Dataset/CRCNS.org/Visual Cortex/pvc1/crcns-ringach-data/neurodata/';
    elif(Dist.lower() == 'debian' and Data_Set=='pvc1'):      DIR = "/media/Shared Volume/0. Data/2. Data Set/CRCNS.org/Visual Cortex/pvc1/crcns-ringach-data/neurodata/";

    FILE = DIR+'SUAs (%s)'%Data_Set;
    with open(FILE+'.dat', "rb") as f:    SUAs= pickle.load(f); 

    U_No = len(SUAs);
    data=[];
    if(str(iCs).lower()=='all'):
        data.append(ARRAY([],0));
        for u in range(U_No):
            data[-1].Clusters.append(SUAs[u]);

    else:
        for iC in range(len(iCs)):
            data.append(ARRAY([],0));
            for iU in range(len(iCs[iC])):
                data[-1].Clusters.append(SUAs[iCs[iC][iU]]);
    return data;

#generate normal clusters
def Data_Gen(Data_Set, C_Nos):
    Dist = distro.linux_distribution()[0];
    if(Dist.lower()   == 'ubuntu' and Data_Set=='pvc1'):      DIR = os.getenv("HOME")+'/Public/Dataset/CRCNS.org/Visual Cortex/pvc1/crcns-ringach-data/neurodata/';
    elif(Dist.lower() == 'debian' and Data_Set=='pvc1'):      DIR = "/media/Shared Volume/0. Data/2. Data Set/CRCNS.org/Visual Cortex/pvc1/crcns-ringach-data/neurodata/";

    FILE = DIR+'SUAs (%s)'%Data_Set;
    with open(FILE+'.dat', "rb") as f:    SUAs= pickle.load(f);

    U_No = len(SUAs);
    data=[];
    if(str(C_Nos).lower()=='all'):
        data.append(ARRAY([],0));
        for u in range(U_No):
            data[-1].Clusters.append(SUAs[u]);

    else:
        for c in range(len(C_Nos)):
            data.append(ARRAY([],0));
            for u in np.int32(np.random.uniform(0, U_No, C_Nos[c]).round()):
                data[-1].Clusters.append(SUAs[u]);
    return data;
    
