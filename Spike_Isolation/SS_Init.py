#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 14:50:04 2018

@author: m-ali
"""


#%% Import modules from libaries

import sys, os, glob, re, time;
import numpy as np;
import scipy as sp;
import scipy.fftpack, scipy.ndimage, scipy.signal, scipy.cluster, scipy.stats as st;
import pywt;

import csv, pickle, pandas;

import matplotlib, pylab as pyl;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;

import matplotlib;
import matplotlib.pyplot as plt;
import mpl_toolkits as mpl
import mpl_toolkits.axisartist as mplax
from mpl_toolkits.mplot3d import Axes3D;
import matplotlib.gridspec as gridspec
from matplotlib import patheffects 
import pylab as pyl;


import sklearn as sk
import sklearn.cluster, sklearn.neighbors, sklearn.svm, sklearn.neural_network, sklearn.naive_bayes, sklearn.tree;

import sklearn.discriminant_analysis

sys.path.append(Dir);
import Spike_Isolation.Event_Detection.Event_Detection as ED;
import Signal_Generator.SUA_preparation as SUA;
import Spike_Isolation.Feature_Selection.Features_Selection as FS;
import Spike_Isolation.Feature_Extraction.Feature_Extraction as FE;
import Spike_Isolation.Event_Classification.Event_Classification as WC;

#%%
if(IMode=='Generate'):
    ifile  = (0,0);
    
    _U = [(4,0), (3,0), (7,0)]; #[(4,0), (3,0), (7,0), (1,1)];
    ilists = ([0, [_U[iu][0] for iu in range(len(_U))]],  );
    ix = (ifile, ilists);
    Clus, Prop = SUA.Load_Data( ('SUA',''),'pvc1',ix);

    Clus[0].Clusters = [ Clus[ic].Clusters[0] for ic in range(len(_U)) ];
    Clus = [Clus[0]];

    iCs=((25,47,38),(25,47,39),(25,27),(36,47,11));
    Clus += SUA.Unit_Gen('pvc1', iCs);

    #C_Nos = [2]*104+[3]*104+[4]*48;
    C_Nos = [2]*6+[3]*6+[4]*4;
    Clus += SUA.Data_Gen('pvc1', C_Nos);

    ##%% Save data clusters to file
    with open('Clusters - r%d.dat'%(_run+1), "wb") as file:  pickle.dump({'Clus':Clus,}, file);

elif(IMode=='Load'):
    with open('Clusters - r%d.dat'%_run, "rb") as file:	dic= pickle.load(file)
    Clus=dic['Clus'];

elif(IMode=='Batch-load'): # Merge existing data files
    Clus = [];
    for r in range(len(_Runs)):
        with open('Clusters - r%d.dat'%_Runs[r], "rb") as file:	dic= pickle.load(file)
        Clus +=dic['Clus'][(5 if r else 0):]

S_No = len(Clus); #Ch. No (recorsing sites)
C_Nos = [len(Clus[iS].Clusters) for iS in range(S_No)]; #dic['C_Nos']

#%% 
O_No = 2; # No of optimal features

Label =[];
Data  =[];
Pc    =[];
Chance_ML=[];

for iS in range(S_No):
    C_No = C_Nos[iS];

    Label.append([]);
    Data.append([]);
    Pc.append([]);
    Chance_ML.append([]);

    for iC  in range(C_No):
        Label[iS] += [iC]*Clus[iS].Clusters[iC].shape[1];
        Data[iS]  += [Clus[iS].Clusters[iC]]
    Data[iS]   = np.concatenate(Data[iS],1);
    Pc[iS]     = np.array([(np.array(Label[iS])==iC).sum()   for iC in range(C_No)])/len(Label[iS]);
    Chance_ML[iS] = [max(np.delete(Pc[iS],iC).sum(),Pc[iS][iC]) for iC in range(C_No)];        
Chance_MC = np.array([max(Pc[iS]) for iS in range(S_No)]);


#%%    2. Event Detection & Extraction (ED,EX)
SD = ED.Detection();
Thd = 20;
Du  = 10;

Event_Mask=[];
Event_Edge=[];
Bi = [[] for s in range(S_No)];
Bj = [[] for s in range(S_No)];
Ei = [[] for s in range(S_No)];
Ej = [[] for s in range(S_No)];

for s in range(S_No):
    Event_Mask.append([]);
    Event_Edge.append([]);
    C_No = len(Clus[s].Clusters);#Clus[s].Clus_No
    for c in range(C_No):
        Event_Mask[s].append(SD.Event_Detectoion(np.mean(Clus[s].Clusters[c],1),Thd,Du,'Dynamic Detect'));
        Event_Edge[s].append(SD.Event_Detectoion(np.mean(Clus[s].Clusters[c],1),20,40,'Static Detect'));

        b = np.where(np.diff(Event_Mask[s][c])==1);  b=b[0][0]+1 if b[0].shape[0]!=0 else len(Event_Mask[s][c]);
        e = np.where(np.diff(Event_Mask[s][c])==-1); e=e[0][0]+1 if e[0].shape[0]!=0 else len(Event_Mask[s][c]);
        Bj[s].append(b);
        Ej[s].append(e);

#%%




