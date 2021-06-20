#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:14:59 2015

@author: mali
"""

#%%
import sys, os, glob, re, time;

Dir=os.getenv("HOME") + '/Downloads/SFS Codes/';
os.chdir(Dir+'Spike_Isolation/Data_Files/');

files = glob.glob('Clusters - r*.dat')
_Runs = sorted([int(re.findall('\d+',os.path.splitext(file.split("- r")[1])[0])[0]) for file in files]); # sorted([1,2,3,4]);
_run = max(_Runs) if len(_Runs) else 1;
_run = 1;

IMode = ['Generate','Load','Batch-load'][1]; #Input_Mode. Run Mode
Init_Dir=Dir+'Spike_Isolation/';
exec(open(Init_Dir+"/SS_Init.py").read());

IMode = ['Generate','Load','Batch-load'][1]; #Input_Mode. Run Mode ??????

#%%
exec(open(Init_Dir+"/Feature_Extraction/FE_Init.py").read());
exec(open(Init_Dir+"/Feature_Extraction/FE_Plot.py").read(), globals());

#%% Salient Feature Selection (SFS)
exec(open(Init_Dir+"/Feature_Selection/FS_Init.py").read());
exec(open(Init_Dir+"/Feature_Selection/FS_Plot.py").read())

#%% Waveshape Classification (WC)
exec(open(Init_Dir+"/Event_Classification/EC_Init.py").read());
exec(open(Init_Dir+"/Event_Classification/EC_Plot.py").read())




