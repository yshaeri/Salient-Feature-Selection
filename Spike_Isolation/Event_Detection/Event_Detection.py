# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 19:35:40 2015

@author: mali
"""
import numpy as np;
import scipy as sp;


class Detection(object):
    Threshold = 0;
    Duration  = 0;
    Mode      = None;
    def __init__(self, Input=[], THD=1.0, Du=0.0, Mode=1.0):
      self.Input     = Input;
      self.Threshold = THD;
      self.Duration  = Du;
      self.Mode      = Mode;

    def Event_Detectoion(self, *args, **kwargs):

        for a in range(min(len(args),4)):
            if(a==0):              self.Input     = args[0];
            elif(a==1):            self.Threshold = args[1];
            elif(a==2):            self.Duration  = args[2];
            elif(a==3):            self.Mode      = args[3];
        count=0;
        INDEX=[];
        End=0;
        Start=0;
        Mask = np.abs(self.Input) >= self.Threshold;Mask=Mask*1;
        if   self.Mode==-1 or self.Mode=='Hard Threshold': self.Mask=Mask;pass;
        elif self.Mode==0  or self.Mode=='Static Detect':
            if self.Input.ndim == 1:
                for k in range(len(self.Input)):
                    if abs(self.Input[k])>=self.Threshold and k>End: INDEX.append(k); End = k+self.Duration;
                self.INDEX = [INDEX,[(x+self.Duration if x+self.Duration < self.Input.shape[0] else self.Input.shape[0]) for x in INDEX]];
            elif self.Input.ndim == 2:
                for n in range(self.Input.shape[1]):
                    for k in range(self.Input.shape[0]):
                        if abs(self.Input[k,n])>=self.Threshold and k>End: INDEX.append(k); End = k+self.Duration;
                    self.INDEX = [INDEX,[[x+self.Duration for x in INDEX]]];# np.array(INDEX)+1
            return self.INDEX;
        elif self.Mode==1  or self.Mode=='Dynamic Detect':
            if self.Input.ndim == 1:
                EoP  = np.where(np.diff(Mask)<0)[0];
                for k in range(len(EoP)):
                    EoM = EoP[k]+self.Duration if EoP[k]+self.Duration< self.Input.shape[0] else self.Input.shape[0];
                    Mask[np.r_[EoP[k]:EoM]] = np.ones(EoM - EoP[k]);
            elif self.Input.ndim == 2:
                for n in range(self.Input.shape[1]):
                    EoP  = np.where(np.diff(Mask[:,n])<0)[0];
                    for k in range(len(EoP)):
                        EoM = EoP[k]+self.Duration if EoP[k]+self.Duration< self.Input.shape[0] else self.Input.shape[0];
                        Mask[np.r_[EoP[k]:EoM],n] = np.ones(EoM - EoP[k]);
            else: pass;
            self.Mask=Mask;
        else: print("Mode", self.Mode, "is not defined"); pass;
        return self.Mask;



