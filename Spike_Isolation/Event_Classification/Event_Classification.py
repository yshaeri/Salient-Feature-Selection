# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:36:20 2017

@author: m-ali
"""

import numpy as np
import scipy as sp;
import itertools
from random import randint
import sys,os
Dir=os.getenv("HOME")+\
'/Documents/Notebooks/Notes/Note/Proposal/Bidirectional_Wireless_multi-channel_recording_&_stimulation_system/\
3._Rec&Stim_System/Recording/Online_Spike_Sorting/Source Codes/Simulation/'
sys.path.append(Dir);

from Spike_Isolation.Event_Classification import Impurity
from Spike_Isolation.Event_Classification.Yang_NSP.dtree_spikes_master.tree_split import tree_split

__all__ = ['WindowDiscrimination', 'TemplateMatch']

class TemplateMatch():
    def __init__(self, *args, **kwargs):
        if(len(args)>0):            self.bs   = args[0];# End with bin/segment size (Resolution)
        else:                       self.bs   = 1;
        if(len(args)>1):            self.norm = args[1]; #L1-norm,L2-norm,L∞,Rho (Correlation in multi dimentional space)
        else:                       self.norm = 2;
        if(len(args)>2):            self.dl = args[2]; #Decision Rule: 'Min-Distance', 'Opt-Distance'
        else:                       self.dl = ['min-distance','opt-distance'][0];
        if(len(args)>3):            self.calc_mode = args[3];
        else:                       self.calc_mode = 1;
    def fit(self, Data, Tag, sample_weight=None):
        Data = np.array(Data);
        self.Labels = sorted(list(set(Tag)));
        C_No   = len(set(Tag));
        self.template = [];
        self.r        = [];
        for iC in self.Labels:
            self.template.append( Data[np.where(Tag==iC)].mean(0) );

        if(self.dl.lower() == 'opt-distance'):
            if([type(x)==bool or type(x)==np.bool_ for x in self.Labels] == [True, True]):C_range = [True];      # 'multi-label'
            else:                                                    C_range = range(C_No); # 'multi-class'
            for iC in C_range:
                if(self.norm==1):       Err =          np.abs(   Data - self.template[iC] ).sum(1);    
                elif(self.norm==2):     Err = np.sqrt( np.square(Data - self.template[iC] ).sum(1) );
                elif(self.norm=='rho'): Rho = np.apply_along_axis(lambda x: np.corrcoef(x,self.template)[0,1], axis=1, arr=Data);

                if(self.norm==1 or self.norm==2):
                    r_max   = np.int16(np.floor(np.max( Err )))+1;
                    R_range = list(np.r_[0:r_max+self.bs:self.bs]);#      list(range(0, r_max,  self.bs));
                elif(self.norm=='rho'):           R_range = list(np.r_[0:1+self.bs:self.bs]);

                if(self.calc_mode==1):
                    if(self.norm==1 or self.norm==2): label_out = ( [list(Err)]*len(R_range) - np.transpose([list(R_range)]*len(Err)) ) <=0;
                    elif(self.norm=='rho'):           label_out = ( [list(Err)]*len(R_range) - np.transpose([list(R_range)]*len(Err)) ) >=0;
                    TP = np.sum( label_out*( Tag==iC)+0, 1);
                    FP = np.sum( label_out*( Tag!=iC)+0, 1);
                    TN = ( Tag!=iC+0 ).sum() - FP;
                    _CA= TP+TN;
                elif(self.calc_mode==2):
                    _CA = [];
                    for r in R_range:
                        if(self.norm==1 or self.norm==2): label_out = Err<=r;
                        elif(self.norm=='rho'):           label_out = Rho>=r;
                        TP = np.sum( label_out*( Tag==iC)+0 );
                        FP = np.sum( label_out*( Tag!=iC)+0 );
                        TN = ( Tag!=iC+0 ).sum() - FP;
                        _CA.append( (TP+TN)/len(Tag) );
                iCA = np.where(_CA>=np.max(_CA))[0].min();
                self.r.append( R_range[iCA] );
        return self;

    def predict(self, Data):
        Data = np.array(Data);
        C_No = len(self.template);
        Tag  = [];
        Err  = [];
        Rho  = [];
        for iC in range(C_No):
            if(  self.norm==1):     Err.append(        np.abs(    Data - self.template[iC] ).sum(1) );
            elif(self.norm==2):     Err.append(np.sqrt(np.square( Data - self.template[iC] ).sum(1)));
            elif(self.norm=='rho'): Rho.append(np.apply_along_axis(lambda x: np.corrcoef(x,self.template[iC])[0,1], axis=1, arr=Data));

        if(self.dl.lower() == 'min-distance'):
            if(self.norm==1 or self.norm==2):     Tag = np.argmin(Err ,axis=0);
            elif(self.norm=='rho'):               Tag = np.argmax(Rho ,axis=0);

        if(self.dl.lower() == 'opt-distance'):
            if([type(x)==bool or type(x)==np.bool_ for x in self.Labels] == [True, True]):C_range = [True];      # 'multi-label'
            else:                                                                         C_range = range(C_No); # 'multi-class'

            for iC in C_range:
                ir = 0 if([type(x)==bool or type(x)==np.bool_ for x in self.Labels] == [True, True]) else iC; #if(len(self.r)==1) elif(len(self.r)==C_No)
                if(self.norm==1 or self.norm==2): label_out = np.array(Err[iC])<=self.r[ir];
                elif(self.norm=='rho'):           label_out = np.array(Rho[iC])>=self.r[ir];
                if([type(x)==bool or type(x)==np.bool_ for x in self.Labels] == [True, True]):Tag = label_out;        # 'multi-label'
                else:                                                    Tag.append(label_out);  # 'multi-class'
        return Tag;

#Fast WD: Use Bayes decision theory to calculate decision boundries (BND)?????
class WindowDiscrimination():
    def __init__(self, *args, **kwargs):
        if(len(args)>0):            self.bs   = args[0];# End with bin/segment size
        else:                       self.bs   = 1;
        if(len(args)>1):            self.bn   = args[1];# Start with number of bins/segments in each feature dimension
        else:                       self.bn   = 8;
        if(len(args)>2):            self.sn   = args[2]; #split no per dimensions
        else:                       self.sn   = -1;
        if(len(args)>3):            self.mode = args[3];#search modes: detailed span, fast (max CA)
        else:                       self.mode = ['min-area','mid-area','max-area','sequence','span','knn-l1','knn-l2'][0];
        if(len(args)>4):            self.calc_mode = args[4];
        else:                       self.calc_mode = 0;
        if(len(args)>5):            self.wd  = args[5];# Window discriminators per stage
        else:                       self.wd  = [-1,5000][0];

        self.Labels = [];
        self.Bounds = [];
        self.AlternativeBounds = [];

    def fit(self, Data, Tag, sample_weight=None):
        Data   = np.array(Data);
        Tag    = np.array(Tag);

        self.Labels = sorted(list(set(Tag)));
        self.Bounds = [];
        self.AlternativeBounds = [];

        C_No   = len(set(Tag));
        O_No   = Data.shape[1]

        if([type(x)==bool or type(x)==np.bool_ for x in self.Labels] == [True, True]):  C_range = [True];      # 'multi-label'
        else:                                                                           C_range = range(C_No); # 'multi-class'

        if(self.bs=='auto'):          bs_End = Data.std(0)/10;#[Data[:,iO].std()/100 for iO in range(O_No)]; #??
        elif(np.isscalar(self.bs)):   bs_End = [self.bs]*O_No;
        elif(len(self.bs)==1):        bs_End = [self.bs[0]]*O_No;
        elif(len(self.bs)==O_No):     bs_End = self.bs;
        else:                         print('Problem with bs parameter value');return -1;     

        x_min, x_max = np.int16( np.ceil( np.max([Data.min(0), Data.mean(0)-4*Data.std(0)],0)) ) - 1,\
                       np.int16( np.floor(np.min([Data.max(0), Data.mean(0)+4*Data.std(0)],0)) ) + 1;

        if(  self.mode.lower() in ['span','sequence']):                                  bs = [self.bs]*O_No;
        elif(self.mode.lower() in ['min-area','mid-area','max-area','knn-l1','knn-l2']): bs = 2**np.ceil(np.log2((x_max-x_min)/self.bn));

        if(self.mode.lower() in ['min-area','mid-area','max-area','span','knn-l1','knn-l2']):
            X_range = [];
            for iO in range(O_No):
                xbl_range = np.r_[np.floor(x_min[iO]/bs[iO])*bs[iO]: x_max[iO]+bs[iO]: bs[iO]]
                X_range.append([[0,0]]);
                for x_bl in xbl_range:
                    xbu_range = np.r_[x_bl: x_max[iO]:  bs[iO]]+bs[iO];
                    for x_bu in xbu_range:  X_range[iO] +=[(x_bl,x_bu)];

        if(self.mode.lower() in ['min-area','mid-area','max-area','knn-l1','knn-l2']):
            Xrange = '';
            for iO in range(O_No): Xrange += 'X_range[%d],'%iO;
            Xrange = np.array(eval('list(itertools.product(%s))'%(Xrange[:-1])));

            for iC in C_range:
                while(True):
                    _CA, label_out = [], [];
                    if(self.calc_mode==0):
                        label_out = np.array([[1]*len(Tag)]*len(Xrange));
                        for iO in range(O_No):
                            _bnd = [np.transpose([Xrange[:,iO,0]]*len(Data)),\
                                    np.transpose([Xrange[:,iO,1]]*len(Data))];
                            _data = [Data[:,iO]]*len(Xrange);
                            label_out *= (_bnd[0] < _data)&(_data < _bnd[1]);
                        TP = np.sum( label_out*( Tag==iC)+0 ,1);
                        FP = np.sum( label_out*( Tag!=iC)+0 ,1);
                        TN = ( Tag!=iC+0 ).sum() - FP;
                        _CA = (TP+TN)/len(Tag);
                    elif(self.calc_mode==1):
                        _label_out = np.zeros((O_No,len(Xrange),len(Data)),bool)
                        for iO in range(O_No):
                            for ib,bnd in enumerate(Xrange):
                                _label_out[iO,ib,:] = (bnd[iO][0] < Data[:,iO])&(Data[:,iO] < bnd[iO][1]);

                        label_out = np.array([[1]*len(Tag)]*len(Xrange));
                        for iO in range(O_No): label_out *= np.array(_label_out[iO]);

                        TP = np.sum( label_out*( Tag==iC)+0 ,1);
                        FP = np.sum( label_out*( Tag!=iC)+0 ,1);
                        TN = ( Tag!=iC+0 ).sum() - FP;
                        _CA = (TP+TN)/len(Tag);
                    elif(self.calc_mode==2):
                        for bnd in Xrange:
                            label_out.append( [1]*len(Data[:,0]) );
                            for iO in range(O_No):
                                label_out[-1] *= (bnd[iO][0] < Data[:,iO])&(Data[:,iO] < bnd[iO][1]);
                        TP = np.sum( label_out*( Tag==iC)+0 ,1);
                        FP = np.sum( label_out*( Tag!=iC)+0 ,1);
                        TN = ( Tag!=iC+0 ).sum() - FP;
                        _CA = (TP+TN)/len(Tag);
                    elif(self.calc_mode==3):
                        for bnd in Xrange:
                            label_out = [1]*len(Data[:,0]);
                            for iO in range(O_No):
                                label_out *= (bnd[iO][0] < Data[:,iO])&(Data[:,iO] < bnd[iO][1]);
                            TP = np.sum( label_out*( Tag==iC)+0 );
                            FP = np.sum( label_out*( Tag!=iC)+0 );
                            TN = ( Tag!=iC+0 ).sum() - FP;
                            _CA.append( (TP+TN)/len(Tag) );

                    iCA = np.where(_CA>=np.max(_CA))[0];
                    np.random.shuffle(iCA);
                    if(self.wd>0): iCA = iCA[:self.wd]

                    if(self.mode.lower()[:3]=='knn'):
                        K = self.bn; #Number of neighbours
                        C = self.sn; #Number of Clusters
                        X = np.array([[Xrange[iCA][i].ravel() for i in range(len(iCA))] for j in range(len(iCA))]);
                        if(self.mode.lower()=='knn-l1'):
                            d = np.sum( abs(X - np.transpose(X,axes=(1,0,2)))   , 2);
                            dnn = np.sort(d)[:,:K].sum(1);


                        elif(self.mode.lower()=='knn-l2'):
                            d = np.sum(    (X - np.transpose(X,axes=(1,0,2)))**2, 2)**.5;
                            dnn = (((np.sort(d)[:,:K]**2).sum(1))**.5);

                        idn = np.where(np.array(dnn)==min(dnn))[0];
                        np.random.shuffle(idn);
                        idn = np.argsort(d[idn])[:C,:K];
                        iCA =  np.unique(iCA[ idn ].ravel());
                    #Choosing WDs based on their area/volume
                    elif(self.mode.lower() in ['min-area','mid-area','max-area']):
                        Area = np.product( np.diff(Xrange[iCA])[:,:,0],1);
                        if(self.mode.lower()=='max-area'):      iCA = iCA[ np.argsort(Area)[:-1-self.sn:-1] ];
                        elif(self.mode.lower()=='min-area'):    iCA = iCA[ np.argsort(Area)[:self.sn] ];
                        elif(self.mode.lower()=='mid-area'):    iCA = iCA[ np.argsort(Area)[(len(iCA)-self.sn)//2:(len(iCA)+self.sn)//2] ];

                    if(np.prod([bs[iO]<= bs_End[iO] for iO in range(O_No)])):    break;
                    for iO in range(O_No):
                        if(bs[iO]> bs_End[iO]): bs[iO] = bs[iO]/2;

                    _bs = '';
                    for iO in range(O_No): _bs += '[-bs[%d],0,bs[%d]],[-bs[%d],0,bs[%d]],'%(iO,iO,iO,iO);
                    _bs = eval('list(itertools.product(%s))'%(_bs));
                    _bs = np.reshape(_bs,(len(_bs),O_No,2));

                    X_range = np.transpose( [_bs.tolist()]*len(iCA), (1,0,2,3) ) + np.array( [Xrange[iCA]]*len(_bs) );
                    X_range = np.concatenate(X_range);
                    X_range = X_range.reshape((X_range.shape[0],X_range.shape[1]*X_range.shape[2])).tolist();
                    X_range = list(set(map( tuple,X_range)));
                    Xrange  = np.reshape(X_range,(len(X_range),O_No,2));

                self.AlternativeBounds = Xrange[iCA];
                iCA = iCA[np.argmin(np.product(np.diff(Xrange[iCA]),1))];
                self.Bounds.append( Xrange[iCA].tolist() );
        elif(self.mode.lower()=='sequence'):
            for ic, iC in enumerate(C_range):
                self.Bounds.append([]);
                for iO in range(O_No):
                    label_out = [];
                    Bnd = [];
                    for x_bl in list(np.r_[np.int(np.floor(x_min[iO]/bs[iO])*bs[iO]): x_max[iO]: bs[iO]]):
                        for x_bu in list(np.r_[x_bl+bs[iO]:x_max[iO]+bs[iO]:bs[iO]]):
                            label_out.append( (x_bl<= Data[:,iO])&(Data[:,iO] <=x_bu) );
                            Bnd.append([x_bl,x_bu]);
                    if(self.calc_mode==0):
                        _label = [1]*len(Data[:,iO]);
                        for io in range(iO):
                            _label *= (self.Bounds[ic][io][0]<= Data[:,io])&(Data[:,io] <=self.Bounds[ic][io][1]);
                        label_out = np.array(label_out)*_label;

                        TP = np.mean( label_out*(Tag==iC),1 );
                        FP = np.mean( label_out*(Tag!=iC),1 );
                        TN = np.mean( Tag!=iC ) - FP;
                        _CA = (TP + TN);

                        iCA = np.where(_CA>=np.max(_CA))[0];
                        Area = np.diff(np.array(Bnd)[iCA])[:,0];
                        iCA = iCA[ np.argmax(Area) ];
                        self.Bounds.append( [Bnd[iCA]] );

                    elif(self.calc_mode==1):
                        if(iO==0):  B_No = 1;
                        else:       B_No = len(self.Bounds[ic]);
                        _label = np.array([[1]*len(Data[:,iO])]*B_No);
                        for io in range(iO):
                            for ib in range(B_No):
                                _label[ib] *= (self.Bounds[ic][ib][io][0]<= Data[:,io])&(Data[:,io] <=self.Bounds[ic][ib][io][1]);

                        TP = np.tensordot( np.array(label_out)*(Tag==iC)+0, _label+0, axes=[1, 1])
                        FP = np.tensordot( np.array(label_out)*(Tag!=iC)+0, _label+0, axes=[1, 1])
                        TN = (Tag!=iC).sum() - FP;
                        _CA = (TP + TN)/len(Tag);

                        iCA = np.array(np.where(_CA>=np.max(_CA)));
                        if(iO==0):  self.Bounds[ic] = [[Bnd[ica]] for ica,ib in iCA.T];
                        else:       self.Bounds[ic] = [self.Bounds[ic][ib]+[Bnd[ica]] for ica,ib in iCA.T];
                        iBs = np.argsort(np.prod(np.diff(self.Bounds[ic]),axis=1)[:,0])[:-1-self.sn:-1];
                        self.Bounds[ic] = np.array(self.Bounds[ic])[iBs].tolist();
                if(self.calc_mode==1):
                    iB = np.argmin(np.prod(np.diff(self.Bounds[ic]),axis=1));
                    self.AlternativeBounds = [self.Bounds[ic][ib] for ib in range(len(self.Bounds[ic])) if ib!=iB];
                    self.Bounds[ic] = self.Bounds[ic][iB];

        elif(self.mode.lower()=='bayes'):pass;
        elif(self.mode.lower()=='auto'):pass;#auto-range bs based on std
        elif(self.mode.lower()=='span'):
            for iC in C_range:
                label_out = [];
                Bnd       = [];
                for iO in range(O_No):
                    label_out.append([]);
                    Bnd.append([]);

                    for ib,[x_bl,x_bu] in enumerate(X_range[iO]):
                        label_out[-1].append( (x_bl<= Data[:,iO])&(Data[:,iO] <=x_bu) );
                        Bnd[-1].append([x_bl,x_bu]);

                if(self.calc_mode==0):
                    if(O_No==1):
                        TP = np.tensordot( np.array(label_out[0])+0, (np.array([Tag])==iC)+0, axes=[1, 1])/len(Tag);
                        TN = np.tensordot(~np.array(label_out[0])+0, (np.array([Tag])!=iC)+0, axes=[1, 1])/len(Tag);
                    elif(O_No>1):
                        CA_arg = "";
                        for iO in range(1,O_No):    CA_arg += ", np.array( label_out[%d]),[%d,0]"%(iO,iO+1);
                        TP = eval('np.einsum(np.array( label_out[0])*(Tag==iC)+0,[1,0]'+CA_arg + ')')/len(Tag);
                        FP = eval('np.einsum(np.array( label_out[0])*(Tag!=iC)+0,[1,0]'+CA_arg + ')')/len(Tag);
                        TN = (np.array( Tag )!=iC).sum()/len(Tag) - FP;

                elif(self.calc_mode==1):
                    Xrange = '';
                    for iO in range(O_No): Xrange += 'X_range[%d],'%iO;
                    Xrange = np.array(eval('list(itertools.product(%s))'%(Xrange[:-1])));

                    _label_out = np.zeros((O_No,len(Xrange),len(Data)),bool)
                    for iO in range(O_No):
                        for ib,bnd in enumerate(Xrange):
                            _label_out[iO,ib,:] = (bnd[iO][0] < Data[:,iO])&(Data[:,iO] < bnd[iO][1]);

                    label_out = np.array([[1]*len(Tag)]*len(Xrange));
                    for iO in range(O_No): label_out *= np.array(_label_out[iO]);

                    TP = np.sum( label_out*( Tag==iC)+0 ,1);
                    FP = np.sum( label_out*( Tag!=iC)+0 ,1);
                    TN = ( Tag!=iC+0 ).sum() - FP;
                    _CA = (TP+TN)/len(Tag);

                elif(self.calc_mode==2):
                    pass;

                _CA = TP + TN;

                if(O_No==1):   self.Bounds.append( [Bnd[0][_CA.argmax()]] );
                elif(O_No>1):
                    iB = np.unravel_index(_CA.argmax(),_CA.shape);
                    self.Bounds.append( [Bnd[iO][iB[iO]] for iO in range(O_No)] );
        return self;

    def predict(self, Data):
        Data = np.array(Data);
        C_No = len(self.Bounds);
        O_No = Data.shape[1];
        Tag= [];

        for iC in range(C_No):
            label_out = [1]*len(Data[:,0]);
            for iO in range(O_No):
                label_out *= (self.Bounds[iC][iO][0]<= Data[:,iO])&(Data[:,iO] <=self.Bounds[iC][iO][1]);
            if(C_No==1):            Tag = label_out;
            elif(C_No>1):           Tag.append(label_out);
        return Tag;





# No of trees=6, No of clusters, No of perturbance(5,10,20,50), metric=gini, entropy(information gain)
"""see https://github.com/KDercksen/pyblique"""
class ObliqueClassifier:
    def __init__(self, metric, Rs=2, Np=10):
        self.metric = metric;
        try:                                        self.Rs = np.abs(np.round(Rs));
        except AttributeError or ValueError:        self.Rs = None;
        except:                                     self.Rs = None;
        self.Np = Np;
        self.tree = {};

    def fit(self, data, label):
        _train = np.concatenate((data,np.transpose([label])),axis=1)
        self.tree = self.__create_decision_tree( _train )
        return self;

    def predict(self, data):
        label = [];
        for record in data:
            cls = self.tree
            while type(cls) is dict:
                Coefs = cls["Coefs"];
                v = self.__checkrel(record, Coefs) > 0;
                if v:	cls = cls["high"];
                else:	cls = cls["low"];
            label.append(cls);
        return label;

    def __create_decision_tree(self, _train):
        train_data =_train[:,:-1]; #train_label=_train[:, -1];
        if len(_train) == 0:	return -1;

        isleaf, leaf = self.__is_leaf_node(_train);
        if isleaf:            return leaf;#Class label
        else:
            n_attrs = train_data.shape[1];# _train.shape[1] - 1
            #First, it finds the best axis-parallel split of T (train data) at a node before looking for an oblique split.
            result = np.array([self.__best_split(_train, attr) for attr in range(n_attrs)]);
            index, split = min(enumerate(result), key=lambda x: x[1][1])

            # in order to make this oblique, we first have to build a vector to enable the linear combination split
            Coefs = np.zeros((len(_train[0]),));#np.zeros((len(train_data[0])+1,));
            Coefs[-1] = -split[0]
            Coefs[index] = 1; #oblique line coefs
            low, high = self. __split_data(_train, Coefs);

            imp = Impurity.impurity([low[:, -1], high[:, -1]], self.metric);

            for r in range(self.Np):
                attr = randint(0, len(train_data[0]));
                imp, Coefs = self.__perturb(_train, Coefs, attr, imp);

            # Splitter at this node is calculated, go to next nodes
            tree = {"Coefs": Coefs}
            low, high = self. __split_data(_train, Coefs)
            subtree_low = self.__create_decision_tree(low)
            tree["low"] = subtree_low
            subtree_high = self.__create_decision_tree(high)
            tree["high"] = subtree_high;
        return tree

    def __best_split(self, data, attr):
        # Will return a tuple of (split test, split value).
        splits = np.convolve( np.sort(data[:, attr]), np.repeat(1.0, 2)/2 )[1:-1];

        split_evals = {}
        for s in splits:
            cond = data[:, attr] <= s
            left, right = data[cond], data[~cond]
            split_evals[s] = Impurity.impurity([left[:, -1], right[:, -1]], self.metric);#amount of impurity
        return min(split_evals.items(), key=lambda x: x[1])

    def __split_data(self, _train, Coefs):
        high = np.zeros(_train.shape);
        low = np.zeros(_train.shape);
        ihigh, ilow = 0, 0
        for record in _train:
            v = self.__checkrel(record[:-1], Coefs) > 0;
            if v:    high[ihigh] = record;  ihigh += 1;
            else:    low[ilow]   = record;  ilow  += 1;
        high = high[:ihigh];
        low  = low [:ilow];
        return low, high

    def __is_leaf_node(self, _train):
        # Returns true/false and the class label (useful if this was a leaf)
        labels = _train[:, -1]
        data   = _train[:,:-1]; #train_label=_train[:, -1];
        return all(label == labels[0] for label in labels) or np.all(data == data[0]), labels[0];

    def __checkrel(self, record, Coefs):
        return np.sum(np.multiply(record, Coefs[:-1])) + Coefs[-1]

    def __perturb(self, _train, Coefs, attr, imp):
        # first calculate all values of U with the current value in Coefs for attr
        us = np.array(sorted([[(Coefs[attr]*record[attr] -self.__checkrel(record[:-1], Coefs))/record[attr]\
                               ] for record in _train if(record[attr])]));

        # now find the best of these splits...
        splits = np.convolve(np.sort(us[:, 0]), [ 0.5,  0.5])[1:-1];
        if(len(splits)==0): return imp, Coefs;#splits = us+[-.5,0,.5]

        amvalues = {}
        for s in set(splits):
            _Coefs_new = np.array(Coefs);
            _Coefs_new[attr] = s;
            _Coefs_new = _Coefs_new/np.abs(_Coefs_new[:-1]).max();
            if(self.Rs!=None): _Coefs_new[:-1] = np.round(_Coefs_new[:-1]*self.Rs)/self.Rs
            _Coefs_new[np.where(_Coefs_new[:-1]==-1)[0]]=-1+2**(-self.Rs+1);

            low, high = self.__split_data(_train, _Coefs_new)
            _imp = Impurity.impurity([low[:, -1], high[:, -1]], self.metric);#amount of impurity
            amvalues[s] = (_imp, _Coefs_new)
        imp_new, Coefs_new = min(amvalues.values(), key=lambda x: x[0])
        if imp_new < imp:       return imp_new, Coefs_new;
        else:                   return imp, Coefs;



#EC = WC.ObliqueQDT()
#http://www.egr.msu.edu/amsac/nsp.htm
class ObliqueQDT:
    def __init__(self, Rs=2, Np=50):
        self.Rs = Rs;
        self.Np = Np;
        self.coef = {};
        self.Leaf_label = {};
        self.Label_set = [];

    def fit(self, Data, Tag):
        self.Label_set = np.unique(Tag);
        self.coef, self.Leaf_label = tree_split(np.array(Data),np.array(Tag),self.Rs,self.Np );
        return self;

    def predict(self, Data):
        N = len(Data);
        Nclass= len(self.Label_set);
        depth = int(np.ceil(np.log2(Nclass)));

        label = np.array( [None]*N );
        for i in range(N):
            inode = 0;
            for idepth in range(1,depth+1):
                V = sum(np.concatenate([Data[i,:],[1]])*self.coef[idepth][inode]);
                if V >= 0:  child=0;
                else:       child=1;
                inode = 2*inode +child;

                if idepth == depth -1:
                    if self.Leaf_label[1][inode] != None:
                        label[i] = self.Leaf_label[1][inode];
                        break;
                if idepth == depth:
                    label[i] = self.Leaf_label[2][inode];

        return label;

