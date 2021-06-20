#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 23:32:15 2018

@author: m-ali
"""

#%% Definitions (parameters, variables, functions)

iG=0;
iD=[1,2][0];
iI=[1,2][1];

FESs = [((' '  ,1),                         (' ',),         ('Multi-class')),\
        ((' '  ,1),                         (' ',),         ('Multi-label')),\
        ((' '  ,1),                         ('SFS', 1),     ('Multi-label')),\
        ((' '  ,1),                         ('SFS', 2),     ('Multi-label')),\
        ((' '  ,1),                         ('SFS', 3),     ('Multi-label'))];


h = .5;  # step size in the mesh
n_neighbors = 10;
shrinkage =[None, 0.1, 1][0];
C = .1;  # SVM regularization parameter
Classifiers = {
    'GNB':        sk.naive_bayes.GaussianNB(),
    'KNN':        sk.neighbors.KNeighborsClassifier(n_neighbors, weights='uniform'),
    'NC':         sk.neighbors.NearestCentroid(shrink_threshold=shrinkage),
    'DT':         sk.tree.DecisionTreeClassifier(),
    ('ODT', 2,50):WC.ObliqueQDT(2, 50),     # Yang'17
    'NN':         sk.neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
    'LinearSVC':  sk.svm.LinearSVC(C=C),
    'SVC Linear': sk.svm.SVC(kernel='linear', C=C), 
    'SVC RBF':    sk.svm.SVC(kernel='rbf', gamma=0.7, C=C), # Gaussian RBF kernel 
    'SVC Poly':   sk.svm.SVC(kernel='poly', degree=3, C=C),  # polynomial kernel

    ('WD', 1,4,16 ):   WC.WindowDiscrimination(1,4,16),
    ('WD', 2,4,16 ):   WC.WindowDiscrimination(2,4,16),
    ('WD', 1,128,'sequence'):  WC.WindowDiscrimination(1,[],128,'sequence',1),
    ('WD', 1,16,'sequence'):   WC.WindowDiscrimination(1,[],16,'sequence',1),
    ('WD', 1,4,8,'knn-l2'):   WC.WindowDiscrimination(1,4,8,'knn-l2',1,1000),
};

ECs  = ['GNB','KNN','DT',('ODT', 2,50),'LinearSVC',\
        ('WD', 1,4,16 ), ('WD', 2, 4, 16),('WD', 1,16,'sequence'),('WD', 1,128,'sequence')];


for norm in [1,2]:
    Classifiers[('TM','md',norm,1)] = WC.TemplateMatch(1, norm, 'min-distance');
    ECs.append( ('TM','md',norm,1) );


#%% Settings/Configurations

_FESs=FESs;
if(IMode=='Generate'):
    FES_data, FES_T = ({},{},);

    for iFES, FES in enumerate(_FESs):
        FE_mode = FES[0];
        FS_mode = FES[1];
        EC_mode = FES[2];
        FS_Clus, FES_data[FES], FES_T[FES] = ([],[],[],);

        if(FE_mode[0].lower()=='lda'):
            O_No = FE_mode[2];
            for iS in range(S_No):
                T0 = time.time();
                C_No = C_Nos[iS];
                if(FE_mode[1].lower()=='multi-class'):
                    FS_Clus.append( [] );
                    FS_Clus[iS] = sk.discriminant_analysis.LinearDiscriminantAnalysis(n_components=O_No if(O_No>2) else 1)\
                                                          .fit(Data[iS].T, Label[iS]).transform(Data[iS].T);
                    FES_data[FES].append([FS_Clus[iS]]);
                    FES_T[FES].append(time.time()-T0);
                elif(FE_mode[1].lower()=='multi-label'):
                    FS_Clus.append( [[] for x in range(C_No)] );
                    for iC in range(C_No):
                        _label = np.array(Label[iS])==iC;
                        FS_Clus[iS][iC] = sk.discriminant_analysis.LinearDiscriminantAnalysis(n_components=O_No if(O_No>2) else 1)\
                                                                  .fit(Data[iS].T, _label ).transform(Data[iS].T);
                    FES_data[FES].append( FS_Clus[iS] );
                    FES_T[FES].append(time.time()-T0);
        else:
            _FE_Feat, _FE_Clus = FE.Feature_Extraxtion( Clus, FE_mode );
            F_No = len(_FE_Feat.Coef.label);
            for iS in range(S_No):
                T0 = time.time();
                C_No = C_Nos[iS];
                FS_Clus.append( [[] for x in range(C_No)] );
                for iC in range(C_No):
                    for iF in range(F_No):      # iF is Feature Set index and F_No is No of Feature Sets
                        FS_Clus[iS][iC].append(_FE_Clus[iS][iC].feature[iF]);
    
            if(FS_mode[0].lower()=='sfs'):
                O_No = FS_mode[1];
                for iS in range(S_No):
                    T0 = time.time();
                    C_No = C_Nos[iS];
                    OF, SF, FD = FS.Optimal_Feature_Selection( FS_Clus[iS], _FE_Feat, O_No, );
                    FES_data[FES].append( [[] for x in range(C_No)] );
                    for iC  in range(C_No):
                        FS_feat = [[] for ic in range(C_No)];
                        for iO in range(0,O_No):
                            FS_feat = [FS_feat[ic]+[FS_Clus[iS][ic][OF[iC][iO][0]][OF[iC][iO][1]]]     for ic in range(C_No)];
                        FES_data[FES][iS][iC] = np.concatenate(FS_feat,axis=1).T.tolist()
                    FES_T[FES].append(time.time()-T0);
            else:
                for iS in range(S_No):
                    T0 = time.time();
                    C_No = C_Nos[iS];
                    FS_feat = [FS_Clus[iS][ic][0]     for ic in range(C_No)];
                    if(EC_mode.lower()=='multi-class'): FES_data[FES].append( [np.concatenate(FS_feat,axis=1).T.tolist()] );
                    elif(EC_mode.lower()=='multi-label'):FES_data[FES].append( [np.concatenate(FS_feat,axis=1).T.tolist()]*C_No );
                    FES_T[FES].append(time.time()-T0);

#%% Event Classification Comparison

_FESs=FESs;
_ECs=[ECs[iE] for iE in {0,3,5,6,7,8,9,10}];

if(IMode=='Generate'):
    CA, C_T = ({},{},);
    for iFES, FES in enumerate(_FESs):
        CA[FES], C_T[FES] = ({},{},);
        for EC_type in _ECs:
            CA[FES][EC_type]=[];
            C_T[FES][EC_type]=[];
    
    T_=time.time();
    for iS in range(S_No):
        D_No = len(Label[iS]); # No of data-points
        ix = np.r_[:D_No];
        np.random.shuffle(ix);
    
        for iFES, FES in enumerate(_FESs):
            FE_mode = FES[0];
            FS_mode = FES[1];
            EC_mode = FES[2];
    
            C_No = len(FES_data[FES][iS]);
            _CA = [[] for x in range(C_No)];
    
            #_std= [];
            for iE,EC_type in enumerate(_ECs):
                T1 = time.time();
                MULTILABEL = EC_mode.lower()=='multi-label';
                MULTICLASS = EC_mode.lower()=='multi-class';
                CLASSI_En  = (EC_type[0].lower()!='wd' or len(FES_data[FES][iS][0][0])<4) and (EC_type[0].lower()!='odt' or 1<=len(FES_data[FES][iS][0][0])<4);
                if(CLASSI_En):
                    for iC  in range(C_No):
                        if(  C_No==C_Nos[iS]):   _FES_data = FES_data[FES][iS][iC];
                        elif(C_No==1        ):   _FES_data = FES_data[FES][iS][0];
                        else:                    _FES_data = FES_data[FES][iS][0]; print('%s(%d) - Warning! Multi-label classification requires 1 or %d features'%(str(FES), iS, C_No) );
    
                        if( MULTILABEL ):                     _label = np.array(Label[iS])==iC;
                        elif(MULTICLASS ):                    _label = np.array(Label[iS]);
                        train_data, test_data = np.array(_FES_data)[ix[:D_No//2]], np.array(_FES_data)[ix[D_No//2:]];
                        train_label,test_label=              _label[ix[:D_No//2]],              _label[ix[D_No//2:]];
    
                        EC = Classifiers[EC_type].fit(train_data, train_label);
                        predict_label = EC.predict(test_data);
    
                        if(np.ndim(predict_label)==1):  _CA[iC] = np.mean(predict_label==test_label);
                        else:                           _CA[iC] = np.mean([np.mean(predict_label[ic]==(test_label==ic))\
                                                                           for ic in range(len(predict_label))]);
                    if(  MULTILABEL):  CA[FES][EC_type].append( sum(Pc[iS]*_CA) );
                    elif(MULTICLASS):  CA[FES][EC_type].append( np.mean(_CA) );
                C_T[FES][EC_type].append(time.time()-T1);
                print((iS,iFES,iE),':','%.3f'%(time.time()-T1), end='; ');
            print('');

    CA['FBSHT'], C_T['FBSHT'] = ({},{},);
    CA['FBSHT'][('ODT', 2,50)], C_T['FBSHT'][('ODT', 2,50)] = ([],[]);
    
    for iS in range(S_No):
        FBSHT, label = ([],[]);
        T1 = time.time();
        for i in range(Data[iS].shape[1]):
            Coefs = pywt.wavedec(Data[iS][:,i], 'haar', level=4);
            FBSHT.append([Coefs[4].max(), Coefs[4].min(), Coefs[0].max(), Coefs[0].min()]);#Yang&Mason'16
            label.append(Label[iS][i]);
    
        train_data, train_label = np.array(FBSHT), np.array(label);
        ODT = Classifiers[('ODT', 2,50)].fit(train_data, train_label);
        CA['FBSHT'][('ODT', 2,50)].append(sum(ODT.predict(train_data)==label)/len(label));
        C_T['FBSHT'][('ODT', 2,50)].append(time.time()-T1);

    with open('EC_Data - r%d.dat'%(_run+1), "wb") as file:  pickle.dump({'CA':CA,'C_T':C_T,'C_Nos':C_Nos}, file);

    print(time.time()-T_)

elif(IMode=='Load'):
    with open('EC_Data - r%d.dat'%_run, "rb") as file:	dic= pickle.load(file)
    C_Nos = dic['C_Nos']; #[len(Clus[iS].Clusters) for iS in range(S_No)];
    S_No = len(C_Nos);
    CA = dic['CA'];
    C_T= dic['C_T'];

    
    _FESs = list(CA.keys())
    _ECs  = list(CA[_FESs[0]].keys())

elif(IMode=='Batch-load'):
    C_Nos, _CA, _C_T = [],[],[]
    CA, C_T = ({},{},);
    for r in range(len(_Runs)):
        with open('EC_Data - r%d.dat'%_Runs[r], "rb") as file:	dic= pickle.load(file)
        C_Nos += dic['C_Nos'][(5 if r else 0):];
        _CA.append(dic['CA'])
        _C_T.append(dic['C_T'])
    
    for fe in list(_CA[0].keys()):
        C_T[fe], CA[fe] = {},{}
        for ec in list(_CA[0][fe].keys()):
            C_T[fe][ec], CA[fe][ec] = [],[]
            for r in range(len(_CA)):
                C_T[fe][ec]   += _C_T[r][fe][ec][(5 if r else 0):];
                CA [fe][ec]   += _CA [r][fe][ec][(5 if r else 0):];

#%%
