"""
Created on Fri Jun  5 20:14:59 2015

@author: mali
"""



#%% Definitions (parameters, variables, functions)
_Color = ['c','y','m','r','g','b','orange','k']
Xforms = ['Derivatives','DWT',' ']; #'SWT',


#%% Settings/Configurations

Mother_Func = 'db1';
level = 4;
xform = Xforms[-1];

##%% Settings/Configurations

O_No = 2; # No of optimal features

#%%

if(IMode=='Generate'):
    FE_Feat, FE_Clus, FS_Clus, OFs, T_SFS, CAG, CAK, CAG_OF, CD_mean,CD_rms, CD_peak, Fstat_mean, Fstat_peak, Fstat1, Fstat2  = ({},{},{},{},{},{},{},{},{},{},{},{},{},{},{});

    for xform in Xforms:
        T0=time.time()
        if(xform[1:3].lower()=='wt'):      FE_mode = (xform, Mother_Func,level);
        elif(xform.lower()=='derivatives'):FE_mode = ('Derivatives',3);
        else:                              FE_mode = (xform,1);

        _FE_Feat, _FE_Clus = FE.Feature_Extraxtion( Clus, FE_mode);
        F_No = len(_FE_Feat.Coef.label);

        FS_Clus[FE_mode],OFs[FE_mode],T_SFS[FE_mode], CAG[FE_mode],CAK[FE_mode],CAG_OF[FE_mode],CD_mean[FE_mode],CD_rms[ FE_mode],CD_peak[FE_mode], Fstat_mean[FE_mode], Fstat_peak[FE_mode], Fstat1[FE_mode],  Fstat2[FE_mode] = ([],[],[],[],[],[],[],[],[],[],[],[],[])

        for iS in range(S_No):
            C_No = C_Nos[iS];

            FS_Clus[FE_mode].append( [[] for x in range(C_No)] );
            for iC in range(C_No):
                for iF in range(F_No):      # iF is Feature Set index and F_No is No of Feature Sets
                    FS_Clus[FE_mode][iS][iC].append(_FE_Clus[iS][iC].feature[iF]);
            T1 = time.time();
            OF, SF, FD = FS.Optimal_Feature_Selection( FS_Clus[FE_mode][iS], _FE_Feat, O_No, );
            
            CD_peak[FE_mode].append( np.abs(   np.log(SF))[:,:,0].max(axis=1)     );
            CD_mean[FE_mode].append( np.abs(   np.log(SF))[:,:,0].mean(axis=1)     );
            CD_rms[ FE_mode].append( np.square(np.log(SF))[:,:,0].mean(axis=1)**.5 );

            T_SFS[FE_mode].append( time.time()-T1 );
            FE_data = [];
            FS_data = [[] for iC in range(C_No)];

            for iC  in range(C_No):
                FE_feat   = [];
                for iFS in range(0,F_No):  FE_feat += list( FS_Clus[FE_mode][iS][iC][iFS] );
                FE_data     += list(np.transpose(FE_feat));

                FS_feat = [[] for ic in range(C_No)];

                for iO in range(0,O_No):
                    FS_feat = [FS_feat[ic]+[FS_Clus[FE_mode][iS][ic][OF[iC][iO][0]][OF[iC][iO][1]]] \
                                        for ic in range(C_No)];
                FS_data[iC] = np.concatenate(FS_feat,axis=1).T.tolist();


            FE_data_ = [np.array(FE_data)[np.where(np.array(Label[iS])==i)[0]]  for i in range(C_No)]
            
            mu  = np.transpose([np.mean(FE_data,0)]);                                       #Overall Mean Vector
            mu_ = [np.transpose([np.mean(FE_data_[i],0)]) for i in range(C_No)];
            Sigma_ = [np.cov(FE_data_[i].T) for i in range(C_No)];                          #Covariance Matrix (or Scatter Matrix)

            d_B = np.sum([Pc[iS][i]*np.outer(mu_[i]-mu,mu_[i]-mu) for i in range(C_No)],0); #Between-class variability
            d_W = np.sum([Pc[iS][i]*Sigma_[i] for i in range(C_No)],0);                     #Within-class variability

            S  = np.linalg.inv(d_W).dot(d_B);           #(d_B.dot(np.linalg.inv(d_W))).T      #Class separation
            Fstat1[FE_mode].append( np.abs(S.diagonal()).sum() )
            Fstat2[FE_mode].append( np.abs(S.diagonal()).mean() ) 
            Fstat_mean[FE_mode].append( np.abs( d_B.diagonal()/d_W.diagonal() ).mean() )
            Fstat_peak[FE_mode].append( np.abs( d_B.diagonal()/d_W.diagonal() ).max() )

            CAG[FE_mode].append(   [[] for iC in range(C_No+1)] );
            CAG_OF[FE_mode].append([[] for iC in range(C_No  )] );
    
            #Multi Dimensional bayes
            GNB = sk.naive_bayes.GaussianNB().fit(FE_data, Label[iS]);
            CAG[FE_mode][iS][0]=sum(GNB.predict(FE_data)==np.array(Label[iS]))/len(Label[iS]);
    
            for iC in range(C_No):
                lbl = np.array(Label[iS])==iC;
    
                GNB = sk.naive_bayes.GaussianNB().fit(FE_data, lbl);
                CAG[FE_mode][iS][iC+1] = sum(GNB.predict(FE_data)==lbl)/len(lbl);
    
                #O_No Dimensional bayes
                GNB = sk.naive_bayes.GaussianNB().fit(FS_data[iC], lbl);
                CAG_OF[FE_mode][iS][iC]=sum(GNB.predict(FS_data[iC])==np.array(lbl))/len(lbl);
        print('FE: %s, elapsed time: %ds'%(xform, time.time()-T0));
##%%
    Thd = 20;
    Du  = 10;
    import Spike_Isolation.Event_Detection.Event_Detection as ED;
    SD = ED.Detection();
    
    CAG_MD, CAG_ZCF, CAG_STE, CAG_SDE, CAG_FSDE, CAG_DDsE, CAG_EDF, CAG_FBSHT = ([],[],[],[],[],[],[],[])
    for iS in range(S_No):
        Event_Mask = SD.Event_Detectoion(Data[iS], Thd, Du, 'Dynamic Detect');
        ZCF, MD ,TCF, STE, SDE, FSDE, DDsE, EDF, FBSHT, label = ([],[],[],[],[],[],[],[],[],[])
        for i in range(Data[iS].shape[1]):
            ix = np.where(Event_Mask[:,i])[0];
            if(ix.shape[0]>Du):
                iM  = ix.min() + np.min((Data[iS][ix,i].argmin(),Data[iS][ix,i].argmax()));
                iM2 = ix.min() + np.max((Data[iS][ix,i].argmin(),Data[iS][ix,i].argmax()));
                iZ  = np.append(ix.min() + np.where((np.sign(Data[iS][ix[:-1],i])*np.sign(Data[iS][ix[1:],i]))<=0)[0], ix.max())+1;
                MD.append( [Data[iS][ix.min():iM   ,i].sum(), Data[iS][iM:ix.max()   -15,i].sum()]);#Li'14
                ZCF.append([Data[iS][ix.min():iZ[0],i].sum(), Data[iS][iZ[0]:ix.max()-15,i].sum()]);#Kamboh'13
            
                TCF.append([ix.min()+ np.where((np.sign(Data[iS][ix[:-1],i]-Thd)*np.sign(Data[iS][ix[1:],i]-Thd))<=0)[0],\
                            ix.min()+ np.where((np.sign(Data[iS][ix[:-1],i]+Thd)*np.sign(Data[iS][ix[1:],i]+Thd))<=0)[0]]);#Rodriguez-Perez'14
                STE.append( [Data[iS][ix,i].max(), ix.min()+Data[iS][ix,i].argmax(), Data[iS][ix,i].min(), ix.min()+Data[iS][ix,i].argmin()]);#Rodriguez-Perez'14
                SDE.append( [Data[iS][ix,i].ptp(), np.diff(Data[iS][ix,i]).max(), np.diff(Data[iS][ix,i]).min()]);#Yang'08,Chae'09
                FSDE.append([np.diff(Data[iS][ix,i]).max(), np.diff(Data[iS][ix,i]).min(),\
                             np.diff(np.diff(Data[iS][ix,i])).max(), np.diff(np.diff(Data[iS][ix,i])).min()]);#Paraskevopoulou'13,Paraskevopoulou'14
                DDsE.append([np.min(Data[iS][ix[:-1],i]-Data[iS][ix[1:],i]),  np.max(Data[iS][ix[:-1],i]-Data[iS][ix[1:],i]),\
                             np.min(Data[iS][ix[:-3],i]-Data[iS][ix[3:],i]),  np.max(Data[iS][ix[:-3],i]-Data[iS][ix[3:],i]),\
                             np.min(Data[iS][ix[:-7],i]-Data[iS][ix[7:],i]),  np.max(Data[iS][ix[:-7],i]-Data[iS][ix[7:],i])]);#Zamani'14
    
                i0  = ix.min() if iM2-ix.min()>1 else ix.min()-2;
                minDDec = np.abs(np.diff(Data[iS][ix.min()  :iM,i])).max() if iM-ix.min()>1 else\
                          np.abs(np.diff(Data[iS][ix.min()-2:iM,i])).max() if    ix.min()>1 else\
                          np.abs(np.diff(Data[iS][:2,i])).max();
    
                minDInc = np.abs(np.diff(Data[iS][iM  :iM2,i])).max() if iM2-iM>1 else\
                          np.abs(np.diff(Data[iS][iM-2:iM2,i])).max() if     iM>1 else\
                          np.abs(np.diff(Data[iS][:2,i])).max();
    
                EDF.append( [Data[iS][ix,i].min(), Data[iS][ix,i].max(), minDDec, minDInc] ); #Liu'18
    
                Coefs = pywt.wavedec(Data[iS][:,i], 'haar', level=4)
                FBSHT.append([Coefs[4].max(), Coefs[4].min(), Coefs[0].max(), Coefs[0].min()]);#Yang&Mason'16
                label.append(Label[iS][i]);

        GNB     = sk.naive_bayes.GaussianNB().fit(np.array(STE  )[::2], label[::2]);   CAG_STE  .append(sum(GNB.predict( np.array(STE  )[1::2])==label[1::2])/len(label[1::2]));
        GNB     = sk.naive_bayes.GaussianNB().fit(np.array(SDE  )[::2], label[::2]);   CAG_SDE  .append(sum(GNB.predict( np.array(SDE  )[1::2])==label[1::2])/len(label[1::2]));
        GNB     = sk.naive_bayes.GaussianNB().fit(np.array(FSDE )[::2], label[::2]);   CAG_FSDE .append(sum(GNB.predict( np.array(FSDE )[1::2])==label[1::2])/len(label[1::2]));
        GNB     = sk.naive_bayes.GaussianNB().fit(np.array(EDF  )[::2], label[::2]);   CAG_EDF  .append(sum(GNB.predict( np.array(EDF  )[1::2])==label[1::2])/len(label[1::2]));
        GNB     = sk.naive_bayes.GaussianNB().fit(np.array(DDsE )[::2], label[::2]);   CAG_DDsE .append(sum(GNB.predict( np.array(DDsE )[1::2])==label[1::2])/len(label[1::2]));
        GNB     = sk.naive_bayes.GaussianNB().fit(np.array(ZCF  )[::2], label[::2]);   CAG_ZCF  .append(sum(GNB.predict( np.array(ZCF  )[1::2])==label[1::2])/len(label[1::2]));
        GNB     = sk.naive_bayes.GaussianNB().fit(np.array(MD   )[::2], label[::2]);   CAG_MD   .append(sum(GNB.predict( np.array(MD   )[1::2])==label[1::2])/len(label[1::2]));
        GNB     = sk.naive_bayes.GaussianNB().fit(np.array(FBSHT)[::2], label[::2]);   CAG_FBSHT.append(sum(GNB.predict( np.array(FBSHT)[1::2])==label[1::2])/len(label[1::2]));

    FES_No = [ len(MD[0]), len(ZCF[0]), len(STE[0]), len(SDE[0]), len(FSDE[0]), len(DDsE[0]), len(EDF[0]), len(FBSHT[0])]#, len(FS_Clus[(' ',1)][0][0][0]) ]; #[2, 2, 4, 3, 4, 6, 4, 4]
    with open('FE_Data - r%d.dat'%(_run+1), "wb") as file:
        pickle.dump({'T_SFS':T_SFS, 'CAG':CAG, 'CAG_OF':CAG_OF, 'CD_mean':CD_mean, 'CD_rms':CD_rms, 'CD_peak':CD_peak, 'Fstat_mean':Fstat_mean, 'Fstat_peak':Fstat_peak, 'CAG_FEs':[CAG_MD, CAG_ZCF, CAG_STE, CAG_SDE, CAG_FSDE, CAG_DDsE, CAG_EDF, CAG_FBSHT],'O_No':O_No, 'FES_No':FES_No}, file);

elif(IMode=='Load'):
    with open('FE_Data - r%d.dat'%_run, "rb") as file:	dic= pickle.load(file)
    T_SFS=dic['T_SFS'];
    CAG=dic['CAG'];
    CAG_OF=dic['CAG_OF'];
    CD_mean=dic['CD_mean'];
    CD_rms=dic['CD_rms']
    CD_peak=dic['CD_peak'];
    Fstat_mean=dic['Fstat_mean'];
    Fstat_peak=dic['Fstat_peak']
    O_No=dic['O_No'];
    FES_No = dic['FES_No'];
    [CAG_MD, CAG_ZCF, CAG_STE, CAG_SDE, CAG_FSDE, CAG_DDsE, CAG_EDF, CAG_FBSHT]=dic['CAG_FEs'];

elif(IMode=='Batch-load'): # Merge existing data files
    _T_SFS, _CAG, _CAG_OF, _CD_mean, _CD_rms, _CD_peak, _O_No, _Fstat_mean, _Fstat_peak = [],[],[],[],[],[],[],[],[],[],[]
    CAG_FEs = [[] for f in range(8)]
    T_SFS, CAG, CAG_OF, CD_mean, CD_rms, CD_peak, Fstat_mean, Fstat_peak = ({},{},{},{},{},{},{},{});
    for r in range(len(_Runs)):
        with open('FE_Data - r%d.dat'%_Runs[r], "rb") as file:	dic= pickle.load(file)
        _T_SFS.append(dic['T_SFS'])
        _CAG.append(dic['CAG'])
        _CAG_OF.append(dic['CAG_OF'])
        _CD_mean.append(dic['CD_mean'])
        _CD_rms.append(dic['CD_rms'])
        _CD_peak.append(dic['CD_peak'])
        _Fstat_mean.append(dic['Fstat_mean'])
        _Fstat_peak.append(dic['Fstat_peak'])

        _O_No.append(dic['O_No']);
        for f in range(len(dic['CAG_FEs'])):
            CAG_FEs[f]    += dic['CAG_FEs'][f][(5 if r else 0):]

    O_No = _O_No[0]
    FES_No = dic['FES_No'];
    for fe in list(_T_SFS[0].keys()):
        T_SFS[fe], CAG[fe], CAG_OF[fe], CD_mean[fe], CD_rms[fe], CD_peak[fe], Fstat_mean[fe], Fstat_peak[fe] = [],[],[],[],[],[],[],[]
        for r in range(len(_T_SFS)):
            T_SFS[fe]  += _T_SFS  [r][fe][(5 if r else 0):];
            CAG[fe]    += _CAG    [r][fe][(5 if r else 0):];
            CAG_OF[fe] += _CAG_OF [r][fe][(5 if r else 0):];
            CD_mean[fe]+= _CD_mean[r][fe][(5 if r else 0):];
            CD_rms[fe] += _CD_rms [r][fe][(5 if r else 0):];
            CD_peak[fe]+= _CD_peak[r][fe][(5 if r else 0):];
            Fstat_mean[fe]  += _Fstat_mean  [r][fe][(5 if r else 0):];
            Fstat_peak[fe]  += _Fstat_peak  [r][fe][(5 if r else 0):];
    [CAG_MD, CAG_ZCF, CAG_STE, CAG_SDE, CAG_FSDE, CAG_DDsE, CAG_EDF, CAG_FBSHT]=CAG_FEs;

#%%
