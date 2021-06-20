# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:14:59 2015

@author: mali
"""
# Modifications: using empirical variance, doble-sided intraction of saliencies
#%%
#exec(open(Init_Dir+"/Feature_Selection/FS_Init.py").read())

#%%
iS = 1; # iS is Recording Site index and S_No is No of Feature Sets
C_No = len(FE_Clus[iS]); # No of Clusters (units)
FS_Clus = [[FE_Clus[iS][iC].feature[iF] for iF in range(F_No)] for iC in range(C_No)]

iD=0;
iI=0;

(dw, d_withins, d_betweens), CDt, CDc, FSM = FS.Class_Discrimination( FS_Clus, );
Pw = np.array([FS_Clus[iC][0].shape[1] for iC in range(C_No)]);   Pw = Pw/Pw.sum();
iFS  = 0;

Homogenity  = np.array([np.product([CDt[iFS][iC,ic]**Pw[ic] for ic in range(C_No) if ic!=iC],axis=0)**(1/(1-Pw[iC]))\
               /(np.sum([CDt[iFS][iC,ic]*Pw[ic]      for ic in range(C_No) if ic!=iC],axis=0)/(1-Pw[iC]))\
               for iC in range(C_No)]);
GM          = np.array([np.product([CDt[iFS][iC,ic]**Pw[ic] for ic in range(C_No) if ic!=iC],axis=0)**(1/(1-Pw[iC]))\
               for iC in range(C_No)]);

DF  = [np.array(CD[iS][iD][iI][iFS])[:,1:] - np.array(CD[iS][iD][iI][iFS])[:,:-1] for iFS in range(F_No)]; 
IDX = [np.array([((DF[iFS][iC,1:]<0)&(DF[iFS][iC,:-1]>=0)) & (np.array(CD[iS][iD][iI][iFS])[iC,1:-1] > np.array(CD[iS][iD][iI][iFS])[iC,1:-1].mean()) for iC in range(C_No)])\
       for iFS in range(F_No)];
iP  = [[np.where(IDX[iFS][iC]==True)[0]+1                  for iC in range(C_No)] for iFS in range(F_No)];# indices of Feature peak
PF  = [[np.array(CD[iS][iD][iI][iFS])[iC,iP[iFS][iC]]          for iC in range(C_No)] for iFS in range(F_No)];

box = dict(facecolor='w', edgecolor='w',pad=0, alpha=0.2)

Nr,Nc = [5,F_No];
gs = gridspec.GridSpec(Nr,Nc);#, width_ratios=[3, 1]) 
fig= plt.figure('Feature Saliency',figsize=(8, 8));
plt.clf();
for iFS in range(F_No):      # iFS is Feature Set index and F_No is No of Feature Sets
#    r,c = iFS//Nc+3, iFS%Nc;
    ax = plt.subplot(gs[:2,iFS]);
    for iC in range(C_No):
        plt.plot(FS_Clus[iC][iFS], 'rgb'[iC], alpha=.1);
        plt.plot(np.mean(FS_Clus[iC][iFS],1),['#E00000','#00AA00','#0000BB'][iC],linewidth=4, label='Unit #%d'%(iC+1));
    if(FE_mode[0]!=' '): ax.set_title(FE_Feat.Coef.label[iFS], y=.8);
    ax.set_xlim( (0, FE_Feat.Coef.ticks[iFS].shape[0]-1) );
    ax.set_xticklabels([]);
    if iFS==0:      ax.set_ylabel("Amplitude", fontname="Arial", position=(0.1,0.1), bbox=box);
    ax.yaxis.set_label_coords(-.06, 0.5);
    if iFS==F_No-1: ax.legend(loc="upper right", bbox_to_anchor=[1, 1], ncol=1, shadow=True, fancybox=True, fontsize=14);

    ax = plt.subplot(gs[2,iFS]);
    for iC in range(C_No):
        ax.plot(GM[iC],'rgb'[iC],lw=2,label='Unit #%d'%(iC+1));
    ax.set_xlim( (0, FE_Feat.Coef.ticks[iFS].shape[0]-1) );
    ax.set_ylim( (1, 100) );
    ax.set_xticklabels([]);
    ax.set_yscale('log');
    if iFS==0:      ax.set_ylabel(r'Geo. mean', fontname="Arial", position=(0.1,0.1), bbox=box);
    ax.yaxis.set_label_coords(-.06, 0.5);

    ax = plt.subplot(gs[3,iFS]);
    for iC in range(C_No):
        ax.plot(Homogenity[iC],'rgb'[iC],lw=2,label='Unit #%d'%(iC+1));
    ax.set_xlim( (0, FE_Feat.Coef.ticks[iFS].shape[0]-1) );
    ax.set_ylim( (.3, 1.1) );
    ax.set_xticklabels([]);
    ax.set_yscale('log');
    plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    yticks=[1,.5]
    plt.gca().axes.set_yticks(yticks);
    plt.gca().axes.set_yticklabels(yticks);

    if iFS==0:      ax.set_ylabel(r'Homogenity', fontname="Arial", position=(0.1,0.1), bbox=box);
    ax.yaxis.set_label_coords(-.06, 0.5);

    ax = plt.subplot(gs[4,iFS]);
    for iC in range(C_No):
        ax.plot(CDc[iFS][iC],'rgb'[iC],lw=2,label=r'$\varsigma_%d$'%(iC+1));
    ax.set_xlim( (0, FE_Feat.Coef.ticks[iFS].shape[0]-1) );
    ax.set_ylim( (1, 100) );
    ax.set_xticklabels([]);
    ax.set_yscale('log');
    if iFS==0:      ax.set_ylabel(r'$\varsigma_i$',fontsize=14, fontname="Arial", position=(0.1,0.1), bbox=box);
    ax.yaxis.set_label_coords(-.06, 0.5);
ax.set_xlabel(r'$k$',fontsize=14);

plt.subplots_adjust(left=0.1, bottom=0.05, right=.99, top=.99, wspace=0.025, hspace=0.05);

#%%
