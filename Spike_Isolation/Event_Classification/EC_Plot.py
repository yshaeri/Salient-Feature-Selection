#python3 "./EC_Plot.py"
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:54:12 2016

@author: mali
"""

#%%
#exec(open(Init_Dir+"/Event_Classification/EC_Init.py").read())

#%%Multi-label Window Discriminators (WD) 
iS   = 0;
iFES = 3;

FES = FESs[iFES];
FE_mode = FES[0];
FS_mode = FES[1];
EC_mode = FES[2];
_FE_Feat, _FE_Clus = FE.Feature_Extraxtion( [Clus[iS]], FE_mode );

O_No = FS_mode[1]
C_No = C_Nos[iS]
F_No = len(_FE_Feat.Coef.label);

FS_Clus = [[] for x in range(C_No)];
for iC in range(C_No):
    for iF in range(F_No):      # iF is Feature Set index and F_No is No of Feature Sets
        FS_Clus[iC].append(_FE_Clus[0][iC].feature[iF]);

OF, SF, FD = FS.Optimal_Feature_Selection( FS_Clus, _FE_Feat, O_No, );


wd_bs   = 1; # End with bin/segment size
wd_bn   = 4; # Start with number of bin/segments in each feature dimension
wd_wd   = 16; # Window discriminators per stage
wd_mode = ['max-area','sequence','span'][1];
wd_calc_mode = [0,1,2,3][1];
WD  = WC.WindowDiscrimination(wd_bs, wd_bn, wd_wd, wd_mode, wd_calc_mode);

Bnd ,T_, CA_= [],[],[];
plt.figure('Multi-label WD Classification',figsize=(14,4.5));
for iC  in range(C_No):
    FS_feat = [[] for ic in range(C_No)];
    for iO in range(0,O_No):
        FS_feat = [FS_feat[ic]+[FS_Clus[ic][OF[iC][iO][0]][OF[iC][iO][1]]]     for ic in range(C_No)];
    FES_data = np.concatenate(FS_feat,axis=1).T;
    T0=time.time();
    WD.fit(FES_data, np.array(Label[iS])==iC);
    T_.append(time.time()-T0)
    Bnd.append(WD.Bounds[0]);
    CA_.append( len(np.where(WD.predict(FES_data) == (np.array(Label[iS])==iC))[0])/len(Label[iS]) );
    ax=plt.subplot(1,3,iC+1);
    for ic in range(C_No):
        ix = np.where(np.array(Label[iS])==ic);
        ax.scatter(FES_data[:,0][ix], FES_data[:,1][ix], s=10,c=['w','rgb'[ic]] [iC==ic], edgecolor='rgb'[ic],alpha=[.5,.5][iC==ic]);
        if(_FE_Feat.Function==''):
            iO=0; ax.set_xlabel( r"$k_%d^%d=%d$"%(iC,iO,OF[iC][iO][1]), fontsize=18);
            iO=1; ax.set_ylabel( r"$k_%d^%d=%d$"%(iC,iO,OF[iC][iO][1]), fontsize=18);

        else:
            iO=0; ax.set_xlabel(r"$k_%d^%d$=%s"%(iC,iO,_FE_Feat.Coef.label[OF[iC][iO][0]])+'[%d]'%OF[iC][iO][1])
            iO=1; ax.set_ylabel(r"$k_%d^%d$=%s"%(iC,iO,_FE_Feat.Coef.label[OF[iC][iO][0]])+'[%d]'%OF[iC][iO][1])

    plt.plot( Bnd[iC][0]+Bnd[iC][0][::-1]+[min(Bnd[iC][0])],\
              [min(Bnd[iC][1])]*2+[max(Bnd[iC][1])]*2+[min(Bnd[iC][1])], color='rgb'[iC], ls='-', lw=3, alpha=.6);
    ax.xaxis.set_label_coords(.5, -0.06);
    ax.yaxis.set_label_coords(-0.1,.5);
plt.margins(.5,.5)
print(np.mean(T_),'s\n', np.mean(CA_)*100,'%')

plt.subplots_adjust(left=0.06, right=0.99, top=0.99, bottom=0.15,  wspace=0.2, hspace=0.05,);



#%%
_FESs=FESs
print(pandas.DataFrame(_FESs,list(range(len(_FESs))), ['FE','FS','EC']),'\n');
print(pandas.DataFrame( np.array([[np.mean(CA [FES][EC_type]) for EC_type in _ECs if(EC_type[0]!='TM' and EC_type[0]!='WD')]\
                                   for FES in _FESs]), list(range(len(_FESs))), [EC_type for EC_type in _ECs if(EC_type[0]!='TM' and EC_type[0]!='WD')]), '\n');
print(pandas.DataFrame( np.array([[np.mean(C_T[FES][EC_type]) for EC_type in _ECs if(EC_type[0]!='TM' and EC_type[0]!='WD')]\
                                   for FES in _FESs]), list(range(len(_FESs))), [EC_type for EC_type in _ECs if(EC_type[0]!='TM' and EC_type[0]!='WD')]), '\n');

print(pandas.DataFrame(_FESs,list(range(len(_FESs))), ['FE','FS','EC']),'\n');
print(pandas.DataFrame( np.array([[np.mean(CA [FES][EC_type]) for EC_type in _ECs if(EC_type[0]=='TM')]\
                                   for FES in _FESs]), list(range(len(_FESs))), [EC_type for EC_type in _ECs if(EC_type[0]=='TM')]), '\n');
print(pandas.DataFrame( np.array([[np.mean(C_T[FES][EC_type]) for EC_type in _ECs if(EC_type[0]=='TM')]\
                                   for FES in _FESs]), list(range(len(_FESs))), [EC_type for EC_type in _ECs if(EC_type[0]=='TM')]), '\n');

print(pandas.DataFrame(_FESs,list(range(len(_FESs))), ['FE','FS','EC']),'\n');
print(pandas.DataFrame( np.array([[np.mean(CA [FES][EC_type]) for EC_type in _ECs if(EC_type[0]=='WD')]\
                                   for FES in _FESs]), list(range(len(_FESs))), [EC_type for EC_type in _ECs if(EC_type[0]=='WD')]), '\n');
print(pandas.DataFrame( np.array([[np.mean(C_T[FES][EC_type]) for EC_type in _ECs if(EC_type[0]=='WD')]\
                                   for FES in _FESs]), list(range(len(_FESs))), [EC_type for EC_type in _ECs if(EC_type[0]=='WD')]), '\n');



#%%
print(np.array([[\
    np.mean(CA[((' ', 1), ('SFS', 2), 'Multi-label')][('WD', 2, 4, 16)]),\
    np.mean(CA[((' ', 1), ('SFS', 2), 'Multi-label')][('WD', 1, 4, 16)]),\
    np.mean(CA[((' ', 1), ('SFS', 2), 'Multi-label')][('WD', 1, 16, 'sequence')]),\
    np.mean(CA[((' ', 1), ('SFS', 2), 'Multi-label')][('WD', 1, 128, 'sequence')])],
    [np.mean(C_T[((' ', 1), ('SFS', 2), 'Multi-label')][('WD', 2, 4, 16)]),\
    np.mean(C_T[((' ', 1), ('SFS', 2), 'Multi-label')][('WD', 1, 4, 16)]),\
    np.mean(C_T[((' ', 1), ('SFS', 2), 'Multi-label')][('WD', 1, 16, 'sequence')]),\
    np.mean(C_T[((' ', 1), ('SFS', 2), 'Multi-label')][('WD', 1, 128, 'sequence')])]]))


CA_SS = [np.mean(CA[((' ', 1), (' ',), 'Multi-class')][('TM', 'md', 1, 1)])*100,\
         np.mean(CA['FBSHT'][('ODT', 2, 50)])*100,\
         np.mean(CA[((' ', 1), ('SFS', 1), 'Multi-label')][('WD', 2, 4, 16)])*100,\
         np.mean(CA[((' ', 1), ('SFS', 2), 'Multi-label')][('WD', 2, 4, 16)])*100];

TCA_SS = [np.mean( (CA[((' ', 1), (' ',), 'Multi-class')][('TM', 'md', 1, 1)] - Chance_MC)/(1 - Chance_MC) )*100,\
         np.mean( (CA['FBSHT'][('ODT', 2, 50)] - Chance_MC)/(1 - Chance_MC) )*100,\
         np.mean( (CA[((' ', 1), ('SFS', 1), 'Multi-label')][('WD', 2, 4, 16)] - Chance_MC)/(1 - Chance_MC)*100 ),\
         np.mean( (CA[((' ', 1), ('SFS', 2), 'Multi-label')][('WD', 2, 4, 16)] - Chance_MC)/(1 - Chance_MC)*100 )];

CT_SS = [np.mean(C_T[((' ', 1), (' ',), 'Multi-class')][('TM', 'md', 1, 1)]),\
         np.mean(C_T['FBSHT'][('ODT', 2, 50)]),\
         np.mean(C_T[((' ', 1), ('SFS', 1), 'Multi-label')][('WD', 2, 4, 16)]),\
         np.mean(C_T[((' ', 1), ('SFS', 2), 'Multi-label')][('WD', 2, 4, 16)])];#Average Calc. Time

CA_SS_sem = [st.sem(CA[((' ', 1), (' ',), 'Multi-class')][('TM', 'md', 1, 1)])*100,\
         st.sem(CA['FBSHT'][('ODT', 2, 50)])*100,\
         st.sem(CA[((' ', 1), ('SFS', 1), 'Multi-label')][('WD', 2, 4, 16)])*100,\
         st.sem(CA[((' ', 1), ('SFS', 2), 'Multi-label')][('WD', 2, 4, 16)])*100];

TCA_SS_sem = [st.sem( (CA[((' ', 1), (' ',), 'Multi-class')][('TM', 'md', 1, 1)] - Chance_MC)/(1 - Chance_MC) )*100,\
         st.sem( (CA['FBSHT'][('ODT', 2, 50)] - Chance_MC)/(1 - Chance_MC) )*100,\
         st.sem( (CA[((' ', 1), ('SFS', 1), 'Multi-label')][('WD', 2, 4, 16)] - Chance_MC)/(1 - Chance_MC)*100 ),\
         st.sem( (CA[((' ', 1), ('SFS', 2), 'Multi-label')][('WD', 2, 4, 16)] - Chance_MC)/(1 - Chance_MC)*100 )];

CT_SS_sem = [st.sem(C_T[((' ', 1), (' ',), 'Multi-class')][('TM', 'md', 1, 1)]),\
         st.sem(C_T['FBSHT'][('ODT', 2, 50)]),\
         st.sem(C_T[((' ', 1), ('SFS', 1), 'Multi-label')][('WD', 2, 4, 16)]),\
         st.sem(C_T[((' ', 1), ('SFS', 2), 'Multi-label')][('WD', 2, 4, 16)])];#Average Calc. Time


SS_Names = [r'$\mathrm{l_1\ TM}$', r'$\mathrm{FBS\ +\ ODT}$', r'$\mathrm{1D\ SFS\ +\ WD}$', '$\mathrm{2D\ SFS\ + WD}$'];
colors='rgbcymk';
width=len(CA_SS)/(len(CA_SS)+1)/2; #4./(len(CA_SS)+1);

plt.close('SS Comparison - CA,CT (%d ch.)'%S_No);
plt.figure('SS Comparison - CA,CT (%d ch.)'%S_No,figsize=(8,7));
plt.bar(np.r_[:len(CA_SS)], CA_SS, yerr= ([0]*len(CA_SS_sem),CA_SS_sem), align='center', width=width, color='none',edgecolor = colors[-1], lw=2, error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=2), label='$\mathrm{CA}$');
for iSS in range(len(CA_SS)):
    plt.text(iSS, CA_SS[iSS]-2.5, '%2.1f'%CA_SS[iSS],\
             horizontalalignment='center', verticalalignment='center', color=colors[-1], weight='bold', clip_on=True,fontsize=12)

plt.bar(np.r_[:len(TCA_SS)], TCA_SS, yerr= ([0]*len(TCA_SS_sem),TCA_SS_sem), align='center', width=.8*width, color=colors[-1], lw=2, error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=2), label='$\mathrm{CA}_\mathrm{CLI}$');
for iSS in range(len(TCA_SS)):
    plt.text(iSS, TCA_SS[iSS]-2.5, '%2.1f'%TCA_SS[iSS],\
             horizontalalignment='center', verticalalignment='center', color='w', weight='bold', clip_on=True, fontsize=12)
plt.xlim([-width,4-width/2]);    plt.ylim([0,100]);
plt.ylabel('CA (%)', fontsize=14)
yticks = plt.yticks()[0]

yscale = np.array(5);
plt.bar(np.r_[:len(CT_SS)]+width, CT_SS*yscale, yerr= ([0]*len(CT_SS_sem),CT_SS_sem), align='center', hatch="...", width=width, color='none',edgecolor = colors[-1], lw=2, error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=2), label='$\mathrm{Calc.\ Time}$');
for iSS in range(len(CT_SS)):
    if(CT_SS[iSS]>.4):
        plt.text(iSS+1*width, (CT_SS[iSS]+.5)*yscale, '%1.2f'%CT_SS[iSS].round(2),\
                 horizontalalignment='center', verticalalignment='center', color='k', weight='bold', clip_on=True,fontsize=12,alpha=.6); #,alpha=1
    else:
        plt.text(iSS+1*width, (CT_SS[iSS]+.5)*yscale, '%1.2f'%CT_SS[iSS].round(2),\
                 horizontalalignment='center', verticalalignment='center', color='k', weight='bold', clip_on=True,fontsize=12,alpha=.6); #,alpha=1

plt.gca().yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.5)
fig.tight_layout()
lgnd = plt.legend(loc='upper left',fancybox=1, shadow=1, framealpha=1, mode=None, borderpad=.4, labelspacing=1, handlelength=2, handletextpad=1, borderaxespad=.2, ncol=3, columnspacing=4, markerscale=5,fontsize=16);


plt.xticks(np.r_[:len(CT_SS)]-0.5*width, SS_Names, fontsize=14, horizontalalignment ='left', verticalalignment='top', rotation_mode="anchor" );
ax2 = plt.twinx()
ax2.set_ylim(0,20);
ax2.set_ylabel('Calculation Time (s)', color='k', fontsize=14)
plt.yticks(yticks/yscale)
plt.subplots_adjust(left=0.08, bottom=0.05, right=.92, top=.98, wspace=0, hspace=0);

#%%
plt.close('SS Comparison - CA&CT (%d ch.)'%S_No);
plt.figure('SS Comparison - CA&CT (%d ch.)'%S_No,figsize=(15,7));
plt.subplot(121);
plt.bar(np.r_[:len(CA_SS)], CA_SS, yerr= ([0]*len(CA_SS_sem),CA_SS_sem), align='center', width=width, color='none',edgecolor = colors[-1], lw=2, error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=2), label='$\mathrm{CA}$');
for iSS in range(len(CA_SS)):
    plt.text(iSS, CA_SS[iSS]-2.5, '%2.1f'%CA_SS[iSS],\
             horizontalalignment='center', verticalalignment='center', color=colors[-1], weight='bold', clip_on=True,fontsize=12)

plt.bar(np.r_[:len(TCA_SS)], TCA_SS, yerr= ([0]*len(TCA_SS_sem),TCA_SS_sem), align='center', width=.8*width, color=colors[-1], lw=2, error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=2), label='$\mathrm{CA}_\mathrm{CLI}$');
for iSS in range(len(TCA_SS)):
    plt.text(iSS, TCA_SS[iSS]-2.5, '%2.1f'%TCA_SS[iSS],\
             horizontalalignment='center', verticalalignment='center', color='w', weight='bold', clip_on=True, fontsize=12);


plt.xlim([-width,4-width/2]);    plt.ylim([0,100]);
plt.ylabel('CA (%)', fontsize=14)
plt.yticks(np.r_[0:101:20], fontsize=11);


plt.gca().yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.5)
fig.tight_layout();
lgnd = plt.legend(loc='upper right',fancybox=1, shadow=1, framealpha=1, mode=None, borderpad=.4, labelspacing=1, handlelength=2, handletextpad=1, borderaxespad=.2, ncol=3, columnspacing=4, markerscale=5,fontsize=16);
plt.xticks(np.r_[:len(CT_SS)]-0.5*width, SS_Names, fontsize=14, horizontalalignment ='left', verticalalignment='top', rotation_mode="anchor" );

plt.subplot(122);
plt.bar(np.r_[:len(CT_SS)], CT_SS, yerr= ([0]*len(CT_SS_sem),CT_SS_sem), align='center', hatch="...", width=width, color='none',edgecolor = colors[-1], lw=2, error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=2), label='$\mathrm{Calc.\ Time}$');
for iSS in range(len(CT_SS)):
    if(CT_SS[iSS]>1):
        plt.text(iSS, (CT_SS[iSS]+.75), '%1.2f'%CT_SS[iSS].round(2),\
                 horizontalalignment='center', verticalalignment='center', color='k', weight='bold', clip_on=True,fontsize=12,alpha=1); #,alpha=1
    else:
        plt.text(iSS, (CT_SS[iSS]+.5), '%1.2f'%CT_SS[iSS].round(2),\
                 horizontalalignment='center', verticalalignment='center', color='k', weight='bold', clip_on=True,fontsize=12,alpha=1); #,alpha=1

plt.xlim([-width,4-width/2]);    plt.ylim(0,20);#ax2.set_yticklabels()
plt.ylabel('Calculation Time (s)', color='k', fontsize=14)
fig.tight_layout()
plt.xticks(np.r_[:len(CT_SS)]-0.5*width, SS_Names, fontsize=14, horizontalalignment ='left', verticalalignment='top', rotation_mode="anchor" );
plt.yticks(np.r_[0:21:5], fontsize=11);

plt.subplots_adjust(left=0.05, bottom=0.05, right=.98, top=.98, wspace=0.12, hspace=0);

#%%


