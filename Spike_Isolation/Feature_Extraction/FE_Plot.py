# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 20:14:59 2015

@author: mali
"""

#%% 
width=.4
chnc = np.array([item for sublist in [[Chance_MC[iS]]*C_Nos[iS] for iS in range(S_No)]   for item in sublist])

CA_Bayes    = [100*CAG[(' ',1)][iS][0] for iS in range(S_No)];
#CA_Bayes_OF = [100*np.mean(CAG_OF[(' ',1)][iS]) for iS in range(S_No)];
FE_CA_     = 100*np.array([ CAG_MD, CAG_ZCF, CAG_FSDE, CAG_SDE, CAG_DDsE, CAG_EDF, CAG_FBSHT,\
[np.mean(item) for item in CAG_OF[(' ', 1)]]]).T;
FE_CA_cli = (FE_CA_.T - 100*np.array(Chance_MC))/(1 - np.array(Chance_MC))

FE_CA_av_  = 100*np.array([ np.mean(CAG_MD), np.mean(CAG_ZCF), np.mean(CAG_SDE), np.mean(CAG_FSDE), np.mean(CAG_DDsE), np.mean(CAG_EDF), np.mean(CAG_FBSHT) ]);#np.mean(CAG_STE),
FE_CA_med_ = 100*np.array([ np.median(CAG_MD), np.median(CAG_ZCF), np.median(CAG_SDE), np.median(CAG_FSDE), np.median(CAG_DDsE), np.median(CAG_EDF), np.median(CAG_FBSHT) ]);#np.median(CAG_STE),
#standard deviation
FE_CA_std_= 100*np.array([ np.std(CAG_MD), np.std(CAG_ZCF), np.std(CAG_SDE), np.std(CAG_FSDE), np.std(CAG_DDsE), np.std(CAG_EDF), np.std(CAG_FBSHT) ]);# np.std(CAG_STE), 
#standard error
FE_CA_sem_= 100*np.array([ st.sem(CAG_MD), st.sem(CAG_ZCF), st.sem(CAG_SDE), st.sem(CAG_FSDE), st.sem(CAG_DDsE), st.sem(CAG_EDF), st.sem(CAG_FBSHT) ]);# st.sem(CAG_STE), 
#confidence interval (.95)
ci=75
FE_CA_ci_= 100*np.array([ np.percentile(CAG_MD, ci), np.percentile(CAG_ZCF, ci), np.percentile(CAG_SDE, ci), np.percentile(CAG_FSDE, ci), np.percentile(CAG_DDsE, ci), np.percentile(CAG_EDF, ci), np.percentile(CAG_FBSHT, ci)]);#np.percentile(CAG_STE, ci), 
#plt.boxplot(data, 1)
#st.t.interval(0.99, len(CAG_MD)-1, loc=np.mean(CAG_MD), scale=st.sem(CAG_MD))
#st.norm.interval(0.99, loc=np.mean(CAG_MD), scale=st.sem(CAG_MD))

#FE_NCA_  = 100*np.array([np.mean((CAG_MD -chnc)/(1-chnc)), np.mean((CAG_ZCF-chnc)/(1-chnc)),  np.mean((CAG_STE-chnc)/(1-chnc)),\
#                         np.mean((CAG_SDE-chnc)/(1-chnc)), np.mean((CAG_FSDE-chnc)/(1-chnc)), np.mean((CAG_DDsE-chnc)/(1-chnc))])
FE_Name = [r'$\mathrm{MD}$',r'$\mathrm{ZCF}$',r'$\mathrm{SDE}$',r'$\mathrm{FSDE}$',r'$\mathrm{DDsE}$', r'$\mathrm{EDF}$', r'$\mathrm{FBS}_{\mathrm{HT}}$'];#r'$\mathrm{STE}$',
FE_Mem  = [];
FE_T,FS_T=([],[]);
#FES_No = [ len(MD[0]), len(ZCF[0]), len(STE[0]), len(SDE[0]), len(FSDE[0]), len(DDsE[0]), len(EDF[0]), len(FBSHT[0])]#, len(FS_Clus[(' ',1)][0][0][0]) ];
#FE_No   = 10*np.array([ len(MD[0]), len(ZCF[0]), len(SDE[0]), len(FSDE[0]), len(DDsE[0]), len(EDF[0]), len(FBSHT[0]), len(FS_Clus[(' ',1)][0][0][0]) ]);#len(STE[0]), 
FE_No   = 10*np.array(FES_No[:2]+FES_No[3:])
FE_CA_OF, FE_NCA_OF, FE_CA, FE_NCA                          = ([],[],[],[])
FE_CA_OF_std, FE_NCA_OF_std, FE_CA_OF_sem, FE_NCA_OF_sem    = ([],[],[],[])

for ixf,xform in enumerate(Xforms):
    if(xform[1:3].lower()=='wt'):      FE_mode = (xform, Mother_Func,level);        FE_Name.append((r'$\mathrm{%s}_{\mathrm{%s}\mathrm{(L%d)}}$'%(xform,Mother_Func,level))  + '\n+\n$\mathrm{SFS}$');
    elif(xform.lower()=='derivatives'):FE_mode = ('Derivatives',3);                 FE_Name.append((r'$\mathrm{%s}$'%xform) + '\n+\n$\mathrm{SFS}$');#(r'$\mathrm{%s}$'%xform)+r'$\mathrm{(L%d)}$'%3  + '\n+\n$\mathrm{SFS}$'
    elif(xform.lower()==' '):          FE_mode = (xform,1);                         FE_Name.append('$\mathrm{SFS}$');
    else:                              FE_mode = (xform,1);                         FE_Name.append((r'$\mathrm{%s}$'%xform) + '\n+\n$\mathrm{SFS}$');

    #print(str(FE_mode)+':',np.mean(CAG_OF[FE_mode]), np.mean(CAG[FE_mode][1:]), CAG[FE_mode][0]);

    print('\n'+str(FE_mode)+':', 100*np.mean([item for sublist in CAG_OF[FE_mode]    for item in sublist]     ),\
                            100*np.mean([item for sublist in CAG[FE_mode]       for item in sublist[1:]] ),\
                            100*np.mean([item for sublist in CAG[FE_mode]       for item in [sublist[0]]]))

    #Multi-label Bayes Classifier
    FE_CA_OF.append(100*np.mean(   [item for sublist in CAG_OF[FE_mode]    for item in sublist]                  ))
    FE_NCA_OF.append(100*np.mean( ([item for sublist in CAG_OF[FE_mode]    for item in sublist] - chnc)/(1-chnc) ))

    FE_CA_OF_std.append(100*np.std(   [item for sublist in CAG_OF[FE_mode]    for item in sublist]                  ))
    FE_NCA_OF_std.append(100*np.std( ([item for sublist in CAG_OF[FE_mode]    for item in sublist] - chnc)/(1-chnc) ))

    FE_CA_OF_sem.append(100*st.sem(   [item for sublist in CAG_OF[FE_mode]    for item in sublist]                  ))
    FE_NCA_OF_sem.append(100*st.sem( ([item for sublist in CAG_OF[FE_mode]    for item in sublist] - chnc)/(1-chnc) ))

CAG_ML    = [];
CAG_ML_OF = [];
CAG_cli_ML_OF = [];
for iS in range(S_No):    CAG_ML    = CAG_ML    + CAG[(' ',1)][iS][1:];
for iS in range(S_No):    CAG_ML_OF = CAG_ML_OF + CAG_OF[(' ',1)][iS];
CAG_ML    = 100* np.array(CAG_ML);
CAG_ML_OF = 100* np.array(CAG_ML_OF);
for iS in range(S_No):    CAG_cli_ML_OF = CAG_cli_ML_OF + list(100*(CAG_OF[(' ',1)][iS]-np.array(Chance_ML[iS]))/(1-np.array(Chance_ML[iS])));

CAG_cli_ML = [100*item for iS in range(S_No) for item in list((CAG[(' ', 1)][iS][1:]-np.array(Chance_ML[iS]))/(1-np.array(Chance_ML[iS])))]
CAG_cli_MC = [100*(np.mean(CAG[(' ', 1)][iS][1:])-Chance_MC[iS])/(1-Chance_MC[iS]) for iS in range(S_No)]

fig, ax = plt.subplots(figsize = [12,8])
fig.canvas.set_window_title("FE&FS_Comparison (Fn=%d, Nch=%d)"%(O_No,S_No))

ixs = np.argsort(FE_CA_av_)
FE_Names = [FE_Name[i] for i in ixs];
ix  = np.r_[:len(FE_CA_av_)]
plt.bar(ix+0.5*width, FE_CA_av_[ixs],     yerr= ([0]*len(ixs),FE_CA_sem_[ixs]),color='w',   edgecolor='k', lw=2, width=width, error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=2), label='$\mathrm{CA}$ (%)');#'Bayes Class. Acc. (\%)'
plt.bar(ix+1.5*width, FE_No[ixs],         hatch="///",color='w',edgecolor='k', lw=2, width=width, label='$\mathrm{\#\ of\ Features}$');#label='Feat. No')
#plt.bar(np.r_[:len(FE_CA_av_)]+width, FE_Mem[ixs], color='g', width=width, label='Memory (Bytes)')
for ib,iB in enumerate(ix):
    plt.text(iB+width/2, FE_CA_av_[ixs][ib]-1.6, '%3.1f'%FE_CA_av_[ixs][ib], horizontalalignment='center', verticalalignment='center', color='k', weight='bold', clip_on=True, fontsize=12)
    plt.text(iB+1.5*width, FE_No[ixs][ib]+1, '%d'%(FE_No[ixs][ib]/10),     horizontalalignment='center', verticalalignment='center', color='k', weight='bold', clip_on=True, fontsize=12)

iFE_OF=np.array([0, 1, 2, ]);
FE_CA_OF_=np.array(FE_CA_OF)[iFE_OF]
ixs = iFE_OF[np.argsort(FE_CA_OF_)];
#ixs = np.argsort(FE_CA_OF)
ixs =np.array([2]);
FE_Names += [FE_Name[i] for i in ixs+ix.max()+1];
ix  = np.r_[:len(ixs)]+ix.max()+1
plt.bar(ix+0.5*width, np.array(FE_CA_OF)[ixs],              yerr= ([0]*len(ixs),np.array(FE_CA_OF_sem)[ixs]),  color='w',    edgecolor='k', lw=2, width=width, error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=2))
plt.bar(ix+1.5*width, [20]*len(np.array(FE_CA_OF)[ixs]),    hatch="///", color='w', edgecolor='k', lw=2, width=width)
for ib,iB in enumerate(ix):
    plt.text(iB+width/2, np.array(FE_CA_OF)[ixs][ib]-1.6, '%3.1f'%np.array(FE_CA_OF)[ixs][ib], horizontalalignment='center', verticalalignment='center', color='k', weight='bold', clip_on=True,fontsize=12)
    plt.text(iB+1.5*width, 20+1, '%d'%2, horizontalalignment='center', verticalalignment='center', color='k', weight='bold', clip_on=True, fontsize=11)

plt.bar(np.r_[:FE_CA_cli.shape[0]]+.5*width, FE_CA_cli.mean(axis=1), yerr=([0]*len(FE_CA_cli),st.sem( FE_CA_cli,axis=1)), color='k',    edgecolor='k', lw=2, width=.8*width, error_kw=dict(ecolor='k', lw=2, capsize=5, capthick=2), label='$\mathrm{CA}_\mathrm{CLI}$(%)')
for ib,iB in enumerate(np.r_[:FE_CA_cli.shape[0]]):
    plt.text(iB+width/2, FE_CA_cli.mean(axis=1)[ib]-1.6, '%3.1f'%FE_CA_cli.mean(axis=1)[ib], horizontalalignment='center', verticalalignment='center', color='w', weight='bold', clip_on=True,fontsize=12)

        
ax.set_ylabel('Bayes CA (%)', color='k', fontsize=14)
plt.xlim([0-.25,ix.max()+1])
plt.ylim([0,100])
ax.set_xticks(np.r_[:ix.max()+1] + width)
ax.set_xticklabels(FE_Names, fontsize=12);
ax2 = ax.twinx()
ax2.set_ylim(0,10);
ax2.set_ylabel('# of Features', color='k', fontsize=14)

handles, labels = ax.get_legend_handles_labels()
order = [0,2,1]
lgnd = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],\
        loc='upper left',fancybox=1, shadow=1, framealpha=1, mode=None, borderpad=.4, labelspacing=1, handlelength=2, handletextpad=1, borderaxespad=1, ncol=2, columnspacing=4, markerscale=5,fontsize=12);#bbox_to_anchor=(0.14, 1)


fig.subplots_adjust(left=.06,right=.94,bottom=0.2,top=.94)
plt.grid(True, linestyle='--', which='major', color='grey', alpha=.5)
ax.tick_params('both',labelsize=14);
ax2.tick_params('both',labelsize=14);

plt.margins(0.02,0)
fig.tight_layout()
plt.subplots_adjust(left=0.07, right=.95, bottom=0.04, top=.98, wspace=0, hspace=0)


print("GNB without dimension reduction:\n\tMulti-class CA: %0.4f\n\tMulti-Label CA: %0.4f"\
%(np.mean([CAG[(' ',1)][iS][0] for iS in range(S_No)]), np.mean([np.mean(CAG[(' ',1)][iS][1:]) for iS in range(S_No)])) )

#%%
lr = [sk.linear_model.LinearRegression(),sk.linear_model.Ridge(alpha=.1),sk.linear_model.RANSACRegressor(),sk.linear_model.TheilSenRegressor(random_state=42)][0]
#lr = sk.linear_model.LinearRegression(normalize=True);

CA_Type = ['CA','CA_CLI'][0];
iF=-1;
CA_Bayes_OF=FE_CA_[:,iF]
xlim, ylim = ([50,100], [50,100]) if CA_Type =='CA' else ([0,100], [0,100]);

if CA_Type =='CA':
    ylabel = '$\mathrm{CA}_\mathrm{CLI, SFS}$ (%)';
    xlabel = '$\mathrm{CA}_\mathrm{CLI}$ (%)';
elif CA_Type =='CA_CLI':
    ylabel = '$\mathrm{CA}';
    xlabel = '$\mathrm{CA}_\mathrm{CLI}$ (%)';


plt.close('%s-%s plot (Nch=%d)'%(CA_Type,CA_Type,S_No)); #$-\ \mathrm{MC}$
fig = plt.figure('%s-%s plot (Nch=%d)'%(CA_Type,CA_Type,S_No), figsize=[12,6], facecolor='w', edgecolor='k', dpi=100)
plt.subplot(121, aspect='equal', xlim=xlim, ylim=ylim);#adjustable='box-forced',

if CA_Type =='CA':          X,Y = np.array(CA_Bayes), np.array(CA_Bayes_OF);
elif CA_Type =='CA_CLI':    X,Y = CAG_cli_MC,         FE_CA_cli[iF];
plt.scatter(X,Y,  color='#666666', s=np.array(C_Nos)*20,alpha=.75);

(m, k), pcov = sp.optimize.curve_fit(lambda x, m, k: m*x+k, X, Y)
lr.fit(np.transpose([X]), Y);
pred = lr.predict(np.transpose([[0,100]]));#lr.predict(np.transpose([CA_Bayes]));
plt.plot(   [0,100]               , np.array(pred), color='k', linewidth=3);

plt.ylabel('$\mathrm{CA}_\mathrm{SFS}$ (%)');
plt.xlabel('$\mathrm{CA}$ (%)');
#plt.text(xlim[0],100, "\n Slope:%.2f"%m,\
plt.text(xlim[0]+.5,99.5, "a\n",\
         horizontalalignment='left', verticalalignment='top', color='k', weight='bold', clip_on=True,fontsize=16)

ax = plt.subplot(122, aspect='equal', xlim=xlim, ylim=ylim);#adjustable='box-forced',
for iF in range(FE_CA_.shape[1]-1,-1,-1):
    CA_Bayes_OF=FE_CA_[:,iF]
    if CA_Type =='CA':          X,Y = CA_Bayes, CA_Bayes_OF;
    elif CA_Type =='CA_CLI':    X,Y = CAG_cli_MC,         FE_CA_cli[iF];

    lr.fit(np.transpose([X]), Y);
    pred = lr.predict(np.transpose([[0,100]]));#lr.predict(np.transpose([CA_Bayes]));
    plt.plot(   [0,100]               , np.array(pred       ),  color=_Color[iF], linewidth=3, label=FE_Names[iF]);
    #plt.plot([0,100], m*np.array([0,100])+k,            color='cymrgbk'[iF],linewidth=3, label=FE_Names[iF])
    plt.ylabel('$\mathrm{CA}_\mathrm{FE}$ (%)');
    plt.xlabel('$\mathrm{CA}$ (%)');
plt.text(xlim[0]+.5,99.5, "b\n",\
         horizontalalignment='left', verticalalignment='top', color='k', weight='bold', clip_on=True,fontsize=16)
ax.legend(bbox_to_anchor=(.21, .5), frameon = False,fontsize=12)
#plt.margins(0.02,0)
fig.tight_layout()
plt.subplots_adjust(left=0.05, bottom=0.08, right=.99, top=.96, wspace=0.1, hspace=0.01)

#%%
li=np.max
plt.figure('CA-CA plot-%d'%iF, figsize=(10,5));

for iF in range(FE_CA_.shape[1]):
    CA_Bayes_OF=FE_CA_[:,iF]

    ax = plt.subplot(121, aspect='equal', xlim=[50,100], ylim=[50,100]);#adjustable='box-forced',
    #plt.title("CA_SFS vs CA_Bayes - MC");
    plt.title('$\mathrm{CA}_\mathrm{SFS}\ \mathrm{vs.}\ \mathrm{CA}\ -\ \mathrm{MC}$');
    #plt.scatter(np.array(CA_Bayes), np.array(CA_Bayes_OF), color='k', s=np.array(C_Nos)*20);

    (m, k), pcov = sp.optimize.curve_fit(lambda x, m, k: m*x+k, np.array(CA_Bayes), np.array(CA_Bayes_OF), method= ['lm', 'trf', 'dogbox'][2])
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress( np.array(CA_Bayes), np.array(CA_Bayes_OF));#m, k = slope, intercept
    lr.fit(np.transpose([CA_Bayes]), CA_Bayes_OF);
    pred = lr.predict(np.transpose([[0,100]]));#lr.predict(np.transpose([CA_Bayes]));
    plt.plot(   [0,100]               , np.array(pred       ),  color=_Color[iF], linewidth=3, label=FE_Names[iF]);
    #plt.plot([0,100], m*np.array([0,100])+k,            color='cymrgbk'[iF],linewidth=3, label=FE_Names[iF])
    plt.ylabel('$\mathrm{CA}_\mathrm{SFS}$ (%)');
    plt.xlabel('$\mathrm{CA}$ (%)');

    m, pcov = sp.optimize.curve_fit(lambda x, m: m*(x-li(CA_Bayes))+li(CA_Bayes_OF), np.array(CA_Bayes), np.array(CA_Bayes_OF))
    m, _, _, _ =np.linalg.lstsq( np.array(CA_Bayes)[:,np.newaxis]-li(CA_Bayes), np.array(CA_Bayes_OF)-li(CA_Bayes_OF))

    plt.subplot(122, aspect='equal', xlim=[0,100], ylim=[0,100]);#adjustable='box-forced',
    plt.title('$\mathrm{CA}_\mathrm{CLI, SFS}\ \mathrm{vs.}\ \mathrm{CA}_\mathrm{CLI}\ -\ \mathrm{MC}$');
    #plt.scatter(CAG_cli_MC, FE_CA_cli[iF], color='k', s=np.array(C_Nos)*20);
    (m, k), pcov = sp.optimize.curve_fit(lambda x, m, k: m*x+k, np.array(CAG_cli_MC), np.array(FE_CA_cli[iF]))

    lr = sk.linear_model.LinearRegression();
    lr.fit(np.transpose([CAG_cli_MC]), FE_CA_cli[iF]);
    pred = lr.predict(np.transpose([[0,100]]));#lr.predict(np.transpose([CA_Bayes]));
    plt.plot(   [0,100]                  ,  np.array(pred),         color=_Color[iF], linewidth=3);
    #plt.plot([0,100], m*np.array([0,100])+k,            color='cymrgbk'[iF],linewidth=3)
    plt.ylabel('$\mathrm{CA}_\mathrm{CLI, SFS}$ (%)');
    plt.xlabel('$\mathrm{CA}_\mathrm{CLI}$ (%)');

ax.legend(bbox_to_anchor=(1.5, 1), frameon = False)
#%% Box-whisker plot

plt.figure()
plt.subplot(2,1,1);
#Boxes are 25th and 75th percentiles, and wiskers are 
plt.boxplot(FE_CA_,1, 'go',1, 0.99,patch_artist=1,showmeans=True,
            meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='r') );#plt.boxplot(FE_CA_av_,1, '')
#plt.boxplot( FE_CA_av_.T.tolist()+[[100*item for sublist in CAG_OF[(' ', 1)]    for item in sublist]] ,1)
plt.ylabel('$\mathrm{CA}_\mathrm{CLI}$ (%)', color='k', fontsize=14)
plt.ylim([0,100]);
plt.xticks(range(1,len(FE_Names)+1), ' '*len(FE_Names), rotation='vertical', fontsize=12)

plt.subplot(2,1,2);
plt.boxplot(FE_CA_cli.T,1, 'go',1, 0.99,patch_artist=1,showmeans=True,
            meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='r') );#plt.boxplot(FE_CA_av_,1, '')
plt.ylabel('$\mathrm{Bayes\ CA_{CLI}}$ (%)', color='k', fontsize=14)
plt.ylim([0,100]);

plt.xticks(range(1,len(FE_Names)+1), FE_Names, rotation='vertical', fontsize=12)

#%%
