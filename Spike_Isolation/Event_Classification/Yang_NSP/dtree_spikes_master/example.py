import numpy as np;
import scipy as sp;
import matplotlib.pyplot as plt;
import os;



#clear();
plt.close('all');
#os.chdir(os.getenv("HOME")+'/Documents/Notebooks/Notes/Note/Proposal/Bidirectional_Wireless_multi-channel_recording_&_stimulation_system/\
#         3._Rec&Stim_System/Recording/Online_Spike_Sorting/Source Codes/Simulation/Spike_Isolation/Event_Classification/Yang NSP/dtree-spikes-master/')
Dir=os.getenv("HOME")+\
'/Documents/Notebooks/Notes/Note/Proposal/Bidirectional_Wireless_multi-channel_recording_&_stimulation_system/\
3._Rec&Stim_System/Recording/Online_Spike_Sorting/Source Codes/Simulation/'
sys.path.append(Dir);

from Spike_Isolation.Event_Classification.Yang_NSP.dtree_spikes_master.tree_split import tree_split;
from Spike_Isolation.Event_Classification.Yang_NSP.dtree_spikes_master.tree_test import tree_test;

import Spike_Isolation.Event_Classification.Event_Classification as WC;





#xx = sp.io.loadmat('../matlab/data-6.mat')
#data  = xx['data']
#labels= xx['labels'].T[0]

sigma=[[1,0],[0,1]];
N=200;
mu=[0,  2 ];     data1 = np.random.multivariate_normal(mu, sigma, N)
mu=[-4, -1];     data2 = np.random.multivariate_normal(mu, sigma, N)
mu=[-1, -4];     data3 = np.random.multivariate_normal(mu, sigma, N)
mu=[-1, 7 ];     data4 = np.random.multivariate_normal(mu, sigma, N)
mu=[-5, 5 ];     data5 = np.random.multivariate_normal(mu, sigma, N)
mu=[-5, -7];     data6 = np.random.multivariate_normal(mu, sigma, N)

data=np.concatenate([data1,data2,data3,data4,data5,data6])
labels=np.concatenate([np.ones(N), np.ones(N)*2, np.ones(N)*3, np.ones(N)*4, np.ones(N)*5, np.ones(N)*6])

plt.plot(data1[:,0],data1[:,1],'o')
plt.hold('on')
plt.plot(data2[:,0],data2[:,1],'ro')
plt.plot(data3[:,0],data3[:,1],'go')
plt.plot(data4[:,0],data4[:,1],'yo')
plt.plot(data5[:,0],data5[:,1],'co')
plt.plot(data6[:,0],data6[:,1],'mo')

bits=2; # specify the resolution
repeats =[5,50][1]
coeff,class_id = tree_split(data,labels,bits,repeats);
err=tree_test(data,labels,coeff,class_id);
#boundaryplot2d(data,labels,coeff)
#title(cat('an oblique decision tree with ',num2str(bits),' bit resolution'))


EC = WC.ObliqueQDT(bits, repeats);
EC.fit(data,labels)
CA = np.mean(EC.predict(data) == labels)