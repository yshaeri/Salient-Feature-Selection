#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 11:45:04 2017

@author: m-ali
"""

T0=time.time()
j=0
while(j<10**6):
    j+=1;
print(time.time()-T0)


xx = [0]*2000000+[1]*1000000;
np.random.shuffle(xx)
yy = np.array(xx)

#Calc time = 0.007s
T0=time.time()
print(yy.sum(),              time.time()-T0)
#print(np.sum(yy),              time.time()-T0)

#Calc time = 0.045s
T0=time.time()
print(sum(xx),              time.time()-T0)

#Calc time = 0.383s
T0=time.time()
print(np.sum(xx),           time.time()-T0)

#Calc time = 0.479s
T0=time.time()
print(len(np.where(xx)[0]), time.time()-T0)

#Calc time = 0.611s
T0=time.time()
print(sum(yy),    time.time()-T0)

#Calc time = 0.660s
T0=time.time()
print(sum(yy),              time.time()-T0)

#Calc time = 1.117s
T0=time.time()
print(sum(np.array(xx)),    time.time()-T0)

#Calc time = 11.819s
T0=time.time()
print(sum(np.logical_not(yy-1)),    time.time()-T0)

#Calc time = 11.849s
T0=time.time()
print(sum(yy==1),    time.time()-T0)

#Calc time = 12.250s
T0=time.time()
print(sum(np.array(xx)==1),    time.time()-T0)



# compare np.where, [i for i,x in enumerate(..) if (x==.)]


xx = np.concatenate([[x]*10**6 for x in range(10)]);
np.random.shuffle(xx);
yy = np.array(xx);

#Calc time = 0.0485s
T0=time.time()
label = (yy>4)
print(time.time()-T0)

#Calc time = 0.322s
T0=time.time()
label = (np.array(xx)>4)
print(time.time()-T0)

#Calc time = 0.345s
T0=time.time()
label = np.zeros(yy.shape);
label[np.where(yy>4)[0]]=1
print(time.time()-T0)

#Calc time = 3.392s
T0=time.time()
label = [x>4 for x in xx]
print(time.time()-T0)


