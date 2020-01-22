import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def tau(data,freq,time,weights):
    data = np.asarray(data)
    freq = np.asarray(freq)
    time = np.asarray(time)
    weights = np.asarray(weights)
    tau=np.arctan((np.asarray([np.sum(weights*np.sin(2*i*time)) for i in freq])-2*np.asarray([np.sum(weights*np.cos(i*time)) for i in freq])*np.asarray([np.sum(weights*np.sin(i*time)) for i in freq]))/(np.asarray([np.sum(weights*np.cos(2*i*time)) for i in freq])-(np.asarray([np.sum(weights*np.cos(i*time)) for i in freq])**2-np.asarray([np.sum(weights*np.sin(i*time)) for i in freq])**2)))/(2*freq)
    return np.asarray(tau)

def YY(data,freq,time,weights):
    data = np.asarray(data)
    freq = np.asarray(freq)
    time = np.asarray(time)
    weights = np.asarray(weights)
    yy=np.asarray(np.sum(weights*data**2))-np.asarray(np.sum(weights*data))**2                                   
    return np.asarray(yy)
    
def YC(data,freq,time,weights):
    data = np.asarray(data)
    freq = np.asarray(freq)
    time = np.asarray(time)
    weights = np.asarray(weights)
    wmatrix = np.asarray(list(zip(freq,tau(data,freq,time,weights))))
    yc = np.asarray([np.sum(weights*data*np.cos(i[0]*(time-i[1]))) for i in wmatrix]) - np.asarray(np.sum(weights*data))*np.asarray([np.sum(weights*np.cos(i[0]*(time-i[1]))) for i in wmatrix])
    return np.asarray(yc)
    
def CC(data,freq,time,weights):
    data = np.asarray(data)
    freq = np.asarray(freq)
    time = np.asarray(time)
    weights = np.asarray(weights)
    wmatrix = np.asarray(list(zip(freq,tau(data,freq,time,weights))))
    cc = np.asarray([np.sum(weights*np.cos(i[0]*(time-i[1]))**2) for i in wmatrix])-np.asarray([np.sum(weights*np.cos(i[0]*(time-i[1]))) for i in wmatrix])**2
    return np.asarray(cc)
    
def YS(data,freq,time,weights):
    data = np.asarray(data)
    freq = np.asarray(freq)
    time = np.asarray(time)
    weights = np.asarray(weights)
    wmatrix = np.asarray(list(zip(freq,tau(data,freq,time,weights))))
    ys = np.asarray([np.sum(weights*data*np.sin(i[0]*(time-i[1]))) for i in wmatrix]) - np.asarray(np.sum(weights*data))*np.asarray([np.sum(weights*np.sin(i[0]*(time-i[1]))) for i in wmatrix])
    return np.asarray(ys)
    
def SS(data,freq,time,weights):
    data = np.asarray(data)
    freq = np.asarray(freq)
    time = np.asarray(time)
    weights = np.asarray(weights)
    wmatrix = np.asarray(list(zip(freq,tau(data,freq,time,weights))))
    ss = np.asarray([np.sum(weights*np.sin(i[0]*(time-i[1]))**2) for i in wmatrix])-np.asarray([np.sum(weights*np.sin(i[0]*(time-i[1]))) for i in wmatrix])**2
    return np.asarray(ss)
    
def Lombs(data,freq,time,weights):
    data = np.asarray(data)
    freq = np.asarray(freq)
    time = np.asarray(time)
    weights = np.asarray(weights)
    return (len(data)-1)/(2)*(1/(YY(data,freq,time,weights))*(((YC(data,freq,time,weights))**2)/(CC(data,freq,time,weights))+((YS(data,freq,time,weights))**2)/(SS(data,freq,time,weights))))
    
def significance_levels(x,M,z):
    return 1 - (1 - np.exp(-x))**M-z
    

    
A_raw_data = np.loadtxt('hd000142.txt')
B_raw_data = np.loadtxt('hd027442.txt')
C_raw_data = np.loadtxt('hd102117.txt')

A_time = A_raw_data[:,0]
A_data = A_raw_data[:,1]
A_weights = 1/(np.sum(1/(A_raw_data[:,2])**2))*1/(A_raw_data[:,2])**2

B_time = B_raw_data[:,0]
B_data = B_raw_data[:,1]
B_weights = 1/(np.sum(1/(B_raw_data[:,2])**2))*1/(B_raw_data[:,2])**2

C_time = C_raw_data[:,0]
C_data = C_raw_data[:,1]
C_weights = 1/(np.sum(1/(C_raw_data[:,2])**2))*1/(C_raw_data[:,2])**2

freq_lim=[0,1]
freq=np.linspace(freq_lim[0], freq_lim[1], 10000, endpoint=True)
levels=((0.05,'-.'),(0.005,'--'),(0.001,':'))


#plt.plot(A_time,A_data)
#plt.scatter(A_time,A_data)
#plt.show()
#plt.plot(B_time,B_data)
#plt.scatter(B_time,B_data)
#plt.show()
#plt.plot(C_time,C_data)
#plt.scatter(C_time,C_data)
#plt.show()

plt.figure(figsize=(10,6))
for i in levels:
    args=(A_raw_data.shape[0],i[0])
    initguess=-np.log(args[1]/args[0])
    plt.hlines(fsolve(significance_levels,initguess,args=args),freq_lim[0],freq_lim[1],linestyles=i[1],label=str(i[0]))
plt.plot(freq,Lombs(A_data,freq,A_time,A_weights))
plt.title('hd000142 with ' + str(A_raw_data.shape[0]) + ' datapoints')
plt.xticks(np.linspace(freq_lim[0],freq_lim[1],10*freq_lim[1]+1, endpoint=True))
plt.xlim(-0.099,1.099)
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in levels:
    args=(A_raw_data.shape[0],i[0])
    initguess=-np.log(args[1]/args[0])
    plt.hlines(fsolve(significance_levels,initguess,args=args),freq_lim[0],freq_lim[1],linestyles=i[1],label=str(i[0]))
plt.plot(freq,Lombs(B_data,freq,B_time,B_weights))
plt.title('hd027442 with ' + str(B_raw_data.shape[0]) + ' datapoints')
plt.xticks(np.linspace(freq_lim[0],freq_lim[1],10*freq_lim[1]+1, endpoint=True))
plt.xlim(-0.099,1.099)
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in levels:
    args=(A_raw_data.shape[0],i[0])
    initguess=-np.log(args[1]/args[0])
    plt.hlines(fsolve(significance_levels,initguess,args=args),freq_lim[0],freq_lim[1],linestyles=i[1],label=str(i[0]))
plt.plot(freq,Lombs(C_data,freq,C_time,C_weights))
plt.title('hd102117 with ' + str(C_raw_data.shape[0]) + ' datapoints')
plt.xticks(np.linspace(freq_lim[0],freq_lim[1],10*freq_lim[1]+1, endpoint=True))
plt.xlim(-0.099,1.099)
plt.legend()
plt.show()