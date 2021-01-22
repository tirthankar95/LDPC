import EncoderPlusChannel as epc
import HG1 as hg1
import matplotlib.pyplot as plt
import numpy as np
import math
import Global
import threading

MODEL=5
alpha=0.5;beta=0.5
ITER=10
arrayStats=[]
# 0 -> original model.
# 1 -> alpha beta model.
# 2 -> min sum model
# 3 -> approximation model.
# 4 -> default model
selModel = [0,1,2,3]
def ATANH(L):
    (r,c)=L.shape
    for i in range(r):
        for j in range(c):
            if L[i,j]!=np.nan:
                try:
                    L[i,j]=math.atanh(L[i,j])
                except:
                    L[i,j]=math.atanh(np.sign(L[i,j])*0.99)
    return L
#My approximation of tanh using ML methods.
def tanh_mine(x):
    x[np.where(np.isnan(x)==True)]=-9999
    indx1 =np.where(( (x!=-9999) & (x>=-1.2) & (x<1.2)))
    indx2 = np.where(((x!=-9999) & (x >= 1.2)))
    indx3 = np.where(((x!=-9999) & (x < -1.2)))
    x[indx1]=0.8435*x[indx1]+0.0003
    x[indx2]=1
    x[indx3]=-1
    x[np.where(x==-9999)] = np.nan
    return x
#My approximation of atanh using ML methods.
def atanh_mine(x):
    x[np.where(np.isnan(x) == True)] = -9999
    indx1=np.where(( (x!=-9999) & (x<-0.85) ))
    indx2=np.where(( (x!=-9999) & (x>=-0.85) & (x<-0.75) ))
    indx3=np.where(( (x!=-9999) & (x>=-0.75) & (x< 0.75) ))
    indx4=np.where(( (x!=-9999) & (x>= 0.75) & (x< 0.85) ))
    indx5=np.where(( (x!=-9999) & (x>=0.85) ))
    x[indx1] = 6.944 * x[indx1] + 4.712
    x[indx2] = 1.777 * x[indx2] + 0.330
    x[indx3] = 1.231 * x[indx3] + 0.003
    x[indx4] = 1.815 * x[indx4] - 0.334
    x[indx5] = 8.347 * x[indx5] - 5.884
    x[np.where(x == -9999)] = np.nan
    return x
#Functions used for generating message bits.
def genAllUtil(tmp,k,num):
    if k==0:
        Global.msg.append(list(tmp))
        Global.example_cnt=Global.example_cnt+1
        return
    tmp.append(0)
    genAllUtil(tmp,k-1,num)
    if Global.example_cnt==num:
        return
    tmp.pop()
    tmp.append(1)
    genAllUtil(tmp,k-1,num)
    if Global.example_cnt==num:
        return
    tmp.pop()
def genAll(tmp,m):
    Global.msg=[];Global.example_cnt=0
    num=min(2**Global.k,m)
    genAllUtil(tmp,Global.k,num)
# General helper functions.
def demod(approx):
    res=approx
    indx=np.where(res>=0)
    indx1=np.where(res<0)
    res[indx]=0
    res[indx1]=1
    return res
def MIN(L):
    (r,c)=L.shape
    L[np.isnan(L)==True]=999
    for i in range(r):
        minF=minS=999
        for j in range(c):
            if minF>L[i,j]:
                minS=minF
                minF=L[i,j]
            elif minS>L[i,j]:
                minS=L[i,j]
        for j in range(c):
            if L[i,j]!=minF and L[i,j]!=999:
                L[i,j]=minF
            elif L[i,j]!=999:
                L[i,j]=minS
    L[L==999]=np.nan
    return L
##############################################################
def decode(H,c_Rx,std):
    mx_iter=4
    l_intrinsic = np.multiply(2 / std**2, c_Rx)
    d_bits=[[] for i in range(MODEL)]
    for model in selModel:
        lin = l_intrinsic
        L = np.multiply(H, lin)
        indx = np.where(L == 0)
        L[indx] = np.nan
        if model==0: # 0 -> original model.
            for i in range(mx_iter):
                L=np.tanh(L/2)
                L_=np.nanprod(L,axis=1).reshape(Global.n-Global.k,1)
                L=np.divide(L_, L)
                L=2*ATANH(L)
                lin=(lin+np.nansum(L,axis=0)).reshape(1,Global.n)
                L=lin-L
            d_bits[model]=demod(lin)
        if model==1: # 1 -> alpha beta model.
            for i in range(mx_iter):
                S=np.sign(L)
                S=np.nanprod(S,axis=1).reshape(Global.n-Global.k,1)
                l=MIN(np.abs(L))
                L=np.sign(L/S)*l
                L=alpha*L+beta
                lin=(lin+np.nansum(L,axis=0)).reshape(1,Global.n)
                L=lin-L
            d_bits[model]=demod(lin)
        if model==2: # 2 -> min sum model
            for i in range(mx_iter):
                S=np.sign(L)
                S=np.nanprod(S,axis=1).reshape(Global.n-Global.k,1)
                l=MIN(np.abs(L))
                L=np.sign(L/S)*l
                lin=(lin+np.nansum(L,axis=0)).reshape(1,Global.n)
                L=lin-L
            d_bits[model]=demod(lin)            
        if model==3: # 3 -> approximation model.
            for i in range(mx_iter):
                L = tanh_mine(L/2)
                L_ = np.nanprod(L, axis=1).reshape(Global.n - Global.k, 1)
                L = np.divide(L_, L)
                L = 2*atanh_mine(L)
                lin = (lin + np.nansum(L, axis=0)).reshape(1, Global.n)
                L = lin - L
            d_bits[model] = demod(lin)
        if model==4: # 4 -> default model
            d_bits[model] = demod(l_intrinsic)
    return np.array(d_bits)
#################### PARALLEL THREADS ########################
class myThread(threading.Thread):
	def __init__(self,threadID,name):
		threading.Thread.__init__(self)
		self.threadID=threadID # std value.
		self.name=name
	def run(self):
		#threadLock.acquire()
		RUN(self.threadID,self.name)
		#threadLock.release()

def RUN(i_upper,name):
	print('---',i_upper,' ~ ',name,'---')
	std=i_upper/10
	alpha, beta = mem[i_upper-1][0],mem[i_upper-1][1]#ml.train()
	c_Rx = epc.rx_message(c_encoded,std)
	for i in range(c_Rx.shape[0]): # FOR DIFF codes
		dec_bits=decode(H,c_Rx[i],std) ## get results for all the models.
		for model in selModel: # FOR DIFF models check the accuracy.
			if (dec_bits[model]==c_encoded[i]).all()==True:
				Global.code_err[model][i_upper - 1] +=1
# Calculating the BLER
	for model in range(MODEL):
		Global.code_err[model][i_upper-1]=1-Global.code_err[model][i_upper-1]/c_Rx.shape[0]
######################## MAIN ################################
tmp=[]
genAll(tmp,32)
H,c_encoded=hg1.encode(Global.msg)
#Decoding
Global.code_err=[[0 for i in range(1,ITER)] for j in range(MODEL)]
#Global.bit_err=[[0 for i in range(1,ITER)] for j in range(MODEL)]
mem=[(1.5014,-0.0022),(1.174,-0.055),(0.8513,0.0464),(0.6218,-0.0205),(0.4713,-0.008),(0.3703,-0.0255),(0.2715,0),(0.3035,-0.0126),(0.2597,0.0063)]
threads=[]

for i_upper in range(1,ITER):# FOR DIFF std
	tmpName="Thread-"+str(i_upper)
	threads.append(myThread(i_upper,tmpName))

for t in threads:
	t.start()

for t in threads:
	t.join()
####################### Plotting the results #################################
x_axis=np.array([10*math.log10((100/(i**2))) for i in range(1,ITER) ])
color=["rx-","bo-","g^-","kx-","yo-"]
for model in selModel:
    plt.plot(x_axis,np.array(Global.code_err[model]),color[model])
plt.legend(('Original','ML-method','Min_max','tnh_atnh approx'))
#plt.legend(('Original','ML-method','Min_max','tnh_atnh approx','No_Decoder'))
plt.xlabel('SNR(db)')
plt.ylabel('BLER (n=%d/k=%d)'%(Global.n,Global.k))
plt.savefig("output.png")
######################################################################
