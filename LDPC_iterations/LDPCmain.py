import numpy as np
import Global
import HG as hg
import time
import warnings
import random
H=[];l_intrinsic=[];Mat=[]
sigmas=[0.1,0.3,0.5,0.7,0.9]

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

def mod(codeword1):
    codeword=codeword1.copy()
    indx0=np.where(codeword==0)
    indx1=np.where(codeword==1)
    codeword[indx0]=1
    codeword[indx1]=-1
    return codeword

def demod(codeword1):
    codeword=codeword1.copy()
    indx0=np.where(codeword>0)
    indx1=np.where(codeword<=0)
    codeword[indx0]=0
    codeword[indx1]=1
    return codeword
    
def LDPC(): #Min Sum Decoder.
    global l_intrinsic,H
    l_intrinsic=np.multiply(2 / Global.SIGMA ** 2, l_intrinsic)
    iter_arr=[]
    for lin in l_intrinsic:
        iter=0
        L = np.multiply(H,lin)
        while iter<Global.MX_ITER:
            S=np.sign(L)
            S=np.nanprod(S,axis=1).reshape(Global.sz,1)
            l=MIN(np.abs(L))
            L=np.sign(L/S)*l
            lin=(lin+np.nansum(L,axis=0)).reshape(1,Global.N)
            res=np.dot(H,demod(lin).T)%2
            iter+=1
            if(np.all(res==0)):break
            L=lin-L
        iter_arr.append(iter)
        if Global.TEST_CASE1==True:
            TMP=demod(lin)[0]
            TMP=[int(x) for x in TMP]
            print('After LDPC codeword  -> ',TMP)
    return iter_arr

if __name__=='__main__':
    if Global.NoWarnings==True:
        warnings.filterwarnings("ignore")
    if Global.TEST_CASE1==True:
        Global.K=4;Global.N=6;Global.SIGMA=0.5
        Global.sz=(Global.N-Global.K)
        H=np.array([[1,0,1,0,1,0],[0,1,0,1,0,1]])
        l_intrinsic.append(mod(np.array([0,0,1,1,1,1]))+np.random.normal(0,Global.SIGMA,(1,Global.N)))
        print('Before LDPC codeword -> ',[0,0,1,1,1,1])
        print('Maximum Iterations = ',LDPC())
    #...
    else:
        np.random.seed(int(time.time()))
        msg=[[random.randint(0,1) for i in range(Global.K)] for j in range(Global.MX_MSG)]
        H,msg=hg.encode(msg)
        srcFile=open('plot.txt','a')
        print('%d:{' % (Global.N),file=srcFile,end='')
        for index,value in enumerate(sigmas):
            Global.SIGMA=value
            # Passing it through Transmitter and adding White noise.
            l_intrinsic=mod(msg)+np.random.normal(0,Global.SIGMA,(Global.MX_MSG,Global.N))
            if index==len(sigmas)-1: print('%lf:%d'%(Global.SIGMA,Global.mode(LDPC())),file=srcFile,end='')
            else: print('%lf:%d,'%(Global.SIGMA,Global.mode(LDPC())),file=srcFile,end='')
        print('},',file=srcFile)
        srcFile.close()
    print('Successfully Compiled.')
    #...