import numpy as np
import math
import Global

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
#My approximation of atanh_mine_poly using ML methods.
def atanh_mine_poly(x):
    return 3.91*x-2.72*x+1.48*x
def genAll(m):
    Global.msg=[];Global.example_cnt=0
    num=min(2**Global.k,m)
    Global.msg=np.random.randint(low=0,high=2,size=(num,Global.k)).tolist()
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
def atanh_NNtrain():
    a=1