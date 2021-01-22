import numpy as np
import Global as Global
import time
import warnings

H=[]
def init():
    global H
    H=np.array([[1,0,0,1,0,0,1,0],[0,1,0,0,1,0,0,1],[1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1]])

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

def demod(codeword1):
    codeword=codeword1.copy()
    indx0=np.where(codeword>0)
    indx1=np.where(codeword<=0)
    codeword[indx0]=0
    codeword[indx1]=1
    return codeword
    
def LayeredLDPC0():
    global Mat,l_intrinsic
    iter=0
    lin=l_intrinsic
    LMat=[[0 for j in range(Global.n)] for i in range(Global.Layers)]
    while iter<Global.MX_ITER:
        for j in range(Global.Layers):
            lin=lin-LMat[j]
            L=np.multiply(Mat[j],lin)
            indx=np.where(L==0)
            L[indx]=np.nan
            S=np.sign(L)
            S=np.nanprod(S,axis=1).reshape(Global.sz,1)
            l=MIN(np.abs(L))
            L=np.sign(L/S)*l
            lin=(lin+np.nansum(L,axis=0)).reshape(1,Global.n)
            LMat[j]=np.nansum(L,axis=0).reshape(1,Global.n)
        res=np.dot(H,demod(lin).T)%2
        iter+=1
        if(np.all(res==0)):
            break
    return iter

if __name__=='__main__':
    if Global.NoWarnings==True:
        warnings.filterwarnings("ignore")
    #...
    init() #This function will load the H matrix.
    np.random.seed(int(time.time()))

    print('Successfully Compiled.')
    #...

    
    
