MX_ITER=25
MX_MSG=10
#initial number of groups of 3.
Z=1
N=68*Z
K=(68-46)*Z
NoWarnings=True
sz=(N-K) # number of rows in each layer.
SIGMA=0
# For checking whether decoding algorithm works or not.
TEST_CASE1=False
# For debugging
##########################################
def mode(arr):
    d={}
    for x in arr:
        if x==MX_ITER:
            continue
        d.setdefault(x,1)
        d[x]+=1
    mx=0;res=25
    for k,v in d.items():
        if mx<v:
            mx=v
            res=k
    return res

