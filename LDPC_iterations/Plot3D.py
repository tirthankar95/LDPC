import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
data={
	136:{0.100000:1,0.300000:1,0.500000:3,0.700000:25,0.900000:25},
	150:{0.100000:6,0.300000:1,0.500000:3,0.700000:25,0.900000:25},
	166:{0.100000:2,0.300000:5,0.500000:3,0.700000:25,0.900000:25}
};
x=[];y=[];z=[]
l_styles = ['-','--','-.',':']
m_styles = ['','.','o','^','*']

fig = plt.figure()
ax = plt.axes(projection='3d')
for k,v in data.items():
    for kchild,vchild in v.items():
        x.append(k)
        y.append(kchild)
        z.append(vchild)
ax.scatter3D(x,y,z,c=z,cmap='viridis')
ax.set_xlabel('Codeword Size.')
ax.set_ylabel('Noise Variance.')
ax.set_zlabel('No. Of Iterations.')
plt.show()