import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
data={
	136:{0.100000:1,0.300000:1,0.500000:3,0.700000:25,0.900000:25},
	150:{0.100000:6,0.300000:1,0.500000:3,0.700000:25,0.900000:25},
	166:{0.100000:2,0.300000:5,0.500000:3,0.700000:25,0.900000:25}
};
x=[];y=[];z=[]
l_styles = ['-','--','-.',':']
l_color_styles=['crimson','deepskyblue','darkgreen','firebrick','black']
m_styles = ['o','^','*']
cmap_styles=['Greens','Greens','Greens','Blues','Blues','Blues','Reds','Reds','Reds']

fig = plt.figure()
ax = plt.axes(projection='3d')
for k,v in data.items():
    for kchild,vchild in v.items():
        x.append(k)
        y.append(kchild)
        z.append(vchild)


ITER=len(data)
plotSize=len(z)//ITER
offset=0
for i in range(ITER):
    ax.plot3D(x[offset:(i+1)*plotSize],y[offset:(i+1)*plotSize],z[offset:(i+1)*plotSize],
              linestyle=l_styles[i%len(l_styles)],color=l_color_styles[i%len(l_color_styles)])
    ax.scatter3D(x[offset:(i+1)*plotSize],y[offset:(i+1)*plotSize],z[offset:(i+1)*plotSize],c=z[offset:(i+1)*plotSize]
                 ,s=100,cmap=cmap_styles[i%len(cmap_styles)],edgecolor='black',linewidth=1,marker=m_styles[i%len(m_styles)])
    offset+=plotSize
ax.set_xlabel('Codeword Size.')
ax.set_ylabel('Noise Variance.')
ax.set_zlabel('No. Of Iterations.')
plt.show()
