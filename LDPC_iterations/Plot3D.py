import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#data is for avg case.
data={
	68:{20.000000:1,10.457575:1,6.020600:1,3.098039:1,0.900000:2},
	136:{20.000000:1,10.457575:1,6.020600:3,3.098039:3,0.900000:4},
	204:{20.000000:1,10.457575:1,6.020600:3,3.098039:4,0.900000:5},
	272:{20.000000:1,10.457575:1,6.020600:3,3.098039:4,0.900000:5},
	340:{20.000000:1,10.457575:1,6.020600:3,3.098039:4,0.900000:5},
	408:{20.000000:1,10.457575:1,6.020600:3,3.098039:4,0.900000:5},
        1020:{20.000000:1,10.457575:1,6.020600:3,3.098039:5,0.900000:5}
};
#data1 is for max case.
data1={
	68:{20.000000:1,10.457575:1,6.020600:1,3.098039:1,0.900000:3},
	136:{20.000000:1,10.457575:1,6.020600:1,3.098039:3,0.900000:4},
	204:{20.000000:1,10.457575:2,6.020600:3,3.098039:5,0.900000:6},
	272:{20.000000:1,10.457575:2,6.020600:4,3.098039:6,0.900000:6},
	340:{20.000000:1,10.457575:2,6.020600:4,3.098039:7,0.900000:7},
	408:{20.000000:1,10.457575:2,6.020600:4,3.098039:7,0.900000:7},
        1020:{20.000000:1,10.457575:3,6.020600:5,3.098039:7,0.900000:8}
};
x=[];y=[];z=[]
l_styles = ['-','--','-.',':']
l_color_styles=['crimson','deepskyblue','darkgreen','firebrick','black']
m_styles = ['o','^','*']

#data is for avg case.
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
                 ,s=10,edgecolor='black',linewidth=1,marker=m_styles[i%len(m_styles)])
    offset+=plotSize
ax.set_xlabel('Codeword Size.')
ax.set_ylabel('SNR.')
ax.set_zlabel('No. Of Iterations.')
plt.savefig("plotAvg.png")

#data1 is for max case.
x=[];y=[];z=[]
fig1 = plt.figure()
ax = plt.axes(projection='3d')
for k,v in data1.items():
    for kchild,vchild in v.items():
        x.append(k)
        y.append(kchild)
        z.append(vchild)
ITER=len(data1)
plotSize=len(z)//ITER
offset=0
for i in range(ITER):
    ax.plot3D(x[offset:(i+1)*plotSize],y[offset:(i+1)*plotSize],z[offset:(i+1)*plotSize],
              linestyle=l_styles[i%len(l_styles)],color=l_color_styles[i%len(l_color_styles)])
    ax.scatter3D(x[offset:(i+1)*plotSize],y[offset:(i+1)*plotSize],z[offset:(i+1)*plotSize],c=z[offset:(i+1)*plotSize]
                 ,s=10,edgecolor='black',linewidth=1,marker=m_styles[i%len(m_styles)])
    offset+=plotSize
ax.set_xlabel('Codeword Size.')
ax.set_ylabel('SNR.')
ax.set_zlabel('No. Of Iterations.')
plt.savefig("plotMax.png")
