import LDPCHelper as LD
import Global as Global

class node:
    def __init__(self,ID,STATE):
        self.id=ID
        self.state=STATE
        self.value=0
        
class graph:
    def __init__(self):
        self.adj={}
    def addEdge(self,u,v):
        if u.id not in self.adj:
            self.adj[u.id]=[]
        self.adj[u.id].append(v)                

counter=0
g=graph()
def build(O,state):
    global g,counter
    if(len(state)==Global.Layers):
        counter=counter+1
        obj=node(counter,state)
        g.addEdge(O,obj)
        return
    tmp={}
    for i in state:
        tmp[i]=True
    for i in LD.NoToStr:
        isPresent=False
        for j in i:
            if j in tmp:
                isPresent=True
                break
        if isPresent==False:
            build(O,state+i)

if __name__=="__main__":
    LD.init()
    O=node(counter,"")
    build(O,"")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    