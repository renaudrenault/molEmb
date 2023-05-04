import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda")

baseCoor=torch.tensor([
[0,0,0],
[1,0,0],
[-0.5000, 0.0000, 0.8660],
[-0.5000, -0.7500, -0.4330],
[-0.5000, 0.7500, -0.4330]
])

def rotate(P,ax,ay,az):
    S=P.clone()
    O=P.clone()
    S[:,1]=np.cos(ax)*O[:,1]-np.sin(ax)*O[:,2]
    S[:,2]=np.sin(ax)*O[:,1]+np.cos(ax)*O[:,2]
    O=S.clone()
    S[:,0]=np.cos(ay)*O[:,0]-np.sin(ay)*O[:,2]
    S[:,2]=np.sin(ay)*O[:,0]+np.cos(ay)*O[:,2]
    O=S.clone()
    S[:,0]=np.cos(az)*O[:,0]-np.sin(az)*O[:,1]
    S[:,1]=np.sin(az)*O[:,0]+np.cos(az)*O[:,1]
    return S

def translate(P,dx,dy,dz):
    S=P.clone()
    S[:,0]+=dx
    S[:,1]+=dy
    S[:,2]+=dz
    return S


Q=torch.concat([baseCoor[1::],-baseCoor[1::]])
#QP=torch.randn([3+nV]) #use to order atoms by type and pos
QP=torch.tensor([2700.0,1000.0,270.,90.,27.,9.0,3.0,1.0])
QP=QP.to(device)
Q=Q.to(device)



def mol2ToTensor(filename,AtomDic,nToken):
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    file1.close()
    c=0
    while c<len(Lines) and Lines[c]!='@<TRIPOS>MOLECULE\n':
        c+=1
    c+=1
    molname=Lines[c][0:-1]
    print(molname)
    while c<len(Lines) and Lines[c]!='@<TRIPOS>ATOM\n':
        c+=1
    c+=1
    O=[]
    while c<len(Lines) and Lines[c]!='@<TRIPOS>BOND\n':
        lineData=Lines[c].split()
        if '.' in lineData[5]:
            element=lineData[5].split('.')[0]
        else:
            element=lineData[5]
        O.append([float(lineData[2])]+[float(lineData[3])]+[float(lineData[4])]+AtomDic[element])
        c+=1
    molsize=len(O)
    O=torch.tensor(O)
    if len(O)>nToken:
        print('too big')
        return torch.tensor([[-1]]),molname,molsize
    if len(O)<nToken:
        O=torch.concat([O,torch.zeros(nToken-O.size(0),O.size(1))],dim=0)
    return O,molname,molsize

def loadMolFolder(folderPath,AtomDic,nToken):
    #load all files into a tensor
    X=[]
    names=[]
    sizes=[]
    for filename in os.listdir():
        print(filename)
        o,mol_name,mol_size=mol2ToTensor(filename,AtomDic,nToken)
        print(mol_name)
        if o.size(1)!=1:
            X.append(o.unsqueeze(0))
            names.append(mol_name)
            sizes.append(mol_size)
    X=torch.concat(X,dim=0)
    return X,names,sizes

import itertools

def createAllPermuteMat(nHead,nV):
    L=list(itertools.permutations(range(nHead), nHead))
    L=torch.tensor(L)
    c=0
    for j in range(L.size(0)):
        temp=torch.zeros(nHead*nV,nHead*nV)
        for i in range(nV):
            temp[L[j]*nV+i,L[0]*nV+i]=1
        if c==0:
            o=temp.clone()
            c=1
        else:
            o=torch.concat([o,temp],dim=1)
    return o

def getV(X,Q):
    D3= (X[:,:,0:3].unsqueeze(3).permute([0,1,3,2])-X[:,:,0:3].unsqueeze(3).permute([0,3,1,2]))
    for i in range(Q.shape[0]):
        D=((D3-Q[i].unsqueeze(0).unsqueeze(0).unsqueeze(0))**2).sum(3)
        D=torch.exp(-1*D)
        #D=torch.softmax(10*D,dim=2)
        #D=torch.softmax(10*D,dim=2)
        #D=torch.softmax(10*D,dim=2)
        temp=D.bmm(X[:,:,:])
        temp[:,:,0:3]=temp[:,:,0:3]-X[:,:,0:3]
        if i==0:
            V=temp
        else:
            V=torch.concat([V,temp],dim=2)
    return V


def distanceMatch(q,k):
    D=q.unsqueeze(-1).permute([0,1,3,2])-k.unsqueeze(-1).permute([0,3,1,2])
    D=(D**2).sum(dim=3)
    D=torch.exp(-D)
    return D

def getClosest(Y):
    D=Y.unsqueeze(-1).permute([0,2,1])-Y.unsqueeze(-1).permute([2,0,1])
    D=(D**2).sum(dim=2)
    D[D==0]=2*D.max()
    D=D.argmin(dim=1)
    return D

def sortV(V,QP):
    temp=V.reshape(V.shape[0]*V.shape[1],nHead,nV+3)
    temp2=QP.unsqueeze(0).repeat([V.shape[0]*V.shape[1],1]).unsqueeze(-1)
    ind=torch.argsort(temp.bmm(temp2).reshape([V.shape[0],V.shape[1],nhead]),dim=2)
    ind=ind*(nV+3)
    ind=ind.unsqueeze(3)
    ind2=ind.clone()
    for h in range(1,nV+3):
        ind2=torch.concat([ind2,ind+h],dim=3)
    ind2=ind2.reshape([V.shape[0],V.shape[1],(nV+3)*nhead])
    V2=V.clone()
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V2[i,j,:]=V[i,j,ind2[i,j,:]]
    return V2



class selfAttentionHeadMetric(torch.nn.Module):
    def __init__(self,nHead,nI,nH,nO):
        super().__init__()#dimO=dimI-3
        self.fc1 =nn.Linear(nI*nHead, nH)
        self.bn1=nn.BatchNorm1d(nH)
        self.fc2 =nn.Linear(nH, nH)
        self.bn2=nn.BatchNorm1d(nH)
        self.fc3 =nn.Linear(nH, nO)
        self.bn3=nn.BatchNorm1d(nO)
        self.nO=nO
        self.Q=torch.nn.parameter.Parameter(torch.randn(nHead,3)/10,requires_grad=True)
        #self.QP=torch.nn.parameter.Parameter(torch.tensor([0.0,0.0,0.0,100.0,10.0,1.0]),requires_grad=True)
        #self.register_parameter(name='Q', param=torch.nn.Parameter(torch.randn(5,3)))
    def forward(self, x):
        y0=getV(x,self.Q)
        y=y0.reshape([y0.shape[0]*y0.shape[1],y0.shape[2],1]).squeeze(-1)
        y=self.fc1(y)
        y=self.bn1(y)
        y=F.leaky_relu(y,0.05)
        y=self.fc2(y)
        y=self.bn2(y)
        y=F.leaky_relu(y,0.05)
        y=self.fc3(y)
        y=self.bn3(y)
        return y.unsqueeze(-1).reshape([y0.shape[0],y0.shape[1],self.nO])#y[:,0,:]#.sum(1)

class CustomNet(torch.nn.Module):
    def __init__(self,nLayer,nHead,nI,nH,nO):
        super().__init__()
        self.Layers=[]
        for i in range(nLayer):
            self.Layers.append(selfAttentionHeadMetric(nHead,nI,nH,nI-3))
        self.Lp=torch.nn.ParameterList(self.Layers)
        self.fc1 =nn.Linear(nI-3, nO)
    def forward(self, x, xmask,n):
        y=x.clone()
        for k in range(n):
            #y[:,:,3::]+=self.Layers[k](x)#.clone()
            y[:,:,3::]=F.leaky_relu(self.Layers[k](x)+x[:,:,3::],0.05)
            y[xmask]=0
            x=y.clone()
        y=self.fc1(y[:,:,3::])
        y=y.sum(1) #do the sum AFTER
        return y

class selfAttentionHead(torch.nn.Module):
    def __init__(self,nI,nQK,nV):
        super().__init__()
        self.fcK1 =nn.Linear(nI, nQK)
        self.fcQ1 =nn.Linear(nI, nQK)
        self.fcV1 =nn.Linear(nI, nV)
        self.distanceMatch=False
    def forward(self, x):
        k=self.fcK1(x)
        q=self.fcQ1(x)
        v=self.fcV1(x)
        if self.distanceMatch:
            Z=distanceMatch(q,k)
        else:
            Z=q.bmm(k.permute([0,2,1]))
        Z=torch.softmax(Z,dim=2)
        y=Z.bmm(v)
        return y

class selfAttentionLayer(torch.nn.Module):
    def __init__(self,nHead,nI,nQK,nV,nH,nO):
        super().__init__()
        self.heads=[]
        for i in range(nHead):
            self.heads.append(selfAttentionHead(nI,nQK,nV))
        self.fc1=nn.Linear(nV*nHead, nH)
        self.bn1=nn.BatchNorm1d(nH)
        self.fc2=nn.Linear(nH, nH)
        self.bn2=nn.BatchNorm1d(nH)
        self.fc3=nn.Linear(nH, nO)
        self.bn3=nn.BatchNorm1d(nO)
        self.head_parameters=torch.nn.ParameterList(self.heads)
        #self.z=torch.zeros()
    def forward(self, x):
        lesz=[]
        for head in self.heads:
            lesz.append(head(x))
        z=torch.concat(lesz,dim=2)
        y=self.fc1(z)
        y=self.bn1(y)
        y=F.leaky_relu(y,0.05)
        y=self.fc2(y)
        y=self.bn2(y)
        y=F.leaky_relu(y,0.05)
        y=self.fc3(y)
        y=self.bn3(y)
        return y


class MLSAN(torch.nn.Module):
    def __init__(self,nLayer,nHead,nI,nQK,nV,nH,nO):
        super().__init__()
        self.Layers=[]
        for i in range(nLayer):
            self.Layers.append(selfAttentionLayer(nHead,nI,nQK,nV,nI))
        self.Lp=torch.nn.ParameterList(self.Layers)
        self.fc1 =nn.Linear(nI, nO)
    def forward(self, x, xmask,n):
        y=x.clone()
        for k in range(n):
            y=F.leaky_relu(self.Layers[k](x)+x,0.05)
            x=y.clone()
        y=self.fc1(y)
        y=y.sum(1) #do the sum AFTER
        return y

def plotQ(Q):
    ax.cla()
    for i in range(len(Q)):
        ax.plot([0,Q[i,0].cpu().detach().numpy()],[0,Q[i,1].cpu().detach().numpy()],[0,Q[i,2].cpu().detach().numpy()],'k')
    ax.scatter(Q[:,0].cpu().detach().numpy(),Q[:,1].cpu().detach().numpy(),Q[:,2].cpu().detach().numpy(),s=100)
    ax.axes.set_xlim3d(left=-0.5, right=0.5)
    ax.axes.set_ylim3d(bottom=-0.5, top=0.5)
    ax.axes.set_zlim3d(bottom=-0.5, top=0.5)
    #ax.axis('equal')
    plt.show(block=False)
    plt.pause(0.01)
