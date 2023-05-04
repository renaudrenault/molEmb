import os,random

exec(open('../makemolecules.py').read())

def maLoss(Y_A,Y_P,Y_N):
    loss=0
    #loss+=0.1*((Y_A**2).sum(-1)-(400)**(2/3)).clip(min=0).mean(0)
    loss+=(((Y_A-Y_P)**2).sum(-1)).mean(0)
    loss+=10*(((1-(1e-10+(Y_A-Y_N)**2).sum(-1).sqrt()).clip(min=0))**2).mean(0)
    return loss


batchsize=26
nLayer=3
nRotation = 25
nToken=32
nHead=8
nV=11
nH=512
nOut=11

AtomDic={'H':[0,0,0,0,0,0,0,0,0,0,1],'C':[0,0,0,0,0,0,0,0,0,1,0],'N':[0,0,0,0,0,0,0,0,1,0,0],'O':[0,0,0,0,0,0,0,1,0,0,0],'F':[0,0,0,0,0,0,1,0,0,0,0],'S':[0,0,0,0,0,1,0,0,0,0,0],'P':[0,0,0,0,1,0,0,0,0,0,0],'B':[0,0,0,1,0,0,0,0,0,0,0],'Sn':[0,0,1,0,0,0,0,0,0,0,0],'Br':[0,1,0,0,0,0,0,0,0,0,0],'Cl':[1,0,0,0,0,0,0,0,0,0,0]}

#DATA contains 1 molecule in the batch direction
X,mol_names,mol_sizes2=loadMolFolder('..\molecules',AtomDic,nToken)
mol_indices=torch.tensor(list(range(X.size(0)))).to(device)
rot_indices =torch.tensor(list(range(nRotation))).to(device)
mol_indices=list(range(X.size(0)))
rot_indices =list(range(nRotation))
#dimension for rotameres ?
#dimension for rotation/translation,
R=[]
for i in range(nRotation):
    Xr=X.clone()
    for j in range(X.size(0)):
        Xr[j]=translate(rotate(Xr[j],2*np.pi*np.random.rand(),2*np.pi*np.random.rand(),0.0),np.random.randn(),np.random.randn(),np.random.randn())
    R.append(Xr.unsqueeze(1))

R=torch.concat(R,dim=1)
X=R.clone()
xmask=(X[:,:,:,3::]**2).sum(-1)==0
xmask=xmask.to(device)
X[xmask]=0

X=X.to(device)
mol_sizes=torch.tensor(mol_sizes2).to(device)

nActive=1
monnet=CustomNet(3,nHead,nV+3,nH,nOut).to(device)
monnet.load_state_dict(torch.load('../sd8.pt'))

L=[]
L1=[]
L2=[]
lesQ0=[]
lesQ1=[]
lesQ2=[]
lesY=[]
t=0
Y_average=monnet(X[:,0,:,:],xmask[:,0,:],nActive)
Y_average=Y_average.detach()
closest=getClosest(Y_average)

optimizer =torch.optim.Adam(monnet.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)
lossmin=100.0

batchsize=64

for t in range(t,300000):
    loss=0
    loss1=0
    loss2=0
    count=0
    for z in range(3):
        count+=1
        random.shuffle(mol_indices)
        random.shuffle(rot_indices)
        ind_AP=mol_indices[0:batchsize]
        ind_N=closest[ind_AP]
        X_A=X[ind_AP,rot_indices[0],:,:]
        X_P=X[ind_AP,rot_indices[1],:,:]
        X_N=X[ind_N,rot_indices[0],:,:]
        Y_A=monnet(X_A,xmask[ind_AP,0,:],nActive)
        Y_P=monnet(X_P,xmask[ind_AP,0,:],nActive)
        Y_N=monnet(X_N,xmask[ind_N,0,:],nActive)
        loss+=maLoss(Y_A,Y_P,Y_N)
        loss1+=(((Y_A-Y_P)**2).sum(-1)).mean(0).detach().cpu()
        loss2+=(((1-(1e-10+(Y_A-Y_N)**2).sum(-1).sqrt()).clip(min=0))**2).mean(0).detach().cpu()
        Y_average[mol_indices[0:batchsize]]*=0.9
        Y_average[mol_indices[0:batchsize]]+=Y_A.detach()*0.1
    loss.backward()
    optimizer.step()
    if t%1000==0:
        if loss<lossmin:
            lossmin=loss.detach().cpu()
            torch.save(monnet.state_dict(),'../sd8.pt')
        lesQ0.append(monnet.Layers[0].Q.detach().cpu().clone())
        lesQ1.append(monnet.Layers[1].Q.detach().cpu().clone())
        lesQ2.append(monnet.Layers[2].Q.detach().cpu().clone())
        lesY.append(Y_average.cpu().detach().clone())
    if t%100==0:
        closest=getClosest(Y_average)
    if t%10==0:
        print(t,loss/count)
        L.append((loss/count).detach().clone())
        L1.append((loss1/count).detach().clone())
        L2.append((loss2/count).detach().clone())
        plt.clf()
        _=plt.plot(torch.tensor(L).log10())
        #_=plt.plot(torch.tensor(L1).log10())
        #_=plt.plot(torch.tensor(L2).log10())
        plt.show(block=False)
        plt.pause(0.1)
        if (loss/count)<1e-10:
            break



plt.clf()
_=plt.plot(torch.tensor(L).log10())
_=plt.plot(torch.tensor(L1).log10())
_=plt.plot(torch.tensor(L2).log10())
plt.show(block=False)
plt.pause(0.1)

ax = plt.figure().add_subplot(projection='3d')
ax.cla()
lesColor='rgbkmyc'
for i in range(100):#(len(mol_names)):
    Y0=monnet(X[i,:,:,:],xmask[i,:,:],nActive).cpu().detach().numpy()
    _=ax.scatter(Y0[:,3],Y0[:,4],Y0[:,5],s=1,color=lesColor[i%7])
    #_=ax.text(Y_average[i,0].cpu().detach().numpy(),Y_average[i,1].cpu().detach().numpy(),Y_average[i,2].cpu().detach().numpy(),mol_names[i])

plt.show(block=False)

ax = plt.figure().add_subplot(projection='3d')
for i in range(len(lesQ0)):
    plotQ(lesQ0[i])


test=X[:,0]
aaa=test.clone()
aaa[:,:,3::]+=monnet.Layers[0](test)
aaa[xmask[:,0]]=0
bbb=aaa.clone()
bbb[:,:,3::]+=monnet.Layers[1](aaa)
bbb[xmask[:,0]]=0
ccc=bbb.clone()
ccc[:,:,3::]+=monnet.Layers[1](bbb)
ccc[xmask[:,0]]=0
ddd=monnet.fc1(ccc[:,:,3::])
ddd[xmask[:,0]]=0

i=13
plt.imshow(test[i,:].detach().cpu()),plt.show(block=False),plt.pause(2)
plt.imshow(aaa[i,:].detach().cpu()),plt.show(block=False),plt.pause(2)
plt.imshow(bbb[i,:].detach().cpu()),plt.show(block=False),plt.pause(2)
plt.imshow(ccc[i,:].detach().cpu()),plt.show(block=False),plt.pause(2)
plt.imshow(ddd[i,:].detach().cpu()),plt.show(block=False),plt.pause(2)






#qqq
