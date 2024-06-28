
import numpy as np
import math
import numpy.linalg as nla
from time import time
import cupy as cp
import cupy.linalg as cla

def rho_gpu(a,b):
    n=len(a)
    e=cp.ones(n)
    f=e.dot(b-a)
    for i,x in enumerate(a):
        f+=a[i]*math.log(a[i]/b[i])
    return f

def beta_tensor_gpu(betas,C,eta):
    B=cp.add.outer(cp.add.outer(betas[0],betas[1]),betas[2])
    B+=-C/eta
    B=cp.exp(B)
    return B

def sinkhorn_gpu(C,p,q,s,OptSol,eta=1,eps=1e-5,max_iters=500):
    C=cp.array(C)
    p_gpu=cp.array(p)
    q_gpu=cp.array(q)
    s_gpu=cp.array(s)
    OptSol=cp.array(OptSol)

    m,n,k=C.shape
    betas=[cp.zeros(m),cp.zeros(n),cp.zeros(k)]
    res=cp.zeros(max_iters)
    res[0]=cp.infty
    objfunc=cp.zeros(max_iters)
    distance=cp.zeros(max_iters)
    t=0

    e = cp.ones(m)
    f = cp.ones(n)
    g= cp.ones(k)

    fg=cp.tensordot(f,g,axes=0)
    eg=cp.tensordot(e,g,axes=0)
    ef=cp.tensordot(e,f,axes=0)
    done=False
    while not done:
        B=beta_tensor_gpu(betas,C,eta)
        r1= cp.tensordot(B,fg,axes=2)
        r2= cp.tensordot(B,eg,((0,2),(0,1)))
        r3= cp.tensordot(ef,B,axes=2)

        a=cp.asnumpy(r1)
        b=cp.asnumpy(r2)
        c=cp.asnumpy(r3)

        dis=[p_gpu,q_gpu,s_gpu]
        r=[r1,r2,r3]
        
        k=np.argmax([rho(p,a),rho(q,b),rho(s,c)])
        betas[k]+=cp.log(dis[k])-cp.log(r[k])
        Ax = cp.hstack((r1-p_gpu,r2-q_gpu,r3-s_gpu))
        res[t]=cla.norm(Ax)
        
        objfunc[t]=(C*B).sum()
        distance[t]=cla.norm(B.reshape(-1)-OptSol)

        t+=1
        done=t>=max_iters or res[t-1]<eps
    B=cp.asnumpy(B)
    
    return {'sol':B,
            'Obj':cp.asnumpy(objfunc),
            'distance':cp.asnumpy(distance),
            'res':cp.asnumpy(res)}

def beta_tensor(betas,C,eta):
    B=np.add.outer(np.add.outer(betas[0],betas[1]),betas[2])
    B+=-C/eta
    B=np.exp(B)
    return B

def rho(a,b):
    n=len(a)
    e=np.ones(n)
    f=e.dot(b-a)
    for i,x in enumerate(a):
        f+=a[i]*(math.log(a[i])-math.log(b[i]))
    return f

# def rho(a,b):
#     n=len(a)
#     e=np.ones(n)
#     f=e.dot(b-a)
#     f+=np.sum(a*(np.log(a/b)))
#     return f


def sinkhorn(C,p,q,s,eta=1,eps=1e-5,max_iters=500):
    m,n,k=C.shape
    betas=[np.zeros(m),np.zeros(n),np.zeros(k)]

    res=np.infty
    t=0

    e = np.ones(m)
    f = np.ones(n)
    g= np.ones(k)

    fg=np.tensordot(f,g,axes=0)
    eg=np.tensordot(e,g,axes=0)
    ef=np.tensordot(e,f,axes=0)

    while t<max_iters and res>eps:
        B=beta_tensor(betas,C,eta)
        r1= np.tensordot(B,fg,axes=2)
        r2= np.tensordot(B,eg,((0,2),(0,1)))
        r3= np.tensordot(ef,B,axes=2)
        r=[r1, r2, r3]
        dis=[p,q,s]
        k=np.argmax([rho(p,r1),rho(q,r2),rho(s,r3)])
        betas[k]+=np.log(dis[k])-np.log(r[k])
        res=math.sqrt(np.sum(r1*r1)+np.sum(r2*r2)+np.sum(r3*r3))
        t+=1
    return B

def beta_tensor_2D(betas,C,eta):
    B=np.zeros(C.shape)
    for i,x in enumerate(B):
        for j,y in enumerate(x):
                B[i][j]=math.exp(betas[0][i]+betas[1][j]-C[i][j]/eta)
    return B
def sinkhorn_2D(C,p,q,eta=1,eps=1e-5,max_iters=500):
    m,n=C.shape
    betas=[np.zeros(m),np.zeros(n)]

    res=np.infty
    t=0

    e = np.ones(m)
    f = np.ones(n)


    while t<max_iters and res>eps:
        B=beta_tensor_2D(betas,C,eta)
        r1= B.dot(f)
        r2= B.T.dot(e)
        r=[r1, r2]
        dis=[p,q]
        k=np.argmax([rho(p,r1),rho(q,r2)])
        for i,b in enumerate(betas[k]):
            betas[k][i]+=math.log(dis[k][i])-math.log(r[k][i])
        res=math.sqrt(np.sum(r1*r1)+np.sum(r2*r2))
        t=t+1
    return B

def rounding(x0,p,q,s):
    x=x0
    m=len(p)
    n=len(q)
    k=len(s)
    e = np.ones(m)
    f = np.ones(n)
    g= np.ones(k)
    u=[e,f,g]
    fg=np.tensordot(f,g,axes=0)
    eg=np.tensordot(e,g,axes=0)
    ef=np.tensordot(e,f,axes=0)
    
    
    for i in range(3):
        dis=[p,q,s]
        r1= np.tensordot(x,fg,axes=2)
        r2= np.tensordot(x,eg,((0,2),(0,1)))
        r3= np.tensordot(ef,x,axes=2)
        r=[r1,r2,r3]
        z=np.minimum(u[i],dis[i]/r[i])
        for j,Z in enumerate(z):
            temp=[i]
            for k in range(3):
                if k!=i:
                    temp+=[k]
            np.transpose(x,temp)[j]*=Z
    r1= np.tensordot(x,fg,axes=2)
    r2= np.tensordot(x,eg,((0,2),(0,1)))
    r3= np.tensordot(ef,x,axes=2)
    err1=p-r1
    err2=q-r2
    err3=s-r3
    x+=np.tensordot(np.tensordot(err1,err2,0),err3,0)/tf.norm(err1,1)**2
    return x 
def sinkhorn_with_rounding(C,p,q,s,eps,max_iters):
    i=3
    m=len(p)
    n=len(q)
    k=len(s)
    e = np.ones(m)
    f = np.ones(n)
    g= np.ones(k)

    eta=eps/(2*m*math.log(n))
    eps_prime=eps/(8*tf.norm(C,np.inf))
    p_prime=(1-eps_prime/(4*m))*p+eps_prime/(4*m*n)*e
    q_prime=(1-eps_prime/(4*m))*q+eps_prime/(4*m*n)*f
    s_prime=(1-eps_prime/(4*m))*s+eps_prime/(4*m*n)*g
    B=sinkhorn_gpu(C,p_prime,q_prime,s_prime,eta=eta,eps=eps_prime,max_iters=max_iters)
    B=rounding(B,p_prime,q_prime,s_prime)
    return B