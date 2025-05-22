import numpy as np
import math
import numpy.linalg as nla
from time import time
from numba import cuda
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jna

@cuda.jit
def nonnegative_update(X,C,step):
    m,n,l=C.shape

    x,y,z=cuda.grid(3)
    for i in range(x,m,100):
        for j in range(y,n,100):
            for k in range(z,l,100):
                X[(i,j,k)]-=step*C[(i,j,k)]
                if X[(i,j,k)]<0:
                    X[(i,j,k)]=0

@cuda.jit
def projection_update(X,phi1,phi2,phi3):
    m,n,l=X.shape
    
    x,y,z=cuda.grid(3)
    for i in range(x,m,100):
        for j in range(y,n,100):
            for k in range(z,l,100):
                X[(i,j,k)]-=phi1[i]+phi2[j]+phi3[k]


def Mdrot_gpu(init, C, p, q, r, **kwargs):
    # Stopping parameters
    opt = kwargs.pop("opt", None)
    max_iters = kwargs.pop("max_iters", 100)
    eps_abs = kwargs.pop("eps_abs", 1e-6)
    eps_rel = kwargs.pop("eps_rel", 1e-15)

    # Stepsize parameters
    step = kwargs.pop("step", 1.0)
    adapt_stepsize = kwargs.pop("adapt_stepsize", False)
    incr = kwargs.pop("incr", 2.0)
    decr = kwargs.pop("decr", 2.0)
    mu = kwargs.pop("mu", 20)
    max_step = kwargs.pop("max_step", 1.0)
    min_step = kwargs.pop("min_step", 1e-4)

    

    # Restart parameters
    fixed_restart = kwargs.pop("fixed_restart", False)
    milestones = kwargs.pop("milestones", [])
    adapt_restart = kwargs.pop("adapt_restart", False)

    # Printing parameters
    verbose = kwargs.pop("verbose", False)
    print_every = kwargs.pop("print_every", 1)
    compute_r_primal = kwargs.pop("compute_r_primal", False)
    compute_r_dual = kwargs.pop("compute_r_dual", False)

    assert (max_step >= min_step), "Invalid range"
    #assert C.flags['F_CONTIGUOUS']

    if verbose:
        print("----------------------------------------------------")
        print(" iter | total res | primal res | dual res | time (s)")
        print("----------------------------------------------------")
    
    if not opt is None:
        opt=jnp.array(opt)

    C=jnp.array(C)
    p=jnp.array(p)
    q=jnp.array(q)
    r=jnp.array(r)
    i = 0
    x = jnp.array(init)
    m, n, k = x.shape
    e = jnp.ones(m)
    f = jnp.ones(n)
    g= jnp.ones(k)

    fg=jnp.tensordot(f,g,axes=0)
    eg=jnp.tensordot(e,g,axes=0)
    ef=jnp.tensordot(e,f,axes=0)

    a1 = jnp.tensordot(x,fg,axes=2)
    a2 = jnp.tensordot(jnp.transpose(x,(1,0,2)),eg,axes=2)
    a3= jnp.tensordot(ef,x,axes=2)
    b = jnp.hstack((p, q, r))

    v=np.zeros(max_iters)
    objfunc=np.zeros(max_iters)
    distance=np.zeros(max_iters)
    runtime=[]
    r_primal = np.zeros(max_iters)
    r_dual = np.zeros(max_iters)
    r_full = jnp.inf
    r_full0 = 0.0
    done = False
    restart = False

    start = time()
    while not done:
        #step=math.sqrt((math.sqrt(np.sum(x*x))+1)/(m+n))

        # Implicit F-order for Numba
        #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
        x=jnp.maximum(x-step*C,0.0)
        objfunc[i]=(x*C).sum()

        if opt is not None:
            distance[i]=jna.norm(x.reshape(-1)-opt).item()


        b1= jnp.tensordot(x,fg,axes=2)
        b2= jnp.tensordot(x,eg,((0,2),(0,1)))
        b3= jnp.tensordot(ef,x,axes=2)
        t1 = 2 * b1 - a1 - p
        t2 = 2 * b2 - a2 - q
        t3 = 2 * b3 - a3 - r
        c1 = (n+k)*e.dot(t1) / (m*n + n*k + m*k)
        c2 = (m+k)*f.dot(t2) / (m*n + n*k + m*k)
        c3 = (m+n)*g.dot(t3) / (m*n + n*k + m*k)

        # Broadcasting
        xx = t1 - c1
        yy = t2 - c2
        zz = t3 - c3
        a1 = b1 - xx - c1
        a2 = b2 - yy - c2
        a3 = b3 - zz - c3
        v[i]=math.sqrt(jnp.sum(x*C))
        if compute_r_dual:
            # r_dual[k] = abs(np.sum(x * C) - (-yy.dot(p)/n - xx.dot(q)/m) / step)
            r_dual[i] = abs(jnp.sum(x * C))
        
        #apply_adjoint_operator_and_override(e, f, yy, xx, x, -1.0/n, -1.0/m)
        x=x-(jnp.tensordot(xx,fg,axes=0)/(n*k)
             +jnp.tensordot(jnp.tensordot(e,yy,axes=0),g,axes=0)/(m*k)
             +jnp.tensordot(ef,zz,axes=0)/(m*n))


        if compute_r_primal:
            Ax = jnp.hstack((b1, b2, b3))
            r_primal[i] = jna.norm(Ax - b)
            #r_primal[i]=math.sqrt(np.sum(x*C))
        if compute_r_primal or compute_r_dual:
            r_full = np.sqrt((r_primal[i]**2 + r_dual[i]**2))
            if i == 0:
                r_full0 = r_full

                     

        if (i % print_every == 0 or i == max_iters-1) and verbose:
            print("{}| {}  {}  {}  {}".format(str(i).rjust(6),
                                        format(r_full, ".5e").ljust(10),
                                        format(r_primal[i], ".5e").ljust(11),
                                        format(r_dual[i], ".5e").ljust(9),
                                        format(time() - start, ".2e").ljust(8)))
        i += 1
        # done = (k >= max_iters) or (r_full <= eps_abs + eps_rel * r_full0)
        done = (i >= max_iters) or (r_primal[i-1] <= eps_abs)
        
    
        end = time()
        runtime.append(end-start)
    # print("Drot terminated at iteration ", i-1)
    x=jnp.maximum(x-step*C,0.0)
    x=np.asarray(x)
    #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
    return {"sol":          x,
            "Obj_list":          np.asarray(objfunc),
            "distance":     np.asarray(distance),
            "primal":       np.asarray(r_primal),
            "dual":         jnp.array(r_dual),
            "num_iters":    i,
            "solve_time":   (end - start),
            'runtime':      runtime}
