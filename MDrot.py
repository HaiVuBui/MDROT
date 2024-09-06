

import numpy as np
import math
import numpy.linalg as nla
from time import time
import cupy as cp
import cupy.linalg as cla
from numba import cuda

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
    
 #   OptSol=cp.array(OptSol)

    C=cp.array(C)
    p=cp.array(p)
    q=cp.array(q)
    r=cp.array(r)
    i = 0
    x = cp.array(init)
    m, n, k = x.shape
    e = cp.ones(m)
    f = cp.ones(n)
    g= cp.ones(k)

    fg=cp.tensordot(f,g,axes=0)
    eg=cp.tensordot(e,g,axes=0)
    ef=cp.tensordot(e,f,axes=0)

    a1 = cp.tensordot(x,fg,axes=2)
    a2 = cp.tensordot(cp.transpose(x,(1,0,2)),eg,axes=2)
    a3= cp.tensordot(ef,x,axes=2)
    b = cp.hstack((p, q, r))
    v=cp.zeros(max_iters)
    objfunc=cp.zeros(max_iters)
#    distance=cp.zeros(max_iters)
    r_primal = cp.zeros(max_iters)
    r_dual = cp.zeros(max_iters)
    r_full = cp.infty
    r_full0 = 0.0
    done = False
    restart = False

    start = time()
    while not done:
        #step=math.sqrt((math.sqrt(np.sum(x*x))+1)/(m+n))

        # Implicit F-order for Numba
        #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
        x=cp.maximum(x-step*C,0.0)
        objfunc[i]=(x*C).sum()
#        distance[i]=cla.norm(x.reshape(-1)-OptSol)


        b1= cp.tensordot(x,fg,axes=2)
        b2= cp.tensordot(x,eg,((0,2),(0,1)))
        b3= cp.tensordot(ef,x,axes=2)
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
        v[i]=math.sqrt(cp.sum(x*C))
        if compute_r_dual:
            # r_dual[k] = abs(np.sum(x * C) - (-yy.dot(p)/n - xx.dot(q)/m) / step)
            r_dual[i] = abs(cp.sum(x * C))
        
        #apply_adjoint_operator_and_override(e, f, yy, xx, x, -1.0/n, -1.0/m)
        x=x-(cp.tensordot(xx,fg,axes=0)/(n*k)
             +cp.tensordot(cp.tensordot(e,yy,axes=0),g,axes=0)/(m*k)
             +cp.tensordot(ef,zz,axes=0)/(m*n))


        if compute_r_primal:
            Ax = cp.hstack((b1, b2, b3))
            r_primal[i] = cla.norm(Ax - b)
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
    print("Drot terminated at iteration ", i-1)
    x=cp.maximum(x-step*C,0.0)
    x=cp.asnumpy(x)
    #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
    return {"sol":          x,
            "Obj_list":          cp.asnumpy(objfunc),
            # "distance":     cp.asnumpy(distance),
            "primal":       cp.asnumpy(r_primal),
            "dual":         cp.array(r_dual),
            "num_iters":    i,
            "solve_time":   (end - start)}

def Mdrot_gpu_32(init, C, p, q, r, **kwargs):
    # Stopping parameters
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
    
    C=cp.array(C,dtype=np.float32)
    p=cp.array(p,dtype=np.float32)
    q=cp.array(q,dtype=np.float32)
    r=cp.array(r,dtype=np.float32)
    i = 0
    x = cp.array(init,dtype=np.float32)
    m, n, k = x.shape
    e = cp.ones(m,dtype=np.float32)
    f = cp.ones(n,dtype=np.float32)
    g= cp.ones(k,dtype=np.float32)

    fg=cp.tensordot(f,g,axes=0)
    eg=cp.tensordot(e,g,axes=0)
    ef=cp.tensordot(e,f,axes=0)

    a1 = cp.tensordot(x,fg,axes=2)
    a2 = cp.tensordot(np.transpose(x,(1,0,2)),eg,axes=2)
    a3= cp.tensordot(ef,x,axes=2)
    b = cp.hstack((p, q, r))
    v=cp.zeros(max_iters)
    r_primal = cp.zeros(max_iters)
    r_dual = cp.zeros(max_iters)
    r_full = cp.infty
    r_full0 = 0.0
    done = False
    restart = False

    start = time()
    while not done:
        #step=math.sqrt((math.sqrt(np.sum(x*x))+1)/(m+n))

        # Implicit F-order for Numba
        #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
        x=cp.maximum(x-step*C,0.0)


        b1= cp.tensordot(x,fg,axes=2)
        b2= cp.tensordot(x,eg,((0,2),(0,1)))
        b3= cp.tensordot(ef,x,axes=2)
        t1 = 2 * b1 - a1 - p
        t2 = 2 * b2 - a2 - q
        t3 = 2 * b3 - a3 - r
        c1 =((n+k)*e.dot(t1) / (m*n + n*k + m*k)).astype(cp.float32)

        c2 = ((m+k)*f.dot(t2) / (m*n + n*k + m*k)).astype(cp.float32)

        c3 = ((m+n)*g.dot(t3) / (m*n + n*k + m*k)).astype(cp.float32)

        # Broadcasting
        xx = t1 - c1
        yy = t2 - c2
        zz = t3 - c3
        a1 = b1 - xx - c1
        a2 = b2 - yy - c2
        a3 = b3 - zz - c3
        v[i]=math.sqrt(cp.sum(x*C))
        if compute_r_dual:
            # r_dual[k] = abs(np.sum(x * C) - (-yy.dot(p)/n - xx.dot(q)/m) / step)
            r_dual[i] = abs(cp.sum(x * C))
        
        #apply_adjoint_operator_and_override(e, f, yy, xx, x, -1.0/n, -1.0/m)
        x=x-(cp.tensordot(xx,fg,axes=0)/(n*k)
             +cp.tensordot(cp.tensordot(e,yy,axes=0),g,axes=0)/(m*k)
             +cp.tensordot(ef,zz,axes=0)/(m*n))


        if compute_r_primal:
            Ax = cp.hstack((b1, b2, b3))
            r_primal[i] = cla.norm(Ax - b)
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
    print("Drot terminated at iteration ", i-1)
    x=cp.maximum(x-step*C,0.0)
    x=cp.asnumpy(x)
    #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
    return {"sol":          x,
            "primal":       cp.asnumpy(r_primal),
            "dual":         cp.array(r_dual),
            "num_iters":    k,
            "solve_time":   (end - start)}




def Mdrot_gpu_kernel(init, C, p, q, r, **kwargs):
    # Stopping parameters
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
    
    C=cp.array(C)
    p=cp.array(p)
    q=cp.array(q)
    r=cp.array(r)
    i = 0
    x = cp.array(init)
    m, n, k = x.shape
    e = cp.ones(m)
    f = cp.ones(n)
    g= cp.ones(k)

    fg=cp.tensordot(f,g,axes=0)
    eg=cp.tensordot(e,g,axes=0)
    ef=cp.tensordot(e,f,axes=0)

    a1 = cp.tensordot(x,fg,axes=2)
    a2 = cp.tensordot(np.transpose(x,(1,0,2)),eg,axes=2)
    a3= cp.tensordot(ef,x,axes=2)
    b = cp.hstack((p, q, r))
    v=cp.zeros(max_iters)
    r_primal = cp.zeros(max_iters)
    r_dual = cp.zeros(max_iters)
    r_full = cp.infty
    r_full0 = 0.0
    done = False
    restart = False

    start = time()
    while not done:
        #step=math.sqrt((math.sqrt(np.sum(x*x))+1)/(m+n))

        # Implicit F-order for Numba
        #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
        nonnegative_update[(25,25,25),(4,4,4)](x,C,step)


        b1= cp.tensordot(x,fg,axes=2)
        b2= cp.tensordot(x,eg,((0,2),(0,1)))
        b3= cp.tensordot(ef,x,axes=2)
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
        v[i]=math.sqrt(cp.sum(x*C))
        if compute_r_dual:
            # r_dual[k] = abs(np.sum(x * C) - (-yy.dot(p)/n - xx.dot(q)/m) / step)
            r_dual[i] = abs(cp.sum(x * C))
        
        #apply_adjoint_operator_and_override(e, f, yy, xx, x, -1.0/n, -1.0/m)
        #x=x-(cp.tensordot(xx,fg,axes=0)/(n*k)
        #     +cp.tensordot(cp.tensordot(e,yy,axes=0),g,axes=0)/(m*k)
        #     +cp.tensordot(ef,zz,axes=0)/(m*n))
        projection_update[(25,25,25),(4,4,4)](x,xx/(n*k),yy/(m*k),zz/(m*n))

        if compute_r_primal:
            Ax = cp.hstack((b1, b2, b3))
            r_primal[i] = cla.norm(Ax - b)
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
    print("Drot terminated at iteration ", i-1)
    nonnegative_update[(25,25,25),(4,4,4)](x,C,step)
    x=cp.asnumpy(x)
    #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
    return {"sol":          x,
            "primal":       np.array(r_primal[:k]),
            "dual":         np.array(r_dual[:k]),
            "num_iters":    k,
            "solve_time":   (end - start)}



def Mdrot_gpu_kernel(init, C, p, q, r, **kwargs):
    # Stopping parameters
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
    
    C=cp.array(C)
    p=cp.array(p)
    q=cp.array(q)
    r=cp.array(r)
    i = 0
    x = cp.array(init)
    m, n, k = x.shape
    e = cp.ones(m)
    f = cp.ones(n)
    g= cp.ones(k)

    fg=cp.tensordot(f,g,axes=0)
    eg=cp.tensordot(e,g,axes=0)
    ef=cp.tensordot(e,f,axes=0)

    a1 = cp.tensordot(x,fg,axes=2)
    a2 = cp.tensordot(np.transpose(x,(1,0,2)),eg,axes=2)
    a3= cp.tensordot(ef,x,axes=2)
    b = cp.hstack((p, q, r))
    v=cp.zeros(max_iters)
    r_primal = cp.zeros(max_iters)
    r_dual = cp.zeros(max_iters)
    r_full = cp.infty
    r_full0 = 0.0
    done = False
    restart = False

    start = time()
    while not done:
        #step=math.sqrt((math.sqrt(np.sum(x*x))+1)/(m+n))

        # Implicit F-order for Numba
        #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
        nonnegative_update[(25,25,25),(4,4,4)](x,C,step)


        b1= cp.tensordot(x,fg,axes=2)
        b2= cp.tensordot(x,eg,((0,2),(0,1)))
        b3= cp.tensordot(ef,x,axes=2)
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
        v[i]=math.sqrt(cp.sum(x*C))
        if compute_r_dual:
            # r_dual[k] = abs(np.sum(x * C) - (-yy.dot(p)/n - xx.dot(q)/m) / step)
            r_dual[i] = abs(cp.sum(x * C))
        
        #apply_adjoint_operator_and_override(e, f, yy, xx, x, -1.0/n, -1.0/m)
        #x=x-(cp.tensordot(xx,fg,axes=0)/(n*k)
        #     +cp.tensordot(cp.tensordot(e,yy,axes=0),g,axes=0)/(m*k)
        #     +cp.tensordot(ef,zz,axes=0)/(m*n))
        projection_update[(25,25,25),(4,4,4)](x,xx/(n*k),yy/(m*k),zz/(m*n))

        if compute_r_primal:
            Ax = cp.hstack((b1, b2, b3))
            r_primal[i] = cla.norm(Ax - b)
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
    print("Drot terminated at iteration ", i-1)
    nonnegative_update[(25,25,25),(4,4,4)](x,C,step)
    x=cp.asnumpy(x)
    
    #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
    return {"sol":          x,
            "primal":       np.array(r_primal[:k]),
            "dual":         np.array(r_dual[:k]),
            "num_iters":    k,
            "solve_time":   (end - start)}

def Mdrot_gpu_32_kernel(init, C, p, q, r, **kwargs):
    # Stopping parameters
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
    
    C=cp.array(C,dtype=np.float32)
    p=cp.array(p,dtype=np.float32)
    q=cp.array(q,dtype=np.float32)
    r=cp.array(r,dtype=np.float32)
    i = 0
    x = cp.array(init,dtype=np.float32)
    m, n, k = x.shape
    e = cp.ones(m,dtype=np.float32)
    f = cp.ones(n,dtype=np.float32)
    g= cp.ones(k,dtype=np.float32)

    fg=cp.tensordot(f,g,axes=0)
    eg=cp.tensordot(e,g,axes=0)
    ef=cp.tensordot(e,f,axes=0)

    a1 = cp.tensordot(x,fg,axes=2)
    a2 = cp.tensordot(np.transpose(x,(1,0,2)),eg,axes=2)
    a3= cp.tensordot(ef,x,axes=2)
    b = cp.hstack((p, q, r))
    v=cp.zeros(max_iters)
    r_primal = cp.zeros(max_iters)
    r_dual = cp.zeros(max_iters)
    r_full = cp.infty
    r_full0 = 0.0
    done = False
    restart = False

    start = time()
    while not done:
        #step=math.sqrt((math.sqrt(np.sum(x*x))+1)/(m+n))

        # Implicit F-order for Numba
        #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
        nonnegative_update[(25,25,25),(4,4,4)](x,C,step)


        b1= cp.tensordot(x,fg,axes=2)
        b2= cp.tensordot(x,eg,((0,2),(0,1)))
        b3= cp.tensordot(ef,x,axes=2)
        t1 = 2 * b1 - a1 - p
        t2 = 2 * b2 - a2 - q
        t3 = 2 * b3 - a3 - r
        c1 =((n+k)*e.dot(t1) / (m*n + n*k + m*k)).astype(cp.float32)

        c2 = ((m+k)*f.dot(t2) / (m*n + n*k + m*k)).astype(cp.float32)

        c3 = ((m+n)*g.dot(t3) / (m*n + n*k + m*k)).astype(cp.float32)

        # Broadcasting
        xx = t1 - c1
        yy = t2 - c2
        zz = t3 - c3
        a1 = b1 - xx - c1
        a2 = b2 - yy - c2
        a3 = b3 - zz - c3
        v[i]=math.sqrt(cp.sum(x*C))
        if compute_r_dual:
            # r_dual[k] = abs(np.sum(x * C) - (-yy.dot(p)/n - xx.dot(q)/m) / step)
            r_dual[i] = abs(cp.sum(x * C))
        
        #apply_adjoint_operator_and_override(e, f, yy, xx, x, -1.0/n, -1.0/m)
        projection_update[(25,25,25),(4,4,4)](x,xx/(n*k),yy/(m*k),zz/(m*n))


        if compute_r_primal:
            Ax = cp.hstack((b1, b2, b3))
            r_primal[i] = cla.norm(Ax - b)
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
    print("Drot terminated at iteration ", i-1)
    nonnegative_update[(25,25,25),(4,4,4)](x,C,step)
    x=cp.asnumpy(x)
    #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
    return {"sol":          x,
            "primal":       np.array(r_primal[:k]),
            "dual":         np.array(r_dual[:k]),
            "num_iters":    k,
            "solve_time":   (end - start)}

def Mdrot_gpu_quadratic_regularizated(init, C, p, q, r,alpha=0.02, **kwargs):
    # Stopping parameters
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
    
    C=cp.array(C)
    p=cp.array(p)
    q=cp.array(q)
    r=cp.array(r)
    i = 0
    x = cp.array(init)
    m, n, k = x.shape
    e = cp.ones(m)
    f = cp.ones(n)
    g= cp.ones(k)

    fg=cp.tensordot(f,g,axes=0)
    eg=cp.tensordot(e,g,axes=0)
    ef=cp.tensordot(e,f,axes=0)

    a1 = cp.tensordot(x,fg,axes=2)
    a2 = cp.tensordot(cp.transpose(x,(1,0,2)),eg,axes=2)
    a3= cp.tensordot(ef,x,axes=2)
    b = cp.hstack((p, q, r))
    v=cp.zeros(max_iters)
    Objfunc=np.zeros(max_iters)
    r_primal = cp.zeros(max_iters)
    r_dual = cp.zeros(max_iters)
    r_full = cp.infty
    r_full0 = 0.0
    done = False
    restart = False

    start = time()
    while not done:
        #step=math.sqrt((math.sqrt(np.sum(x*x))+1)/(m+n))

        # Implicit F-order for Numba
        #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
        x=(1/(1+alpha*step))*cp.maximum(x-step*C,0.0)
        Objfunc[i]=np.sum(x*C)

        b1= cp.tensordot(x,fg,axes=2)
        b2= cp.tensordot(x,eg,((0,2),(0,1)))
        b3= cp.tensordot(ef,x,axes=2)
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
        v[i]=math.sqrt(cp.sum(x*C))
        if compute_r_dual:
            # r_dual[k] = abs(np.sum(x * C) - (-yy.dot(p)/n - xx.dot(q)/m) / step)
            r_dual[i] = abs(cp.sum(x * C))
        
        #apply_adjoint_operator_and_override(e, f, yy, xx, x, -1.0/n, -1.0/m)
        x=x-(cp.tensordot(xx,fg,axes=0)/(n*k)
             +cp.tensordot(cp.tensordot(e,yy,axes=0),g,axes=0)/(m*k)
             +cp.tensordot(ef,zz,axes=0)/(m*n))


        if compute_r_primal:
            Ax = cp.hstack((b1, b2, b3))
            r_primal[i] = cla.norm(Ax - b)
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
    print("Drot terminated at iteration ", i-1)
    x=(1/(1+alpha*step))*cp.maximum(x-step*C,0.0)
    x=cp.asnumpy(x)
    #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
    return {"sol":          x,
            'Obj':          Objfunc,
            "primal":       cp.asnumpy(r_primal),
            "dual":         cp.asnumpy(r_dual),
            "num_iters":    k,
            "solve_time":   (end - start)}

def Mdrot_gpu_quadratic_regularizated_32(init, C, p, q, r,alpha, **kwargs):
    # Stopping parameters
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
    
    C=cp.array(C,dtype=np.float32)
    p=cp.array(p,dtype=np.float32)
    q=cp.array(q,dtype=np.float32)
    r=cp.array(r,dtype=np.float32)
    i = 0
    x = cp.array(init,dtype=np.float32)
    m, n, k = x.shape
    e = cp.ones(m,dtype=np.float32)
    f = cp.ones(n,dtype=np.float32)
    g= cp.ones(k,dtype=np.float32)

    fg=cp.tensordot(f,g,axes=0)
    eg=cp.tensordot(e,g,axes=0)
    ef=cp.tensordot(e,f,axes=0)

    a1 = cp.tensordot(x,fg,axes=2)
    a2 = cp.tensordot(np.transpose(x,(1,0,2)),eg,axes=2)
    a3= cp.tensordot(ef,x,axes=2)
    b = cp.hstack((p, q, r))
    v=cp.zeros(max_iters)
    r_primal = cp.zeros(max_iters)
    r_dual = cp.zeros(max_iters)
    r_full = cp.infty
    r_full0 = 0.0
    done = False
    restart = False

    start = time()
    while not done:
        x=(1/(1+alpha*step))*cp.maximum(x-step*C,0.0)


        b1= cp.tensordot(x,fg,axes=2)
        b2= cp.tensordot(x,eg,((0,2),(0,1)))
        b3= cp.tensordot(ef,x,axes=2)
        t1 = 2 * b1 - a1 - p
        t2 = 2 * b2 - a2 - q
        t3 = 2 * b3 - a3 - r
        c1 =((n+k)*e.dot(t1) / (m*n + n*k + m*k)).astype(cp.float32)

        c2 = ((m+k)*f.dot(t2) / (m*n + n*k + m*k)).astype(cp.float32)

        c3 = ((m+n)*g.dot(t3) / (m*n + n*k + m*k)).astype(cp.float32)

        # Broadcasting
        xx = t1 - c1
        yy = t2 - c2
        zz = t3 - c3
        a1 = b1 - xx - c1
        a2 = b2 - yy - c2
        a3 = b3 - zz - c3
        v[i]=math.sqrt(cp.sum(x*C))
        if compute_r_dual:
            # r_dual[k] = abs(np.sum(x * C) - (-yy.dot(p)/n - xx.dot(q)/m) / step)
            r_dual[i] = abs(cp.sum(x * C))
        
        
        x=x-(cp.tensordot(xx,fg,axes=0)/(n*k)
             +cp.tensordot(cp.tensordot(e,yy,axes=0),g,axes=0)/(m*k)
             +cp.tensordot(ef,zz,axes=0)/(m*n))


        if compute_r_primal:
            Ax = cp.hstack((b1, b2, b3))
            r_primal[i] = cla.norm(Ax - b)
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
        
    x=(1/(1+alpha*step))*cp.maximum(x-step*C,0.0)
    end = time()
    print("Drot terminated at iteration ", i-1)
    x=cp.maximum(x-step*C,0.0)
    x=cp.asnumpy(x)
    
    return {"sol":          x,
            "primal":       cp.asnumpy(r_primal),
            "dual":         cp.asnumpy(r_dual),
            "num_iters":    k,
            "solve_time":   (end - start)}

