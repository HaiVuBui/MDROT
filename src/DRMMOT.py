from math import isnan, nan
from os import wait
import numpy as np
from time import time
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jna
import cupy as cp
import cupy.linalg as cla
import math

from src.prepare_data import prepare_input

def drmmot(init, C, p, q, r, **kwargs):
    # Stopping parameters
    ground_truth = kwargs.pop("ground_truth", None)
    max_iters = kwargs.pop("max_iters", 100)
    eps_abs = kwargs.pop("eps_abs", 1e-6)

    # Stepsize parameters
    step = kwargs.pop("step", 1.0)
    max_step = kwargs.pop("max_step", 1.0)
    min_step = kwargs.pop("min_step", 1e-4)

    assert (max_step >= min_step), "Invalid range"

    if not ground_truth is None:
        ground_truth=jnp.array(ground_truth)

    C=jnp.array(C)
    p=jnp.array(p)
    q=jnp.array(q)
    r=jnp.array(r)
    iter = 0
    X = jnp.array(init)
    m, n, k = X.shape
    e = jnp.ones(m)
    f = jnp.ones(n)
    g= jnp.ones(k)

    fg=jnp.tensordot(f,g,axes=0)
    eg=jnp.tensordot(e,g,axes=0)
    ef=jnp.tensordot(e,f,axes=0)

    a1 = jnp.tensordot(X,fg,axes=2)
    a2 = jnp.tensordot(jnp.transpose(X,(1,0,2)),eg,axes=2)
    a3= jnp.tensordot(ef,X,axes=2)

    objective_values=np.zeros(max_iters)
    distances=np.zeros(max_iters)
    computational_time=[]
    done = False

    start = time()
    while not done:
        print((X*C).sum())
        X=jnp.maximum(X-step*C,0.0)
        print((X*C).sum())
        objective_values[iter]=(X*C).sum()

        if ground_truth is not None:
            distances[iter]=jna.norm(X.reshape(-1)-ground_truth).item()

        b1= jnp.tensordot(X,fg,axes=2)
        b2= jnp.tensordot(X,eg,((0,2),(0,1)))
        b3= jnp.tensordot(ef,X,axes=2)
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
        a1 = jnp.tensordot(X,fg,axes=2)
        a2 = jnp.tensordot(jnp.transpose(X,(1,0,2)),eg,axes=2)
        a3= jnp.tensordot(ef,X,axes=2)
        
        X=X-(jnp.tensordot(xx,fg,axes=0)/(n*k)
             +jnp.tensordot(jnp.tensordot(e,yy,axes=0),g,axes=0)/(m*k)
             +jnp.tensordot(ef,zz,axes=0)/(m*n))

        iter += 1
        done = (iter >= max_iters) #or (r_primal[iter-1] <= eps_abs)
    
        end = time()
        computational_time.append(end-start)

        # if iter%100==0:
        #     print(f'iter{iter}')

    X=jnp.maximum(X-step*C,0.0)
    X=np.asarray(X)
    # print("MDROT finished")
    return {'solution':              X,
            'objective_values':      np.asarray(objective_values),
            'distances':             np.asarray(distances),
            'computational_time':    computational_time}

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
        opt=cp.array(opt)

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
    distance=cp.zeros(max_iters)
    runtime=[]
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

        if opt is not None:
            distance[i]=cla.norm(x.reshape(-1)-opt).item()


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
        runtime.append(end-start)
    # print("Drot terminated at iteration ", i-1)
    x=cp.maximum(x-step*C,0.0)
    x=cp.asnumpy(x)
    #trace_nonnegative_prox_nb(x.T.reshape(-1), C.T.reshape(-1), step)
    return {"sol":          x,
            "Obj_list":          cp.asnumpy(objfunc),
            "distance":     cp.asnumpy(distance),
            "primal":       cp.asnumpy(r_primal),
            "dual":         cp.array(r_dual),
            "num_iters":    i,
            "solve_time":   (end - start),
            'runtime':      runtime}
def main():
    exp_set=range(0,10)
    size = 60
    max_iter = 1000
    ep = 1e-4

    exp_idx = 0
    # Define the data folder and parameters
    data_folder = f'data/size{size}/seed20/'
    X = np.fromfile(data_folder + f'exp_number{exp_idx}.npy', dtype=np.float64)

    # Prepare the input data
    Cost, p, q, s = prepare_input(k=exp_idx,N=10,size=size)
    x0 = np.tensordot(p, np.tensordot(q, s, 0), 0)
    opt = (X * Cost.reshape(-1)).sum().item()

    # Combine p, q, and s into target_mu
    target_mu = np.concatenate([p, q, s], axis=0)


    # out = drmmot(x0, Cost, p,q,s, max_iters=max_iter, step=ep, compute_r_primal=True, eps_abs=1e-15, verbose=False, print_every=100)
    out = Mdrot_gpu(x0, Cost, p, q, s,opt=X ,max_iters=max_iter, step=ep, compute_r_primal=True, eps_abs=1e-15, verbose=False, print_every=100)
    for i in range(max_iter):
        if i%100==0:
            print(out['Obj_list'])
        i+=1
    print(opt)
    print('finished')

    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(out['Obj_list'] - opt)
    plt.savefig('fig.png')


if __name__ == '__main__':
    main()

