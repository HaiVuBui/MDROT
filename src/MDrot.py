import numpy as np
import math
import numpy.linalg as nla
from time import time
from numba import cuda
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jna

def Mdrot_gpu(init, C, p, q, r, **kwargs):
    # Stopping parameters
    opt = kwargs.pop("opt", None)
    max_iters = kwargs.pop("max_iters", 100)
    eps_abs = kwargs.pop("eps_abs", 1e-6)

    # Stepsize parameters
    step = kwargs.pop("step", 1.0)
    max_step = kwargs.pop("max_step", 1.0)
    min_step = kwargs.pop("min_step", 1e-4)

    # Printing parameters
    verbose = kwargs.pop("verbose", False)
    print_every = kwargs.pop("print_every", 1)
    compute_r_primal = kwargs.pop("compute_r_primal", False)
    compute_r_dual = kwargs.pop("compute_r_dual", False)

    assert (max_step >= min_step), "Invalid range"

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
            r_dual[i] = abs(jnp.sum(x * C))
        
        x=x-(jnp.tensordot(xx,fg,axes=0)/(n*k)
             +jnp.tensordot(jnp.tensordot(e,yy,axes=0),g,axes=0)/(m*k)
             +jnp.tensordot(ef,zz,axes=0)/(m*n))


        if compute_r_primal:
            Ax = jnp.hstack((b1, b2, b3))
            r_primal[i] = jna.norm(Ax - b)
        if compute_r_primal or compute_r_dual:
            r_full = np.sqrt((r_primal[i]**2 + r_dual[i]**2))


        if (i % print_every == 0 or i == max_iters-1) and verbose:
            print("{}| {}  {}  {}  {}".format(str(i).rjust(6),
                                        format(r_full, ".5e").ljust(10),
                                        format(r_primal[i], ".5e").ljust(11),
                                        format(r_dual[i], ".5e").ljust(9),
                                        format(time() - start, ".2e").ljust(8)))
        i += 1
        done = (i >= max_iters) or (r_primal[i-1] <= eps_abs)
    
        end = time()
        runtime.append(end-start)

        if i%100==0:
            print(f'iter{i}')
    x=jnp.maximum(x-step*C,0.0)
    x=np.asarray(x)
    return {"sol":          x,
            "Obj_list":          np.asarray(objfunc),
            "distance":     np.asarray(distance),
            "primal":       np.asarray(r_primal),
            "dual":         jnp.array(r_dual),
            "num_iters":    i,
            "solve_time":   (end - start),
            'runtime':      runtime}
