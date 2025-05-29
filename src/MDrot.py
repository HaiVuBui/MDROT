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

    objective_values=np.zeros(max_iters)
    distances=np.zeros(max_iters)
    computation_time=[]
    done = False

    start = time()
    while not done:
        x=jnp.maximum(x-step*C,0.0)
        objective_values[iter]=(x*C).sum()

        if ground_truth is not None:
            distances[iter]=jna.norm(x.reshape(-1)-ground_truth).item()

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
        
        x=x-(jnp.tensordot(xx,fg,axes=0)/(n*k)
             +jnp.tensordot(jnp.tensordot(e,yy,axes=0),g,axes=0)/(m*k)
             +jnp.tensordot(ef,zz,axes=0)/(m*n))

        iter += 1
        done = (iter >= max_iters) #or (r_primal[iter-1] <= eps_abs)
    
        end = time()
        computation_time.append(end-start)

        if iter%100==0:
            print(f'iter{iter}')

    x=jnp.maximum(x-step*C,0.0)
    x=np.asarray(x)
    print("MDROT finished")
    return {"solution":          x,
            "objective_values":          np.asarray(objective_values),
            "distances":     np.asarray(distances),
            'computation_time':      computation_time}
