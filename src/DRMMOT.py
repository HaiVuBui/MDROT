from math import isnan, nan
from os import wait
import numpy as np
from time import time
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jna

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

def main():
    exp_set=range(0,10)
    size = 20
    max_iter = 100
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


    out = drmmot(x0, Cost, p,q,s, max_iters=max_iter, step=ep, compute_r_primal=True, eps_abs=1e-15, verbose=False, print_every=100)
    print(out['objective_values'])
    print('finished')

if __name__ == '__main__':
    main()

