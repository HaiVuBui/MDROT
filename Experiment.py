import numpy as np
from MDrot import Mdrot_gpu
from prepare_data import prepare_input

data_folder='data/size60/seed20/'
X=np.fromfile(data_folder+'exp_number0.npy')

# C=np.fromfile(data_folder+'/exp_number0.npy')

C,p,q,s=prepare_input(N=10)
x0 = np.tensordot(p,np.tensordot(q,s,0),0)


A=Mdrot_gpu(x0, C, p, q, s, max_iters=1000, step=1e-6, compute_r_primal=True, eps_abs=1e-15, verbose=False, print_every=100)
print(A['Obj'][-1],(X*C.reshape(-1)).sum())
