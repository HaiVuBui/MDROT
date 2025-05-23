import numpy as np
from src.MDrot import Mdrot_gpu
from src.prepare_data import prepare_input

size = 60
exp_idx = 0
max_iter = 1000
ep = 1e-5

data_folder = f'data/size{size}/seed20/'
X = np.fromfile(data_folder + f'exp_number{exp_idx}.npy', dtype=np.float64)

Cost, p, q, s = prepare_input(k=exp_idx,N=10,size=size)
x0 = np.tensordot(p, np.tensordot(q, s, 0), 0)
opt = (X * Cost.reshape(-1)).sum().item()



out = Mdrot_gpu(x0, Cost, p, q, s,opt=X ,max_iters=max_iter, step=ep, compute_r_primal=True, eps_abs=1e-15, verbose=False, print_every=100)

print(out['sol'].shape)
print("done")
