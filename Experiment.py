import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__),'src'))

import argparse
from MDrot import Mdrot_gpu
from prepare_data import prepare_input
from MOT_models_Cupy_new import solve_multi_sinkhorn, solve_rrsinkhorn, solve_multi_greenkhorn, solve_pd_aam

def single_experiment(alg,size,max_iter,ep): 
    exp_set=range(0,10)
    Result={}
    for exp_idx in exp_set:
        # Define the data folder and parameters
        data_folder = f'data/size{size}/seed20/'
        X = np.fromfile(data_folder + f'exp_number{exp_idx}.npy', dtype=np.float64)

        # Prepare the input data
        Cost, p, q, s = prepare_input(k=exp_idx,N=10,size=size)
        x0 = np.tensordot(p, np.tensordot(q, s, 0), 0)
        opt = (X * Cost.reshape(-1)).sum().item()

        # Combine p, q, and s into target_mu
        target_mu = np.concatenate([p, q, s], axis=0)

        
        if alg == 'M':
            temp = Mdrot_gpu(x0, Cost, p, q, s,opt=X ,max_iters=max_iter, step=ep, compute_r_primal=True, eps_abs=1e-15, verbose=False, print_every=100)
        elif alg == 'A':
            temp = solve_multi_sinkhorn(Cost, target_mu,opt=X, epsilon=ep, max_iter=max_iter)
        elif alg == 'B':
            temp = solve_rrsinkhorn(Cost, target_mu, opt=X, epsilon=ep, max_iter=max_iter)
        elif alg == 'C':
            temp = solve_multi_greenkhorn(Cost, target_mu,opt=X, epsilon=ep, max_iter=max_iter)
        elif alg == 'D':
            temp = solve_pd_aam(Cost,target_mu,opt=X ,epsilon0=ep, max_iterate=max_iter)

        if exp_idx==0:
            for key in ['Obj_list', 'runtime', 'distance']:
                if key=='Obj)list':
                    Result[key]=abs(temp[key]-opt)
                else:
                    Result[key]=temp[key]
        else:
            for key in ['Obj_list', 'runtime', 'distance']:
                if key=='Obj_list':
                    Result[key]+=abs(temp[key]-opt)
                else:
                    Result[key]+=temp[key]
    # for key in ['Obj_list', 'runtime', 'distance']:
    #     print(key)
    #     Result[key]/=10
    #     print(key)

    return Result

def main(max_iter,size):
    #parameters
      
    #sace folder
    #Ensure output directory exists
    output_folder = f'output/output_{size}/'+'max_iter-'+f'{max_iter}'
    os.makedirs(output_folder, exist_ok=True)

    algs={'M':range(-7,4), 'A':range(-3,3),'B':range(-3,3), 'C':range(-1,5)}  
    # Plot and save the image

    for alg in algs:
        for i in algs[alg]:
            #set epsilon
            ep=i

            #save folder and name
            save_folder = output_folder + f'/{alg}-{max_iter}'
            os.makedirs(save_folder, exist_ok=True)

            #check if computed 
            is_necessary = False
            for key in ['Obj_list', 'runtime', 'distance']:
                npy_filename = os.path.join(save_folder, f'{alg}_{key}{ep}.npy')
                if not os.path.exists(npy_filename):
                    is_necessary = True
            if not is_necessary:
                print(f'{alg} with epsilon = {ep} already exists. Skipping computation.')
                continue

            #compute
            Result=single_experiment(alg,size,max_iter,ep)
            print('===============================')
            print(f'Computing {alg} with epsilon = {ep} finished')
        
            # Save the Result to .npy files
            save_folder = output_folder + f'/{alg}-{max_iter}'
            for key in ['Obj_list', 'runtime', 'distance']:
                npy_filename = os.path.join(save_folder, f'{alg}_{key}{ep}.npy')
                if not os.path.exists(npy_filename):
                    np.save(npy_filename, Result[f'{key}'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the plot script with a specified number of maximum iterations.')
    parser.add_argument('--max_iter', type=int, default=10000, help='The maximum number of iterations for the algorithm.')
        
    parser.add_argument('--size', type=int, default=40, help='size')

    args = parser.parse_args()
    main(args.max_iter,args.size)
