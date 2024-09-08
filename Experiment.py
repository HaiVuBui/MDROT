import numpy as np
import os
import argparse
from MDrot import Mdrot_gpu
from prepare_data import prepare_input
from MOT_models_Cupy_new import solve_multi_sinkhorn, solve_rrsinkhorn, solve_multi_greenkhorn, solve_pd_aam

def main(max_iter):
    # Define the data folder and parameters
    data_folder = 'data/size60/seed20/'
    X = np.fromfile(data_folder + 'exp_number0.npy', dtype=np.float64)

    # Prepare the input data
    Cost, p, q, s = prepare_input(N=10)
    x0 = np.tensordot(p, np.tensordot(q, s, 0), 0)
    opt = (X * Cost.reshape(-1)).sum().item()

    # Combine p, q, and s into target_mu
    target_mu = np.concatenate([p, q, s], axis=0)

    # Ensure output directory exists
    output_folder = 'output_test/'+'max_iter-'+f'{max_iter}'
    os.makedirs(output_folder, exist_ok=True)

    algs={'M':range(-8,-4), 'A':range(-3,3), 'B':range(-3,3), 'C':range(1,5), 'D':range(1,5)}  
    # Plot and save the image
    for alg in algs:
        for i in algs[alg]:
            ep=10**i
            save_folder = output_folder + f'/{alg}-{max_iter}'
            os.makedirs(save_folder, exist_ok=True)
            is_necessary = False
            for key in ['Obj_list', 'runtime', 'distance_list']:
                npy_filename = os.path.join(save_folder, f'{alg}_{key}_ep{ep}.npy')
                if not os.path.exists(npy_filename):
                    is_necessary = True
            if not is_necessary:
                print(f'{alg} with epsilon = {ep} already exists. Skipping computation.')
                continue
            
            
            # Compute the Result if the file does not exist
            if alg == 'M':
                Result = Mdrot_gpu(x0, Cost, p, q, s, max_iters=max_iter, step=ep, compute_r_primal=True, eps_abs=1e-15, verbose=False, print_every=100)
            elif alg == 'A':
                Result['A'] = solve_multi_sinkhorn(Cost, target_mu, epsilon=ep, max_iter=max_iter)
            elif alg == 'B':
                Result['B'] = solve_rrsinkhorn(Cost, target_mu, epsilon=ep, max_iter=max_iter)
            elif alg == 'C':
                Result['C'] = solve_multi_greenkhorn(Cost, target_mu, epsilon=ep, max_iter=max_iter)
            elif alg == 'D':
                Result['D'] = solve_pd_aam(Cost,target_mu, epsilon0=ep, max_iterate=max_iter)
            
            print('===============================')
            print(f'Computing {alg} with epsilon = {ep} finished')
        
        # Save the Result to .npy files
            save_folder = output_folder + f'/{alg}-{max_iter}'
            for key in ['Obj_list', 'runtime', 'distance_list']:
                npy_filename = os.path.join(save_folder, f'{alg}_{key}{ep}.npy')
                if not os.path.exists(npy_filename):
                    np.save(npy_filename, Result[f'{key}'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the plot script with a specified number of maximum iterations.')
    parser.add_argument('--max_iter', type=int, default=10000, help='The maximum number of iterations for the algorithm.')
    
    args = parser.parse_args()
    main(args.max_iter)
