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
    output_folder = 'output_test/'+'-'+f'{max_iter}'
    os.makedirs(output_folder, exist_ok=True)

    # Plot and save the image
    for i in range(-10, 10):
        ep = 10**i
        Results = {}
        for alg in ['M', 'A', 'B', 'C']:
            save_folder = output_folder + f'/{alg}-{max_iter}'
            os.makedirs(save_folder, exist_ok=True)
            npy_filename = os.path.join(save_folder, f'{alg}_Obj_list_ep{ep}.npy')
            
            # Check if the file already exists
            if os.path.exists(npy_filename):
                print(f'{npy_filename} already exists. Skipping computation.')
                continue
            
            # Compute the results if the file does not exist
            if alg == 'M':
                Results['M'] = Mdrot_gpu(x0, Cost, p, q, s, max_iters=max_iter, step=ep, compute_r_primal=True, eps_abs=1e-15, verbose=False, print_every=100)
            elif alg == 'A':
                Results['A'] = solve_multi_sinkhorn(Cost, target_mu, epsilon=ep, max_iter=max_iter)
            elif alg == 'B':
                Results['B'] = solve_rrsinkhorn(Cost, target_mu, epsilon=1, max_iter=max_iter)
            elif alg == 'C':
                Results['C'] = solve_multi_greenkhorn(Cost, target_mu, epsilon=1, max_iter=max_iter)
            
            print('===============================')
            print(f'Computing {alg} with epsilon = {ep} finished')
        
        # Save the results to .npy files
        for Result in Results:
            save_folder = output_folder + f'/{Result}-{max_iter}'
            os.makedirs(save_folder, exist_ok=True)
            npy_filename = os.path.join(save_folder, f'{Result}_Obj_list_ep{ep}.npy')
            np.save(npy_filename, Results[Result]['Obj_list'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the plot script with a specified number of maximum iterations.')
    parser.add_argument('--max_iter', type=int, default=10000, help='The maximum number of iterations for the algorithm.')
    
    args = parser.parse_args()
    main(args.max_iter)
