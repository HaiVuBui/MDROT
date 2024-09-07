import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from MDrot import Mdrot_gpu
from prepare_data import prepare_input
from MOT_models_Cupy_new import solve_multi_sinkhorn, solve_rrsinkhorn, solve_multi_greenkhorn, solve_pd_aam

def main(max_iter):
    # Define the data folder and parameters
    data_folder = 'data/size60/seed20/'
    X = np.fromfile(data_folder + 'exp_number0.npy')

    # Prepare the input data
    Cost, p, q, s = prepare_input(N=10)
    x0 = np.tensordot(p, np.tensordot(q, s, 0), 0)
    opt = (X * Cost.reshape(-1)).sum().item()

    # Combine p, q, and s into target_mu
    target_mu = np.concatenate([p, q, s], axis=0)

    # Ensure output directory exists
    output_folder = 'output_test'
    os.makedirs(output_folder, exist_ok=True)

    # Plot and save the image
    plt.figure()
    for i in range(-5, 5):
        ep = 10**i
        save_folder=output_folder+f'/A-{max_iter}'
        os.makedirs(save_folder, exist_ok=True)
        npy_filename = os.path.join(save_folder, f'A_Obj_list_ep{ep}.npy')
        M = Mdrot_gpu(x0, Cost, p, q, s, max_iters=max_iter, step=ep, compute_r_primal=True, eps_abs=1e-15, verbose=False, print_every=100)
        A=solve_multi_sinkhorn(Cost, target_mu, epsilon=ep, max_iter=max_iter)
        B=solve_rrsinkhorn(Cost, target_mu, epsilon=1, max_iter=max_iter)
        C=solve_multi_greenkhorn(Cost, target_mu, epsilon=1, max_iter=max_iter)
        print('===============================')
        print(f'Ep=10**{i} finished')
        Results={'M':M,'A':A,'B':B,'C':C}
        for Result in Results:
            save_folder=output_folder+f'/{Result}-{max_iter}'
            os.makedirs(save_folder, exist_ok=True)
            npy_filename = os.path.join(save_folder, f'{Result}_Obj_list_ep{ep}.npy')  
            np.save(npy_filename, Results[Result]['Obj_list'])

    # plt.plot(abs(A['Obj_list'] - opt), label='A')
    # plt.plot(abs(B['Obj_list'] - opt), label='B')
    # plt.plot(abs(C['Obj_list'] - opt), label='C')
    # plt.plot(abs(D['Obj_list'] - opt), label='D')
    # plt.plot(abs(M['Obj'] - opt), label='DR')
    plt.yscale('log')
    plt.legend()

    # Save the figure
    figure_filename = os.path.join(output_folder, f'A-{max_iter}.png')  # Change 'plot.png' to your desired file name and format
    plt.savefig(figure_filename, bbox_inches='tight')  # Save the figure to the specified folder

    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the plot script with a specified number of maximum iterations.')
    parser.add_argument('--max_iter', type=int, default=10000, help='The maximum number of iterations for the algorithm.')
    
    args = parser.parse_args()
    main(args.max_iter)
