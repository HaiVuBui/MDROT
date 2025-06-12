import jax
import jax.numpy as jnp
import jax.numpy.linalg as jna 
import numpy as np
from functools import reduce
from itertools import product
from math import log, sqrt
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import pickle as pkl
import time
import argparse
import torch
from time import time


def rotate(l, n):
    return l[n:] + l[:n]

def tensor_sum(list_of_lists):
    # return a tensor of size len(list_of_lists[0]) x ... x len(list_of_lists[-1]) where the value at index (i_1, ..., i_n) is the sum of the values at the same index in the lists
    shape = tuple(len(lst) for lst in list_of_lists)
    tensor = jnp.zeros(shape)
    m = len(shape)
    idxes = list(range(len(shape)))
    for idx, l in enumerate(list_of_lists):
        l = np.array(l.copy())
        tensor += torch.from_numpy(l).reshape(*([1]*idx + [-1] + [1]*(m - idx - 1))).expand(*shape).numpy()
    return tensor

def binary_search(a, b, f, delta = 1e-3):
    while True:
        c = (a+b)/2
        if f(c+delta) < f(c):
            a = c
        else:
            b = c
        if abs(a-b) <= delta:
            break
    return (a+b)/2

def larger_root_qr(a, b, c):
    delta = (b**2) - (4 * a*c)
    assert delta >= 0
    ans1 = (-b - sqrt(delta))/(2 * a)
    ans2 = (-b + sqrt(delta))/(2 * a)
    return max(ans1, ans2)

def create_normed_cost_tensor(list_list_of_lists):
    shape = tuple(len(lst) for lst in list_list_of_lists[0])
    res = jnp.zeros(shape)
    for list_of_lists in list_list_of_lists:
        tensor = tensor_sum(list_of_lists)
        res += (tensor / len(shape)) ** 2
    return res.astype(jnp.float64), jnp.array([1/n for n in shape for _ in range(n)]).astype(jnp.float64)

def create_cost_tensor(list_list_of_lists):
    shape = tuple(len(lst) for lst in list_list_of_lists[0])
    res = jnp.zeros(shape)
    for list_of_lists in list_list_of_lists:
        res += tensor_sum(list_of_lists) ** 2
    return res.astype(jnp.float64), jnp.array([1/n for n in shape for _ in range(n)]).astype(jnp.float64)

def create_cov_cost_tensor(list_list_of_lists):
    shape = tuple(len(lst) for lst in list_list_of_lists[0])
    res = jnp.ones(shape)
    for list_of_lists in list_list_of_lists:
        res *= tensor_sum(list_of_lists)
    return res.astype(jnp.float64), jnp.array([1/n for n in shape for _ in range(n)]).astype(jnp.float64)

def ravel_index(dim, index, shape):
    return sum([shape[i] for i in range(dim)]) + index

def convert_to_list(U, shape):
    U_list = []
    i = 0
    for s in shape:
        U_list.append(U[i:i+s])
        i += s
    return U_list
def marginal_k(X, k):
    # marginalize tensor X over dimension k
    return jnp.sum(X, axis=tuple(axis for axis in range(X.ndim) if axis != k))
def get_marginal_k(p, k, shape):
    return p[ravel_index(k, 0, shape): ravel_index(k, shape[k], shape)]

def projection(X, p):
    V = X.copy()
    m, shape = X.ndim, X.shape
    idxes = list(range(len(shape)))
    for r in range(m):
        X_r = jnp.minimum(get_marginal_k(p, r, shape) / marginal_k(V, r), 1)
        V *= jnp.transpose(jnp.tensordot(X_r, jnp.ones(shape[r+1:] + shape[:r]), axes=0), axes = rotate(idxes, -r))

    err_list = [get_marginal_k(p, r, shape) - marginal_k(V, r) for r in range(m)]
    # V += reduce(jnp.multiply, jnp.ix_(*err_list)) / (jnp.abs(err_list[-1]).sum() ** (m-1))
    V += reduce(jnp.multiply, jnp.ix_(*err_list)) / (jnp.abs(err_list[-1]).sum() ** (m-1))
    return V

def rho(a, b):
    return jnp.exp(a) @ (a - b)


def solve_lp(costs, target_mu):
    shape = costs.shape
    A = jnp.zeros((jnp.sum(shape), jnp.prod(jnp.array(shape))))
    for j, s in enumerate(shape):
        for i in range(s):
            for idx in product(*map(range, shape)):
                if idx[j] == i:
                    A[ravel_index(j, i, shape), jnp.ravel_multi_index(idx, shape)] = 1
    c = costs.flatten()
    res = linprog(c, A_eq=A, b_eq=target_mu, bounds=[0, 1])
    return res.fun, res.x.reshape(shape)

def solve_multi_sinkhorn(costs, target_mu, epsilon=1e-2, target_epsilon=1e-4, ground_truth = None, epsilon_scale_num = 0.99, epsilon_scale_gap = 100, cost_scale = 1, iter_gap = 100, max_iter = 5000, out_dir = 'test'):
    """solve using Sinkhorn's algorithm

    Args:
        costs (array): Cost matrix for the optimal transport problem.
        target_mu (list): Target probability distributions.
        epsilon (float): Starting epsilon for the Sinkhorn iterations. Defaults to 1e-2.
        target_epsilon (float): Target end epsilon for the Sinkhorn iterations. Defaults to 1e-4.
        verbose (int): Verbosity level. Print intermediate results if verbose >= 2. Defaults to 0.
        epsilon_scale_num (float): Factor to scale epsilon after `epsilon_scale_gap` iterations. Defaults to 0.99.
        epsilon_scale_gap (int): Number of iterations after which to scale epsilon. Defaults to 100.
        cost_scale (int, optional): Scale factor for the cost matrix. Divide the cost matrix by `cost_scale` before calculating, then scale back at the end. Defaults to 1.
        iter_gap (int, optional): Print results every `iter_gap` iterations if verbose >= 2. Defaults to 100.
        max_iter (int, optional): Maximum number of iterations to run. Defaults to 5000.
        out_dir (str, optional): Directory to save the log files. Defaults to 'test'.

    Returns:
        tuple: A tuple containing the following elements:
            - float: Final objective value.
            - float: Lower bound of the objective.
            - array: Weights of the optimal transport plan.

    References:
        Lin, Tianyi et al. (2022). “On the complexity of approximating multimarginal optimal transport”. In: The Journal of Machine Learning Research 23.1, pp. 2835–2877.
    """
    ########## initialization ###########
    
    costs /= cost_scale
    shape = costs.shape
    M = len(shape)
    # print("shape: ", shape)
    eta = 4 * sum([log (n) for n in shape]) / epsilon
    epsilon_prime = epsilon / 8 / costs.max()
    min_cost = costs.min()
    A_stable = jnp.exp(-eta * (costs - min_cost))
    A = jnp.exp(-eta * min_cost) * A_stable
    m = [jnp.zeros(s) for s in shape]
    B = A_stable / jnp.sum(jnp.abs(A_stable))
    iter = 0
    
    objective_values = []
    lb_list = []
    eps_list = []
    dis_list = []
    epsp_list = []
    computational_time = []
    distances = []
    
    ########## helper function ###########
    
    def dist(B):
        return sum(jnp.sum(jnp.abs(marginal_k(B, i) - get_marginal_k(target_mu, i, shape))) for i in range(M))
    def stable_update():
        target_marginal_list = []
        gamma_marginal_list = []
        rho_list = []
        for k in range(M):
            exponents = tensor_sum(m) - eta * costs
            idxes = list(range(M))
            idxes.remove(k)
            tmp_max = exponents.max(axis = tuple(idxes))
            tmp_max = np.array(tmp_max.copy())
            tmp_tensor = torch.from_numpy(tmp_max).reshape(*([1]*k + [-1] + [1]*(M - k - 1))).expand(*shape).numpy()
            target_marginal_list.append(jnp.log(get_marginal_k(target_mu, k, shape)))
            gamma_marginal_list.append(jnp.log(jnp.exp(exponents - tmp_tensor).sum(axis = tuple(idxes))) + tmp_max)
            rho_list.append(rho(target_marginal_list[k], gamma_marginal_list[k]))
        rho_list = jnp.array(rho_list)
        k_star = jnp.argmax(rho_list)
        m[k_star] += target_marginal_list[k_star] - gamma_marginal_list[k_star]
    
    
    ########## training starts ###########
    
    start = time()
    while True:
    # while iter < 500:
        if iter == 0: # update first step
            exponents = tensor_sum(m) - eta * costs
            idxes = list(range(M))
            idxes.remove(1)
            tmp_max = exponents.max(axis = tuple(idxes))
            tmp_max = np.array(tmp_max.copy())
            tmp_tensor = torch.from_numpy(tmp_max).reshape(*([1]*1 + [-1] + [1]*(M - 1 - 1))).expand(*shape).numpy()
            m[1] += jnp.log(get_marginal_k(target_mu, 1, shape)) - jnp.log(jnp.exp(exponents - tmp_tensor).sum(axis = tuple(idxes))) - tmp_max
        else:
            stable_update()
            
        B = jnp.exp(tensor_sum(m) - eta * costs)
        distance = dist(B)
        # primal objective
        obj = jnp.sum(projection(B, target_mu) * costs) * cost_scale
        # lower bound
        lb = sum([sum(get_marginal_k(target_mu, k, shape) * m[k]) for k in range(M)]) * cost_scale / eta     
        end = time()

        
    ########## logging results ###########
    
        eps_list.append(epsilon)
        dis_list.append(distance)
        epsp_list.append(epsilon_prime)
        objective_values.append(obj)
        lb_list.append(lb)
        if ground_truth is not None:
            dis = jna.norm( projection(B, target_mu).reshape(-1) - ground_truth)
            distances.append(dis)
        computational_time.append(end - start)


        
        if ((tensor_sum(m) - eta * costs) > 0).any().any():
            raise Exception("tensor_sum(m) can't be greater than eta * costs")

        
    ########## update parameters ###########
        iter += 1
        if distance < epsilon_prime:
            if abs(obj) > 4 * epsilon:
                break # stop when epsilon is small enough and converges
            else:
                if abs(obj) < 2 * target_epsilon:
                    raise Exception("obj too small. smaller than target 2 * eps: ", 2 * target_epsilon)
                elif  iter >= max_iter:
                    break
                else:
                    scale_factor = max(target_epsilon / epsilon, epsilon_scale_num)
                    epsilon *= scale_factor
                    epsilon_prime *= scale_factor
                    eta /= scale_factor
                    m = [l / scale_factor for l in m]
        else:
            if iter >= max_iter:
                break
            elif iter % epsilon_scale_gap == 0:
                scale_factor = max(target_epsilon / epsilon, epsilon_scale_num)
                epsilon *= scale_factor
                epsilon_prime *= scale_factor
                eta /= scale_factor
                m = [l / scale_factor for l in m]
    
    ########## return final results ###########
    
    B = jnp.exp(tensor_sum(m) - eta * costs)
    weights = projection(B, target_mu)
    lb = sum([sum(get_marginal_k(target_mu, k, shape) * m[k]) for k in range(M)]) * cost_scale / eta
    # print('multi sinkhorn finished')
    return {'solution': weights,
            'objective_values': objective_values,
            'distances': distances,
            'computational_time': computational_time
           }

def solve_rrsinkhorn(costs, target_mu, epsilon=1e-2, target_epsilon=1e-4, ground_truth = None, epsilon_scale_num = 0.99, epsilon_scale_gap = 100, cost_scale = 1, iter_gap = 100, max_iter = 5000, out_dir = 'test'):
    """solve using Sinkhorn's algorithm in a round robin fashion

    Args:
        costs (array): Cost matrix for the optimal transport problem.
        target_mu (list): Target probability distributions.
        epsilon (float): Starting epsilon for the Sinkhorn iterations. Defaults to 1e-2.
        target_epsilon (float): Target end epsilon for the Sinkhorn iterations. Defaults to 1e-4.
        verbose (int): Verbosity level. Print intermediate results if verbose >= 2. Defaults to 0.
        epsilon_scale_num (float): Factor to scale epsilon after `epsilon_scale_gap` iterations. Defaults to 0.99.
        epsilon_scale_gap (int): Number of iterations after which to scale epsilon. Defaults to 100.
        cost_scale (int, optional): Scale factor for the cost matrix. Divide the cost matrix by `cost_scale` before calculating, then scale back at the end. Defaults to 1.
        iter_gap (int, optional): Print results every `iter_gap` iterations if verbose >= 2. Defaults to 100.
        max_iter (int, optional): Maximum number of iterations to run. Defaults to 5000.
        out_dir (str, optional): Directory to save the log files. Defaults to 'test'.

    Returns:
        tuple: A tuple containing the following elements:
            - float: Final objective value.
            - float: Lower bound of the objective.
            - array: Weights of the optimal transport plan.

    References:
        Lin, Tianyi et al. (2022). “On the complexity of approximating multimarginal optimal transport”. In: The Journal of Machine Learning Research 23.1, pp. 2835–2877.
    """
    ########## initialization ###########
    
    costs /= cost_scale
    shape = costs.shape
    M = len(shape)
    eta = 4 * sum([log (n) for n in shape]) / epsilon
    epsilon_prime = epsilon / 8 / costs.max()
    min_cost = costs.min()
    A_stable = jnp.exp(-eta * (costs - min_cost))
    A = jnp.exp(-eta * min_cost) * A_stable
    m = [jnp.zeros(s) for s in shape]
    B = A_stable / jnp.sum(jnp.abs(A_stable))
    iter = 0
    
    objective_values = []
    lb_list = []
    eps_list = []
    dis_list = []
    epsp_list = []
    computational_time = []
    distances = []
    
    ########## helper function ###########
    
    def dist(B):
        return sum(jnp.sum(jnp.abs(marginal_k(B, i) - get_marginal_k(target_mu, i, shape))) for i in range(M))
    def stable_update():
        for k in range(M):
            exponents = tensor_sum(m) - eta * costs
            idxes = list(range(M))
            idxes.remove(k)
            tmp_max = exponents.max(axis = tuple(idxes))
            tmp_max = np.array(tmp_max.copy()) 
            tmp_tensor = torch.from_numpy(tmp_max).reshape(*([1]*k + [-1] + [1]*(M - k - 1))).expand(*shape).numpy()
            m[k] += jnp.log(get_marginal_k(target_mu, k, shape)) - jnp.log(jnp.exp(exponents - tmp_tensor).sum(axis = tuple(idxes))) - tmp_max
    
    ########## training starts ###########
    
    start = time()
    while True:
    # while iter < 500:
        stable_update()
        B = jnp.exp(tensor_sum(m) - eta * costs)
        distance = dist(B)
        # primal objective
        obj = jnp.sum(projection(B, target_mu) * costs) * cost_scale

        # lower bound
        lb = sum([sum(get_marginal_k(target_mu, k, shape) * m[k]) for k in range(M)]) * cost_scale / eta     
        end = time()
        
    ########## logging results ###########
    
        eps_list.append(epsilon)
        dis_list.append(distance)
        epsp_list.append(epsilon_prime)
        objective_values.append(obj)
        lb_list.append(lb)
        if ground_truth is not None:
            dis = jna.norm( projection(B, target_mu).reshape(-1) - ground_truth)
            distances.append(dis)
        computational_time.append(end-start)
        
        if ((tensor_sum(m) - eta * costs) > 0).any().any():
            raise Exception("tensor_sum(m) can't be greater than eta * costs")

    ########## update parameters ###########
        iter += 1
        if distance < epsilon_prime:
            if abs(obj) > 4 * epsilon:
                break # stop when epsilon is small enough and converges
            else:
                if abs(obj) < 2 * target_epsilon:
                    raise Exception("obj too small. smaller than target 2 * eps: ", 2 * target_epsilon)
                elif  iter >= max_iter:
                    break
                else:
                    scale_factor = max(target_epsilon / epsilon, epsilon_scale_num)
                    epsilon *= scale_factor
                    epsilon_prime *= scale_factor
                    eta /= scale_factor
                    m = [l / scale_factor for l in m]
        else:
            if iter >= max_iter:
                break
            elif iter % epsilon_scale_gap == 0:
                scale_factor = max(target_epsilon / epsilon, epsilon_scale_num)
                epsilon *= scale_factor
                epsilon_prime *= scale_factor
                eta /= scale_factor
                m = [l / scale_factor for l in m]
    
    ########## return final results ###########
    
    B = jnp.exp(tensor_sum(m) - eta * costs)
    weights = projection(B, target_mu)
    lb = sum([sum(get_marginal_k(target_mu, k, shape) * m[k]) for k in range(M)]) * cost_scale / eta
    return {'solution': weights,
            'objective_values': objective_values,
            'distances': distances,
            'computational_time': computational_time
           }

def Rho(a, b):
    return b - a + a * jnp.log(a / b)

def solve_multi_greenkhorn(costs, target_mu, epsilon=1e-2, target_epsilon=1e-4, ground_truth = None, epsilon_scale_gap = 100, cost_scale = 1, iter_gap = 100, max_iter = 5000, out_dir = 'test'):
    """Solve using the Greenkhorn algorithm

    Args:
        costs (array): Cost matrix for the optimal transport problem.
        target_mu (list): Target probability distributions.
        epsilon (float): Starting epsilon for the Greenkhorn iterations. Defaults to 1e-2.
        target_epsilon (float): Target end epsilon for the Greenkhorn iterations. Defaults to 1e-4.
        verbose (int): Verbosity level. Print intermediate results if verbose >= 2. Defaults to 0.
        epsilon_scale_num (float): Factor to scale epsilon after `epsilon_scale_gap` iterations. Defaults to 0.99.
        epsilon_scale_gap (int): Number of iterations after which to scale epsilon. Defaults to 100.
        cost_scale (int, optional): Scale factor for the cost matrix. Divide the cost matrix by `cost_scale` before calculating, then scale back at the end. Defaults to 1.
        iter_gap (int, optional): Print results every `iter_gap` iterations if verbose >= 2. Defaults to 100.
        max_iter (int, optional): Maximum number of iterations to run. Defaults to 5000.
        out_dir (str, optional): Directory to save the log files. Defaults to 'test'.

    Returns:
        tuple: A tuple containing the following elements:
            - float: Final objective value.
            - float: Lower bound of the objective.
            - array: Weights of the optimal transport plan.

    References:
        Altschuler, Jason, Jonathan Weed, and Philippe Rigollet (2018). “Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration.” arXiv: 1705.09634 [cs.DS].
    """
    costs /= cost_scale
    shape = costs.shape
    M = len(shape)
    epsilon = epsilon
    eta = 4 * sum([log(n) for n in shape]) / epsilon
    epsilon_prime = epsilon / 8 / costs.max()
    min_cost = costs.min()
    A_stable = jnp.exp(-eta * (costs - min_cost))
    A = jnp.exp(-eta * min_cost) * A_stable
    B = A_stable / jnp.sum(jnp.abs(A_stable))
    m = [jnp.zeros(s) for s in shape]
    iter = 0
    
    objective_values = []
    lb_list = []
    eps_list = []
    dis_list = []
    epsp_list = []
    distances = []
    computational_time = []
    
    def dist(B):
        return sum(jnp.sum(jnp.abs(marginal_k(B, i) - get_marginal_k(target_mu, i, shape))) for i in range(B.ndim))
        
    start = time()
    while True:
        max_v = []
        buffer = []
        for k in range(B.ndim):
            marginal = marginal_k(B, k)
            tmp = Rho(marginal, get_marginal_k(target_mu, k, shape))
            im = jnp.argmax(tmp)
            max_v.append(tmp[im])
            buffer.append((marginal[im], im))
        max_v = np.array(max_v)
        kmax = jnp.argmax(max_v)
        rki, imax = buffer[kmax]
        m[kmax].at[imax].set( m[kmax][imax] + jnp.log(target_mu[ravel_index(kmax, imax, shape)] / rki))
        B = jnp.exp(tensor_sum(m) - eta * costs)
        
        distance = dist(B)
        # primal objective
        obj = jnp.sum(projection(B, target_mu) * costs) * cost_scale
        # lower bound
        lb = sum([sum(get_marginal_k(target_mu, k, shape) * m[k]) for k in range(M)]) * cost_scale / eta     
        if ground_truth is not None:
            dis = jna.norm( projection(B, target_mu).reshape(-1) - ground_truth)
            distances.append(dis)
        end = time()
        
    ########## logging results ###########
    
        eps_list.append(epsilon)
        dis_list.append(distance)
        epsp_list.append(epsilon_prime)
        objective_values.append(obj)
        lb_list.append(lb)
        computational_time.append(end - start)
        
        #if iter % iter_gap == 0:
        #    memo = {"m": m, "objective_values":objective_values, "lb_list": lb_list, 
        #            "eps_list": eps_list, "dis_list": dis_list, "epsp_list": epsp_list,
        #            "iter": iter, "obj": obj, "lb": lb, "dist": distance, "epsp": epsilon_prime,
        #            "eps": epsilon, "eta": eta, 
        #            "iter_gap": iter_gap, "epsilon_scale_num": epsilon_scale_num, 
        #            "epsilon_scale_gap": epsilon_scale_gap, "cost_scale": cost_scale,
        #           "data_file": args.data_file, "start_epsilon": args.start_epsilon,
        #           "target_epsilon": target_epsilon, "error": None}
        #    pkl.dump(memo, open( 'log/' + out_dir + '.pkl', 'wb'))
        #    if verbose >= 2:
        #        print("iter: ", iter, 
        #              "obj: ", round(obj, 6), 
        #              "lb: ", round(lb, 6), 
        #              "dist: ", round(dist(B), 6),
        #              "eps: ", round(epsilon, 10),
        #              "eps_prime: ", round(epsilon_prime, 10),
        #             )
        
        
    ########## update parameters ###########
        iter += 1
        if distance < epsilon_prime:
            if abs(obj) > 4 * epsilon:
                break # stop when epsilon is small enough and converges
            else:
                if abs(obj) < 2 * target_epsilon:
                    raise Exception("obj too small. smaller than target 2 * eps: ", 2 * target_epsilon)
                elif  iter >= max_iter:
                    break
                else:
                    scale_factor = max(target_epsilon / epsilon, epsilon_scale_num)
                    epsilon *= scale_factor
                    epsilon_prime *= scale_factor
                    eta /= scale_factor
                    m = [l / scale_factor for l in m]
        else:
            if iter >= max_iter:
                break
            elif iter % epsilon_scale_gap == 0:
                scale_factor = max(target_epsilon / epsilon, epsilon_scale_num)
                epsilon *= scale_factor
                epsilon_prime *= scale_factor
                eta /= scale_factor
                m = [l / scale_factor for l in m]
    
    ########## return final results ###########
    
    B = jnp.exp(tensor_sum(m) - eta * costs)
    weights = projection(B, target_mu)
    lb = sum([sum(get_marginal_k(target_mu, k, shape) * m[k]) for k in range(M)]) * cost_scale / eta
    return {'solution': weights,
            'objective_values': objective_values,
            'distances': distances,
            'computational_time': computational_time
           }

def solve_pd_aam(costs, target_mu, epsilon_final = 1e-6, verbose = 0, print_itr = 0, max_iterate = 50, method = "binary_search", cost_scale = 1, epsilon0 = 0.01, halflife = 1, out_dir = 'test'):
    """Solve using the Accelerated Alternating Minimization (AAM) algorithm

    Args:
        costs (array): Cost matrix for the optimal transport problem.
        target_mu (list): Target probability distributions.
        epsilon_final (float): Final epsilon value for stopping criterion. Defaults to 1e-6.
        verbose (int): Verbosity level. Print intermediate results if verbose >= 2. Defaults to 0.
        print_itr (int): Iteration interval for printing intermediate results if verbose >= 2. Defaults to 0.
        max_iterate (int): Maximum number of iterations. Defaults to 50.
        method (str): Method for finding the minimum of the dual objective function ('binary_search', 'minimize', 'fix', or 'minimize_scalar'). Defaults to "binary_search".
        cost_scale (int, optional): Scale factor for the cost matrix. Divide the cost matrix by `cost_scale` before calculating, then scale back at the end. Defaults to 1.
        epsilon0 (float): Initial epsilon value for the iterations. Defaults to 0.01.
        halflife (float): Halflife parameter for epsilon decay. Defaults to 1.
        out_dir (str, optional): Directory to save the log files. Defaults to 'test'.

    Returns:
        tuple: A tuple containing the following elements:
            - float: Final objective value.
            - float: Lower bound of the objective.
            - array: Weights of the optimal transport plan.

    References:
        Tupitsa, Nazarii et al. (2020). “Multimarginal optimal transport by accelerated alternating minimization”. In: 2020 59th IEEE Conference on Decision and Control (CDC). IEEE, pp. 6132–6137.
    """
    costs /= cost_scale
    shape = costs.shape
    m, n = len(shape), max(shape) # (n = (5, 5, 8, 12))
    MAX_DIGIT = 25
    C_norm = costs.max().max()
    
    p = jnp.array(target_mu)
    uniform = jnp.array([1/n for n in shape for _ in range(n)])
    

    def l1_error(X, p_tilde):
        # Compute l1 error for each dimension and sum them
        errors = [jnp.abs(marginal_k(X, k) - get_marginal_k(p_tilde, k, X.shape)).sum() for k in range(X.ndim)]
        return jnp.sum(jnp.array(errors))


    def F(X, gamma):
        return (costs * X).sum().sum() - gamma * (X * jnp.log(X)).sum().sum()
    def B_stable(U, gamma):
        # convert U to a list of lists with list length equal to shape
        U_list = convert_to_list(U, shape)
        U_sum = tensor_sum(U_list)
        U_tmp = U_sum - costs / gamma
        max_U_tmp = U_tmp.max().max()
        U_tmp = jnp.maximum(U_tmp, max_U_tmp - MAX_DIGIT)
        min_U_tmp = U_tmp.min().min()
        return jnp.exp(U_tmp - min_U_tmp), min_U_tmp
    def log_B(U, gamma):
        exp_U_tmp, min_U_tmp = B_stable(U, gamma)
        return min_U_tmp + jnp.log(exp_U_tmp.sum().sum())
    def psi(U, p, gamma):
        return gamma * (log_B(U, gamma) - U @ p)
    def primal(U, gamma):
        exp_U_tmp, _ = B_stable(U, gamma)
        return exp_U_tmp / exp_U_tmp.sum().sum()
    def g_k(k, U, p, gamma):
        exp_U_tmp, _ = B_stable(U, gamma)
        return marginal_k(exp_U_tmp, k) / exp_U_tmp.sum().sum() - get_marginal_k(p, k, shape)
    def g(theta, p, gamma):
        return jnp.concatenate([g_k(k, theta, p, gamma) for k in range(m)])
    def g_l2_norm(theta, p, gamma):
        return jnp.sum(jnp.array([g_l2_norm_k(k, theta, p, gamma) for k in range(m)]))
    def g_l2_norm_k(k, theta, p, gamma):
        return jnp.square(g_k(k, theta, p, gamma)).sum() # marginal_k(U_tmp, k)
    def update_eta(theta, i, gamma): # lemma 2 iteration step
        eta = theta.copy()
        exp_U_tmp, min_U_tmp = B_stable(eta, gamma)
        log_B_marginal = min_U_tmp + jnp.log(jnp.sum(exp_U_tmp, axis=tuple(axis for axis in range(exp_U_tmp.ndim) if axis != i)))
        eta_update = get_marginal_k(theta, i, shape) + jnp.log(get_marginal_k(p, i, shape)) - log_B_marginal
        eta.at[ravel_index(i, 0, shape): ravel_index(i, shape[i], shape)].set(eta_update)
        return eta
    
    A = 0
    a = 0
    eta = jnp.zeros(len(p))
    zeta = jnp.zeros(len(p))
    theta = jnp.zeros(len(p))
    x0 = 0.1

    # TODO: what's the initialization for X?
    X = jnp.ones(costs.shape)

    objective_values = []
    dual_obj_list = []
    condition_list = []
    lb_obj_list = []
    max_lb_obj_list = []
    beta_list = []
    itr = 0
    computational_time = [] 
    
    
    epsilon = (epsilon0 - epsilon_final) * jnp.exp(-itr * halflife) + epsilon_final
    gamma = epsilon / (2 * m * jnp.log(n))
    epsilon_p = epsilon / (8 * C_norm)
    p_tilde = (1 - epsilon_p/(4 * m)) * jnp.array(target_mu) + epsilon_p / (4 * m) * uniform
    
    start = time()
    while 2 * l1_error(X, p_tilde) + F(X, gamma) + psi(eta, p_tilde, gamma) > epsilon / 2:
        if verbose >= 2 and itr % 100 == 0: print("="* 30, "itr", itr, "epsilon", epsilon, "gamma", gamma)
        itr += 1
        if itr > max_iterate:
            if verbose >= 2 and itr >= print_itr: print("exit %d iterations " % (max_iterate) + "!" * 80, 2 * l1_error(X, p_tilde) + F(X, gamma) + psi(eta, p_tilde, gamma), epsilon / 2)
            break
    
        # apply PD-AAM to the dual problem
        def f(beta):
            return psi(eta + beta * (zeta - eta), p_tilde, gamma)
        # find the minimum of f(beta)
        # add constraints that beta is in [0, 1]
        
        if method == "minimize":
            # method 1
            const = ({'type': 'ineq', 'fun': lambda beta: beta}, {'type': 'ineq', 'fun': lambda beta: 1 - beta})
            beta = max(min(optimize.minimize(f, x0 = x0, constraints=const).x[0], 1), 0)
        elif method == "binary_search":
            # method 2
            beta = binary_search(0, 1, f)
        elif method == "fix":
            beta = 0.1
        else:
            # method 3
            beta = optimize.minimize_scalar(f, bounds=(0, 1), method = 'bounded').x
        
        beta_list.append(beta)
        theta = eta + beta * (zeta - eta)
        if verbose >= 100 and itr >= print_itr: print("beta, eta, zeta, theta", beta, eta, zeta, theta)

        # pick margin
        
        i = jnp.argmax(jnp.array([g_l2_norm_k(k, theta, p_tilde, gamma) for k in range(m)]))
        
        eta = update_eta(theta, i, gamma)
        
        if verbose >= 100 and itr >= print_itr: print("theta, eta", theta, eta)
        diff = (psi(theta, p_tilde, gamma) - psi(eta, p_tilde, gamma))
        
        # TODO: hacky way to avoid unsolvable a
        if diff < 0: 
            if verbose >= 2 and itr >= print_itr: print("early stopping " + "!" * 80, tmp)
            a = 0
            # break
        else:
            tmp = diff / g_l2_norm(theta, p_tilde, gamma) 
            a = larger_root_qr(1, - 2 * tmp, - 2 * tmp * A)
        A_t = A
        A += a
        zeta = zeta - a * g(theta, p_tilde, gamma)
        
        dual_obj= - psi(theta, p, gamma) * cost_scale
        
        lb_obj = dual_obj - epsilon * jnp.log(jnp.prod(jnp.array(shape)))
        
        X = (a * primal(theta, gamma) + A_t * X) / A
        
        X = projection(X, p) # TODO: project X to the probability simplex (not in the original code)
        obj = (costs * X).sum().sum() * cost_scale
        gap = 2 * l1_error(X, p_tilde) + F(X, gamma) + psi(eta, p, gamma)

        end = time()
        
        
        
        condition_list.append(gap)
        objective_values.append(obj)
        lb_obj_list.append(lb_obj)
        max_lb_obj_list.append(max(lb_obj_list))
        dual_obj_list.append(dual_obj)
        computational_time.append(end - start)
        
        if verbose >= 2 and itr >= print_itr and itr % 100 == 0: 
            print("*", "obj: ", obj, "dual_obj", dual_obj, "lb_obj", lb_obj, "gap:", gap, "l1_error", l1_error(X, p_tilde), "F", F(X, gamma), "psi", psi(eta, p, gamma), "eps", epsilon / 2, "*")
            memo = {"objective_values":objective_values, "lb_list": lb_obj_list, 
                    "iter": iter, "obj": obj, "lb": lb_obj, 
                    "eps": epsilon / 2, "eta": eta, 
                    "cost_scale": cost_scale,
                   "data_file": args.data_file, "start_epsilon": args.start_epsilon,
                   "target_epsilon": args.target_epsilon, "error": None}
            pkl.dump(memo, open( 'log/' + out_dir + '.pkl', 'wb'))
        
        epsilon = (epsilon0 - epsilon_final) * jnp.exp(-itr * halflife) + epsilon_final
        gamma = epsilon / (2 * m * jnp.log(n))
        epsilon_p = epsilon / (8 * C_norm)
        p_tilde = (1 - epsilon_p/(4 * m)) * jnp.array(target_mu) + epsilon_p / (4 * m) * uniform
    
    print("*", "obj: ", obj, "dual_obj", dual_obj, "lb_obj", lb_obj, "max lb_obj", max_lb_obj_list[-1], "gap:", gap, "l1_error", l1_error(X, p_tilde), "F", F(X, gamma), "psi", psi(eta, p, gamma), "eps", epsilon / 2, "*")
    lb_obj = dual_obj - epsilon * jnp.log(jnp.prod(jnp.array(shape)))
        
    X = (a * primal(theta, gamma) + A_t * X) / A

    X = projection(X, p) # TODO: project X to the probability simplex (not in the original code)
    obj = (costs * X).sum().sum() * cost_scale
    # return obj, lb_obj, X
    return {'solution': X,
            'objective_values': objective_values,
            # 'distances': distances,
            'computational_time': computational_time
           }


def get_res_single(tmp_list, solver = solve_multi_sinkhorn, return_res = False, print_res = True, cost_type='square'):
    if cost_type == 'cov':
        cost_fn = create_cov_cost_tensor
    elif cost_type == 'normed_square':
        cost_fn = create_normed_cost_tensor
    else:
        cost_fn = create_cost_tensor
    costs, target_mu = cost_fn(tmp_list)
    lb = jnp.sum([jnp.mean([jnp.mean(t) for t in tmp])**2 for tmp in tmp_list])
    try:
        dis, dlb, weight = solver(costs, target_mu)
    except Exception as error:
        print("An exception occurred:", error)
    if print_res: print("shape:", weight.shape)
    if print_res: print("mean sq:", lb, "primal obj value: ", dis, "dual lb: ", dlb)
    if return_res: return dis, dlb, weight

if __name__ == "__main__":
    
    args = parse_args()
    out_dir = "solver-{:}data_file-{:}cost-{:}start_eps-{:}tgt_eps-{:}eps_scale-{:}scale_gap-{:}max_iter-{:}".format(args.solver, args.data_file, args.cost_type, args.start_epsilon, args.target_epsilon, args.epsilon_scale_num, args.epsilon_scale_gap, args.max_iter)
    print(args)
    tmp_list = pkl.load(open('data/' + args.data_file + ".pkl", "rb"))
    tmp_list = [[l.astype(jnp.float64) for l in tmp_l] for tmp_l in tmp_list]
    tmp_list = [[l - l.mean() for l in tmp_l] for tmp_l in tmp_list]
    tic = time.perf_counter()
    
    if args.solver == 'aam':
        dis, lb, weight = get_res_single(tmp_list, 
                                 solver = lambda a, b: solve_pd_aam(a, b, epsilon_final = args.target_epsilon, verbose = 2, max_iterate=args.max_iter, method = "binary_search", cost_scale = 1, halflife = 0.004, epsilon0=args.start_epsilon, out_dir = out_dir), return_res = True, cost_type = args.cost_type)
    else:
        if args.solver == "sinkhorn":
            solver_fn = solve_multi_sinkhorn
        elif args.solver == "rrsinkhorn":
            solver_fn = solve_rrsinkhorn
        elif args.solver == "greenkhorn":
            solver_fn = solve_multi_greenkhorn
        else:
            raise ValueError("Solver not found")
        dis, lb, weight = get_res_single(tmp_list, 
                                 solver = lambda a, b: solver_fn(a, b,
                                                                      epsilon=args.start_epsilon, 
                                                                      target_epsilon = args.target_epsilon, 
                                                                      verbose = args.verbose, 
                                                                      epsilon_scale_num = args.epsilon_scale_num, 
                                                                      epsilon_scale_gap = args.epsilon_scale_gap, 
                                                                      cost_scale = args.cost_scale, 
                                                                      iter_gap = args.iter_gap, 
                                                                      max_iter = args.max_iter,
                                                                      out_dir = out_dir), 
                                 return_res = True, cost_type = args.cost_type)
    
    toc = time.perf_counter()
    print(f"single run: {toc - tic} seconds with lb {lb}")
