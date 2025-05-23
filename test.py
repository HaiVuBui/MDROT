import torch
import jax.numpy as jnp
import numpy as np

list_of_list = [[1,2,3,4],[50,60,70,80,90],[1000,2000,3000,4000]]

def tensor_sum(list_of_lists):
    # return a tensor of size len(list_of_lists[0]) x ... x len(list_of_lists[-1]) where the value at index (i_1, ..., i_n) is the sum of the values at the same index in the lists
    shape = tuple(len(lst) for lst in list_of_lists)
    for lis in list_of_lists:
        print(len(lis))
    tensor = jnp.zeros(shape)
    m = len(shape)
    print("m:",m)
    idxes = list(range(len(shape)))
    for idx, l in enumerate(list_of_lists):
        l = np.array(l.copy())
        tensor += torch.from_numpy(l).reshape(*([1]*idx + [-1] + [1]*(m - idx - 1))).expand(*shape).numpy()
    print(tensor.shape)
    return tensor

def build_tensor(vector_list):
    shape = tuple(len(lst) for lst in list_of_list)




out = tensor_sum(list_of_list) 


print(out)
