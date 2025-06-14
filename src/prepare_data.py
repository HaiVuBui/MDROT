from jax._src.lax.lax import rng_bit_generator
import jax.numpy as jnp
import numpy as np
# from numba import cuda
from keras.datasets import mnist

from src.make_ellipse import make_nested_ellipses


def  weighted_outer_sum( array, vector, a,b):
    return np.add.outer(a*array,b*vector)

def weighted_sum(vector_list,weight):
    sum=vector_list[0]
    for i in range(len(weight)-1):
        sum= weighted_outer_sum(sum,vector_list[i+1],1,weight[i+1])
    return sum

def compute_support(supports_list,weight):
    support=list()
    for i in range(len(supports_list[0])):
        temp=list()
        for j in range(len(supports_list)):
            temp.append(supports_list[j][i])
        support.append(weighted_sum(temp, weight))
        del temp
    return support

def support_filter(support, mass):
    real_support=list()
    for i in support:
        real_support.append(list())
    real_mass=list()
    for i in range(len(mass)):
        if mass[i]!=0:
            real_mass.append(mass[i])
            for j in range(len(support)):
                real_support[j].append(support[j][i])
    for i in real_support:
        i=np.array(i)
    return real_support, real_mass

def convert_support (support):
    real_support=list((list(),list()))
    for i in range(len(support)):
        for j in range(len(support[i])):
            if support[i][j]!=0:
                real_support[0].append(i)
                real_support[1].append(j)
    real_support[0]=np.array(real_support[0])
    real_support[1]=np.array(real_support[1])
    return real_support

def convert_support_and_mass (support):
    real_support=list((list(),list()))
    mass=list()
    for i,x in enumerate(support):
        for j,y in enumerate(x):
            if y!=0:
                real_support[0].append(i)
                real_support[1].append(j)
                mass.append(y)
    real_support[0]=np.array(real_support[0])
    real_support[1]=np.array(real_support[1])
    mass=np.array(mass)
    return real_support,mass
def convert_mass (support):
    mass=list()
    for i,x in enumerate(support):
        for j,y in enumerate(x):
            if y!=0:
                
                mass.append(y)
    mass=np.array(mass)
    return mass

def cost_tensor_3D(supports_list,weight):
    temp=list()
    for i in supports_list:
        temp.append(len(i[0]))
    T=np.zeros(temp)
    del temp
    for n in range(2):
        for i,a in enumerate(supports_list[0][n]):
            for j,b in enumerate(supports_list[1][n]):
                for k,c in enumerate(supports_list[2][n]):
                    bary_center=(a+b+c)/3
                    T[i][j][k]+=((a-bary_center)**2+(b-bary_center)**2+(c-bary_center)**2)/6
    return T


# @cuda.jit
# def cost_tensor_kernel(T,p,q,s):
#
#     x,y,z=cuda.grid(3)
#     for i in range(x,len(p),100):
#         for j in range(y,len(q),100):
#             for k in range(z,len(s),100):
#                 T[i][j][k]+=((p[i]-(p[i]+q[j]+s[k])/3)**2+(q[j]-(p[i]+q[j]+s[k])/3)**2+(s[k]-(p[i]+q[j]+s[k])/3)**2)/6

# def cost_tensor(p,q,s):
#     c=np.zeros((len(p[0]),len(q[0]),len(s[0])))
#     c_d=cuda.to_device(c)
#     cost_tensor_kernel[(25,25,25),(4,4,4)](c_d,p[0],q[0],s[0])
#     cost_tensor_kernel[(25,25,25),(4,4,4)](c_d,p[1],q[1],s[1])
#     c=c_d.copy_to_host()
#     del c_d
#     return c


# def cost_tensor_2D(supports_list,weight):
#     temp=list()
#     for i in supports_list:
#         temp.append(len(i[0]))
#     T=np.zeros(temp)
#     del temp
#     for n in range(2):
#         for i,a in enumerate(supports_list[0][n]):
#             for j,b in enumerate(supports_list[1][n]):
#                 bary_center=(a+b)/2
#                 T[i][j]+=((a-bary_center)**2+(b-bary_center)**2)/4
#     return T

def cost_tensor_kernel_jax(distribution):
    # Create meshgrids for all combinations of p, q, s
    distribution = jnp.meshgrid(*distribution, indexing='ij')
    
    # Compute mean for each combination
    m = len(distribution)
    mean = jnp.zeros(distribution[0].shape)
    for k in range(m):
        mean += distribution[k]
    mean /= m
    
    # Compute variance components
    vars = [(distribution[k] - mean) ** 2 for k in range(m)]    

    # Return the cost tensor
    return sum(vars) / (m * 2)

def cost_tensor(distribution):
    # Initialize result tensor
    dims = [len(dis[0]) for dis in distribution]
    c = jnp.zeros(dims)
    
    c += cost_tensor_kernel_jax([dis[0] for dis in distribution])
    c += cost_tensor_kernel_jax([dis[1] for dis in distribution])
    
    return c

def convert(image):
    x=[]
    y=[]
    mass=[]
    for i,a in enumerate(image):
        for j,b in enumerate(a):
            x+=[i]
            y+=[j]
            mass+=[b]
    x=np.array(x)
    y=np.array(y)
    mass=np.array(mass)
    return (x,y),mass

def prepare_input(m=3, k=0,size=60,N=10,seed=20):
    sample_number = m*N
    width = size
    ellipses = make_nested_ellipses(width, sample_number, seed=seed)
    
    supports_list=list()
    for i in range(m*k,m*k+m):
        supports_list.append(convert_support(ellipses[i]))
    weight=list(np.ones(m)/m)
    support=compute_support(supports_list,weight)
    mass=list()
    for i in range(m*k,m*k+m):
        mass.append(convert_mass(ellipses[i])/convert_mass(ellipses[i]).sum())
    C=cost_tensor(supports_list)

    return C, mass

def prepare_mnist(k=0,N=10):
    (X,Y), (Xt,Yt) = mnist.load_data()
    img=list()
    for i,j in enumerate(X):
        if Y[i]==0:
            img.append(X[i])

    n=3
    supports_list=list()
    for i in range(3*k,3*k+3):
        supports_list.append(convert_support(img[i]))
    weight=list(np.ones(n)/n)
    support=compute_support(supports_list,weight)
    mass=list()
    for i in range(3*k,3*k+3):
        mass.append(convert_mass(img[i])/convert_mass(img[i]).sum())
    p,q,s=mass
    p_s,q_s,s_s=supports_list
    C=cost_tensor(p_s,q_s,s_s)
    return C,p,q,s
