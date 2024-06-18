import os
import torch
import numpy as np
from scipy.io import loadmat
import copy
import json
from tqdm import tqdm 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cpu")
def kernel_function(diffs,temp,eps):
    if temp==0:
        return 1 / (torch.sqrt(1 + (torch.norm(diffs, dim=2) * eps) ** 2))
    else:
        return torch.exp(-eps * ((torch.norm(diffs, dim=2)) ** 2))

def calculate_massmatrix(X, tmp=0,eps=0.1):
    X = torch.tensor(X, device=device, dtype=torch.float32)
    m, n = X.shape
    print(X.shape,eps)
    X_flatten = X.reshape(-1, 1)
    idx_flatten = torch.stack(torch.meshgrid(
        torch.linspace(0, m - 1, m, dtype=torch.float32, device=device) / (m),
        torch.linspace(0, n - 1, n, dtype=torch.float32, device=device) / (n),
        indexing="ij"
    ), dim=-1).reshape(-1, 2)
    diffs = idx_flatten.unsqueeze(0) - idx_flatten.unsqueeze(1)
    mass_matrix = kernel_function(diffs, tmp,eps)
    print("Mass Matrix prepared Well")
    condition_number = torch.linalg.cond(mass_matrix)
    print(condition_number)
    if(condition_number > 10 or condition_number==0):
        print("ill matrix!")
        exit()
    determinant = torch.det(mass_matrix)
    print(determinant)
    if determinant.abs().item() < 1e-10:
        print("Using pseudo-inverse due to near-singular matrix.")
        res = torch.linalg.pinv(mass_matrix).to(device)
    else:
        res = torch.linalg.inv(mass_matrix).to(device)

    print("Calculated Finish!")
    return res

# IMQ
eps_list = [47000,707000,2007000]
sample_rate_list = [10,4,2]
for sr in tqdm(range(0,2)):
    sample_rate = sample_rate_list[sr]
    data = loadmat('IQ_data/PALA_InVivoRatBrain_{}.mat'.format(str(1).zfill(3)))['IQ']
    downsampled_data = copy.deepcopy(data)
    eps = eps_list[sr]
    downsampled_data = downsampled_data[...,::sample_rate]
    mass_matrix = calculate_massmatrix(downsampled_data[0],0,eps)
    np.save("mass_matrix_{}_quar.npy".format(sample_rate), mass_matrix.cpu().numpy())

# GA
eps_list = [930000,295000,105000]
sample_rate_list = [2,4,10]
for sr in tqdm(range(0,2)):
    sample_rate = sample_rate_list[sr]
    data = loadmat('IQ_data/PALA_InVivoRatBrain_{}.mat'.format(str(1).zfill(3)))['IQ']
    downsampled_data = copy.deepcopy(data)
    eps = eps_list[sr]
    downsampled_data = downsampled_data[...,::sample_rate]
    mass_matrix = calculate_massmatrix(downsampled_data[0],1,eps)
    np.save("mass_matrix_{}_gauss.npy".format(sample_rate), mass_matrix.cpu().numpy())

