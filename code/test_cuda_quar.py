import os
import torch
import numpy as np
from scipy.io import loadmat
import copy
import json
from tqdm import tqdm 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")

def kernel_function(diffs, eps):
    return torch.exp(-eps * ((torch.norm(diffs, dim=2)) ** 2))
    # return torch.sqrt(1 + (eps * torch.norm(diffs, dim=2)) ** 2)
def calculate_massmatrix(X, eps=0.1):
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
    mass_matrix = kernel_function(diffs, eps)
    print("Mass Matrix prepared Well")
    print(torch.linalg.cond(mass_matrix))
    determinant = torch.det(mass_matrix)
    print(determinant)
    if determinant.abs().item() < 1e-10:
        print("Using pseudo-inverse due to near-singular matrix.")
        res = torch.linalg.pinv(mass_matrix).to(device)
    else:
        res = torch.linalg.inv(mass_matrix).to(device)

    print("Calculated Finish!")
    return res
def solve_parameter(X, mass_matrix):
    X = torch.tensor(X, device=device, dtype=torch.float32)
    m, n = X.shape
    # print(m,n)
    # print(X[0][0])
    X_flatten = X.reshape(-1, 1)
    res = mass_matrix @ X_flatten
    
    return res

def rbf_interpolation(weights,kernel_values):
    values = torch.mm(kernel_values, weights)
    return values
eps_list = [47000,707000,2007000]
sample_rate_list = [10,4,2]
for sr in range(0,len(sample_rate_list)):
    sample_rate = sample_rate_list[sr]
    data = loadmat('IQ_data/PALA_InVivoRatBrain_{}.mat'.format(str(1).zfill(3)))['IQ']
    downsampled_data = copy.deepcopy(data)
    eps = eps_list[sr]
    downsampled_data = downsampled_data[...,::sample_rate]
    mass_matrix = np.load("mass_matrix_{}_quar.npy".format(sample_rate))
    _,t,q = data.shape
    print(t,q)
    new_points = torch.stack(torch.meshgrid(
        torch.linspace(0, t-1, t, dtype=torch.float32, device="cpu")/(t),
        torch.linspace(0, q-1, q, dtype=torch.float32, device="cpu")/(q),
        indexing="ij"
    ), dim=-1).reshape(-1, 2)
    control_points = torch.stack(torch.meshgrid(
        torch.linspace(0, downsampled_data[0, ...].shape[0] - 1, downsampled_data[0, ...].shape[0], dtype=torch.float32, device="cpu") / (downsampled_data[0, ...].shape[0]),
        torch.linspace(0, downsampled_data[0, ...].shape[1] - 1, downsampled_data[0, ...].shape[1], dtype=torch.float32, device="cpu") / (downsampled_data[0, ...].shape[1]),
        indexing="ij"
    ), dim=-1).reshape(-1, 2)
    diffs = new_points.unsqueeze(1) - control_points.unsqueeze(0)
    kernel_values = kernel_function(diffs, eps).to("cuda")
    mass_matrix = torch.tensor(mass_matrix, device=device, dtype=torch.float32)
    os.makedirs("IQ_{}_quar".format(sample_rate),exist_ok=True)
    for idx in tqdm(range(1,241)):
        
        data = loadmat('IQ_data/PALA_InVivoRatBrain_{}.mat'.format(str(idx).zfill(3)))['IQ']
        width = 0
        downsampled_data = copy.deepcopy(data)
        downsampled_data = downsampled_data[...,::sample_rate]
        interpolated_data = np.empty_like(data)
        # mass_matrix = calculate_massmatrix(downsampled_data[0],eps)
        # np.save("mass_matrix_{}.npy".format(sample_rate), mass_matrix.cpu().numpy())
        # exit()
        loss_dic = {"real_loss":[],"image_loss":[]}
        for width in range(downsampled_data.shape[0]):
            A = downsampled_data[width, ...]
            raw_data = data[width,...]

            raw_real = np.real(raw_data)
            raw_imag = np.imag(raw_data)

            A_real = np.real(A)
            A_imag = np.imag(A)

            A_real = torch.tensor(A_real, device=device, dtype=torch.float32)
            A_imag = torch.tensor(A_imag, device=device, dtype=torch.float32)

            raw_real = torch.tensor(raw_real, device=device, dtype=torch.float32)
            raw_imag = torch.tensor(raw_imag, device=device, dtype=torch.float32)

            weights_real = solve_parameter(A_real.cpu().numpy(),mass_matrix)
            weights_imag = solve_parameter(A_imag.cpu().numpy(),mass_matrix)

            reconstructed_A_real = rbf_interpolation(weights_real.to("cuda"), kernel_values)
            reconstructed_A_real = reconstructed_A_real.view(118, 800).to(device)

            reconstructed_A_imag = rbf_interpolation(weights_imag.to("cuda"), kernel_values)
            reconstructed_A_imag = reconstructed_A_imag.view(118, 800).to(device)

            reconstructed_A = reconstructed_A_real + 1j * reconstructed_A_imag

            loss_real = torch.mean((raw_real - reconstructed_A_real) ** 2)
            loss_imag = torch.mean((raw_imag - reconstructed_A_imag) ** 2)
            interpolated_data[width,...] = reconstructed_A.cpu().numpy()

            # print({'Real loss': loss_real.item(), 'Imag loss': loss_imag.item()})
            loss_dic["real_loss"].append(loss_real.item())
            loss_dic["image_loss"].append(loss_imag.item())
        with open("IQ_{}_quar/loss_dic_{}.json".format(sample_rate,idx),"w") as f:
            json.dump(loss_dic,f,indent=4)
        np.save("IQ_{}_quar/my_result_{}.npy".format(sample_rate,idx),interpolated_data)
