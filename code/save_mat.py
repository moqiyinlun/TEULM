import numpy as np
from scipy.io import loadmat, savemat
import cv2
from tqdm import tqdm 
import os
# idx = 1
# data = loadmat('IQ_data/PALA_InVivoRatBrain_{}.mat'.format(str(idx).zfill(3)))
# IQ_data = data['IQ']
# for i in [2,4,10]:
#     interpolate_data = np.load("my_result_quar_{}.npy".format(i))
#     print(np.mean(np.abs(interpolate_data-IQ_data)))
#     if IQ_data.shape == interpolate_data.shape:
#         data['IQ'] = interpolate_data
#         savemat('PALA_InVivoRatBrain_{}_interpolated.mat'.format(str(i).zfill(3)), data)
#         print("Data saved successfully.")
#     else:
#         print("The shape of interpolate_data does not match the shape of the original IQ data.")
for i in [2,4,10]:
    data_path = "IQ_{}_quar".format(i)
    new_path = "IQ_interpolation_data_{}_quar".format(i)
    os.makedirs(new_path,exist_ok=True)
    for idx in tqdm(range(1,241)):
        raw_data = loadmat('IQ_data/PALA_InVivoRatBrain_{}.mat'.format(str(idx).zfill(3)))
        IQ  = np.load(os.path.join(data_path,"my_result_{}.npy".format(idx)))
        raw_data["IQ"] = IQ 
        savemat(os.path.join(new_path,"PALA_InVivoRatBrain_{}.mat".format(str(idx).zfill(3))),raw_data)
for i in [2,4,10]:
    data_path = "IQ_{}_gauss".format(i)
    new_path = "IQ_interpolation_data_{}_gauss".format(i)
    os.makedirs(new_path,exist_ok=True)
    for idx in tqdm(range(1,241)):
        raw_data = loadmat('IQ_data/PALA_InVivoRatBrain_{}.mat'.format(str(idx).zfill(3)))
        IQ  = np.load(os.path.join(data_path,"my_result_{}.npy".format(idx)))
        raw_data["IQ"] = IQ 
        savemat(os.path.join(new_path,"PALA_InVivoRatBrain_{}.mat".format(str(idx).zfill(3))),raw_data)
