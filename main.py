import os
import time
import math
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from statistics import mean
from numpy import *
from dvp_linear_inv_cassi import (gap_denoise, admm_denoise,
                            GAP_TV_rec, ADMM_TV_rec)
from utils import (A, At, psnr,shift,shift_back)
from bm3d import bm3d_deblurring, BM3DProfile, gaussian_kernel


datasetdir = './Dataset' # dataset
resultsdir = './results' # results
datname = 'kaist_crop256_01' # name of the dataset
matfile = datasetdir + '/' + datname + '.mat' # path of the .mat data file
method = 'GAP'          # 'ADMM'
## data operation
r, c, nC, step = 256, 256, 31, 1
random.seed(5)
mask=np.zeros((r, c + step * (nC - 1)))
mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, nC))
mask256=sio.loadmat(r'.mask/mask256.mat')['mask']
#mask512=sio.loadmat(r'.mask/mask512.mat')['mask']
#mask1024=sio.loadmat(r'.mask/mask1024.mat')['mask']
for i in range(nC):
     mask_3d[:, i:i+256, i]=mask256
truth = sio.loadmat(matfile)['img']

truth_shift = np.zeros((r, c + step * (nC - 1), nC))
for i in range(nC):
    truth_shift[:,i*step:i*step+256,i]=truth[:,:,i]
meas = np.sum(mask_3d*truth_shift,2)
plt.figure()
plt.imshow(meas,cmap='gray')
plt.savefig('./result_img/{}_meas.png'.format(datname))
Phi = mask_3d
Phi_sum = np.sum(mask_3d**2,2)
Phi_sum[Phi_sum==0]=1

if method == 'GAP':
    _lambda = 1 # regularization factor
    accelerate = True # enable accelerated version of GAP
    denoiser = 'hsicnn' # total variation (TV); deep denoiser(hsicnn)
    iter_max = 20 # maximum number of iterations
    tv_weight = 6 # TV denoising weight (larger for smoother but slower)
    tv_iter_max = 5 # TV denoising maximum number of iterations each
    begin_time = time.time()
    vgaptv,psnr_gaptv = gap_denoise(meas,Phi,A,At,_lambda, 
                        accelerate, denoiser, iter_max, 
                        tv_weight=tv_weight, 
                        tv_iter_max=tv_iter_max,
                        X_orig=truth,sigma=[130,130,130,130,130,130,130,130])#
    end_time = time.time()
    vrecon = shift_back(vgaptv,step=1)
    tgaptv = end_time - begin_time
    print('GAP-{} PSNR {:2.2f} dB, running time {:.1f} seconds.'.format(
        denoiser.upper(), psnr_gaptv[-1], tgaptv))
elif method == 'ADMM':
    ## [2.1] ADMM [for baseline reference]
    _lambda = 1 # regularization factor
    gamma = 0.02 # enable accelerated version of GAP
    denoiser = 'tv' # total variation (TV)
    iter_max = 50 # maximum number of iterations
    tv_weight = 0.1 # TV denoising weight (larger for smoother but slower)
    tv_iter_max = 5 # TV denoising maximum number of iterations each
    begin_time = time.time()
    vadmmtv,psnr_admmtv,ssim_admmtv = admm_denoise(meas,Phi,A,At,_lambda,
                        0.01, denoiser, iter_max, 
                        tv_weight=tv_weight, 
                        tv_iter_max=tv_iter_max,
                        X_orig=ori_truth,sigma=[100,100,80,70,60,90])
    end_time = time.time()
    vrecon = shift_back(vadmmtv,step=2)
    tadmmtv = end_time - begin_time
    print('ADMM-{} PSNR {:2.2f} dB, running time {:.1f} seconds.'.format(
        denoiser.upper(), psnr_admmtv[-1], tadmmtv))
else:
    print('please input correct method.')
sio.savemat('./result_img/{}_result.mat'.format(datname),{'img':vrecon})
fig = plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(vrecon[:,:,(i+1)*3], cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig('./result_img/{}_result.png'.format(datname))
