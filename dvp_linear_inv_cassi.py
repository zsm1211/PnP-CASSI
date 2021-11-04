import time
import math
import numpy as np
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage.measure import (compare_psnr, compare_ssim)
from utils import (A, At, psnr, shift, shift_back,calculate_ssim,TV_denoiser)
# import skimage.metrics.peak_signal_noise_ratio as psnr
# import skimage.metrics.structural_similarity as ssim
from hsi import HSI_SDeCNN as net
import torch
from bm3d import bm3d_deblurring, BM3DProfile, gaussian_kernel
import scipy.io as sio


def gap_denoise(y, Phi, A, At, _lambda=1, accelerate=True, 
                denoiser='tv', iter_max=50, noise_estimate=True, sigma=None, 
                tv_weight=0.1, tv_iter_max=5, multichannel=True, x0=None, 
                X_orig=None, model=None, show_iqa=True):
    '''
    Alternating direction method of multipliers (ADMM)[1]-based denoising 
    regularization for snapshot compressive imaging (SCI).

    Parameters
    ----------
    y : two-dimensional (2D) ndarray of ints, uints or floats
        Input single measurement of the snapshot compressive imager (SCI).
    Phi : three-dimensional (3D) ndarray of ints, uints or floats, omitted
        Input sensing matrix of SCI with the third dimension as the 
        time-variant, spectral-variant, volume-variant, or angular-variant 
        masks, where each mask has the same pixel resolution as the snapshot
        measurement.
    Phi_sum : 2D ndarray,
        Sum of the sensing matrix `Phi` along the third dimension.
    A : function
        Forward model of SCI, where multiple encoded frames are collapsed into
        a single measurement.
    At : function
        Transpose of the forward model.
    proj_meth : {'admm' or 'gap'}, optional
        Projection method of the data term. Alternating direction method of 
        multipliers (ADMM)[1] and generalizedv alternating projection (GAP)[2]
        are used, where ADMM for noisy data, especially real data and GAP for 
        noise-free data.
    gamma : float, optional
        Parameter in the ADMM projection, where more noisy measurements require
        greater gamma.
    denoiser : string, optional
        Denoiser used as the regularization imposing on the prior term of the 
        reconstruction.
    _lambda : float, optional
        Regularization factor balancing the data term and the prior term, 
        where larger `_lambda` imposing more constrains on the prior term. 
    iter_max : int or uint, optional 
        Maximum number of iterations.
    accelerate : boolean, optional
        Enable acceleration in GAP.
    noise_estimate : boolean, optional
        Enable noise estimation in the denoiser.
    sigma : one-dimensional (1D) ndarray of ints, uints or floats
        Input noise standard deviation for the denoiser if and only if noise 
        estimation is disabled(i.e., noise_estimate==False). The scale of sigma 
        is [0, 255] regardless of the the scale of the input measurement and 
        masks.
    tv_weight : float, optional
        weight in total variation (TV) denoising.
    x0 : 3D ndarray 
        Start point (initialized value) for the iteration process of the 
        reconstruction.
    model : pretrained model for image/video denoising.

    Returns
    -------
    x : 3D ndarray
        Reconstructed 3D scene captured by the SCI system.

    References
    ----------
    .. [1] X. Liao, H. Li, and L. Carin, "Generalized Alternating Projection 
           for Weighted-$\ell_{2,1}$ Minimization with Applications to 
           Model-Based Compressive Sensing," SIAM Journal on Imaging Sciences, 
           vol. 7, no. 2, pp. 797-823, 2014.
    .. [2] X. Yuan, "Generalized alternating projection based total variation 
           minimization for compressive sensing," in IEEE International 
           Conference on Image Processing (ICIP), 2016, pp. 2539-2543.
    .. [3] Y. Liu, X. Yuan, J. Suo, D. Brady, and Q. Dai, "Rank Minimization 
           for Snapshot Compressive Imaging," IEEE Transactions on Pattern 
           Analysis and Machine Intelligence, doi:10.1109/TPAMI.2018.2873587, 
           2018.

    '''
    # [0] initialization
    if x0 is None:
        print(At)
        x0 = At(y, Phi) # default start point (initialized value)
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)
    y1 = np.zeros_like(y) 
    Phi_sum = np.sum(Phi,2)
    Phi_sum[Phi_sum==0]=1
    # [1] start iteration for reconstruction
    x = x0 # initialization
    psnr_all = []
    ssim_all=[]
    k = 0
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = net()
    model.load_state_dict(torch.load(r'./check_points/deep_denoiser.pth'))
    model.eval()
    for q, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    for idx, nsig in enumerate(sigma): # iterate all noise levels
        for it in range(iter_max[idx]):
            #print('max1_{0}_{1}:'.format(idx,it),np.max(x))
            yb = A(x,Phi)
            if accelerate: # accelerated version of GAP
                y1 = y1 + (y-yb)
                x = x + _lambda*(At((y1-yb)/Phi_sum,Phi)) # GAP_acc
            else:
                x = x + _lambda*(At((y-yb)/Phi_sum,Phi)) # GAP
            x = shift_back(x,step=1)
            # switch denoiser
            if denoiser.lower() == 'tv': # total variation (TV) denoising
                x = denoise_tv_chambolle(x, nsig / 255, n_iter_max=tv_iter_max, multichannel=multichannel)
                #x= TV_denoiser(x, tv_weight, n_iter_max=tv_iter_max)
            elif denoiser.lower() == 'hsicnn':
                l_ch=10
                m_ch=10
                h_ch=10
                if (k>123 and k<=125 ) or (k>=119 and k<=121) or (k>=115 and k<=117) or (k>=111 and k<=113) or (k>=107 and k<=109) or (k>=103 and k<=105) or (k>=99 and k<=101) or (k>=95 and k<=97) or  (k>=91 and k<=93) or (k>=87 and k<=89) or (k>=83 and k<=85):
                    tem = None
                    for i in range(31):
                        net_input = None
                        if i < 3:
                            ori_nsig = nsig

                            if i==0:
                                net_input = np.dstack((x[:, :, i], x[:, :, i], x[:, :, i], x[:, :, i:i + 4]))
                            elif i==1:
                                net_input = np.dstack((x[:, :, i-1], x[:, :, i-1], x[:, :, i-1], x[:, :, i:i + 4]))
                            elif i==2:
                                net_input = np.dstack((x[:, :, i-2], x[:, :, i-2], x[:, :, i-1], x[:, :, i:i + 4]))
                            net_input = torch.from_numpy(np.ascontiguousarray(net_input)).permute(2,0,1).float().unsqueeze(0)
                            net_input = net_input.to(device)
                            Nsigma = torch.full((1, 1, 1, 1), l_ch / 255.).type_as(net_input)
                            output = model(net_input, Nsigma)
                            output = output.data.squeeze().cpu().numpy()
                            if k<0:
                                output = denoise_tv_chambolle(x[:, :, i], nsig / 255, n_iter_max=tv_iter_max,multichannel=False)
                            nsig = ori_nsig
                            if i == 0:
                                tem = output
                            else:
                                tem = np.dstack((tem, output))
                        elif i > 27:
                            ori_nsig=nsig
                            if k>=45:
                                nsig/=1
                            if i==28:
                                net_input = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i+1], x[:, :, i+2], x[:, :, i+2]))
                            elif i==29:
                                net_input = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i+1], x[:, :, i+1], x[:, :, i+1]))
                            elif i==30:
                                net_input = np.dstack((x[:, :, i - 3:i + 1], x[:, :, i], x[:, :, i], x[:, :, i]))
                            net_input = torch.from_numpy(np.ascontiguousarray(net_input)).permute(2, 0,1).float().unsqueeze(0)
                            net_input = net_input.to(device)
                            Nsigma = torch.full((1, 1, 1, 1), m_ch / 255.).type_as(net_input)
                            output = model(net_input, Nsigma)
                            output = output.data.squeeze().cpu().numpy()
                            if k<0:
                                output = denoise_tv_chambolle(x[:, :, i], 10 / 255, n_iter_max=tv_iter_max,multichannel=False)
                            tem = np.dstack((tem, output))
                            nsig=ori_nsig
                        else:
                            ori_nsig = nsig
                            net_input = x[:, :, i - 3:i + 4]
                            net_input = torch.from_numpy(np.ascontiguousarray(net_input)).permute(2, 0,1).float().unsqueeze(0)
                            net_input = net_input.to(device)
                            Nsigma = torch.full((1, 1, 1, 1), h_ch / 255.).type_as(net_input)
                            output = model(net_input, Nsigma)
                            output = output.data.squeeze().cpu().numpy()
                            tem = np.dstack((tem, output))
                            nsig = ori_nsig
                    #x = np.clip(tem,0,1)
                    x=tem

                else:
                    x = denoise_tv_chambolle(x, nsig / 255, n_iter_max=tv_iter_max, multichannel=multichannel)
                    #x = TV_denoiser(x, tv_weight, n_iter_max=tv_iter_max)
            elif denoiser.lower() =='bm3d':
                sigma = nsig/255
                v = np.zeros((15, 15))
                for x1 in range(-7, 8, 1):
                    for x2 in range(-7, 8, 1):
                        v[x1 + 7, x2 + 7] = 1 / (x1 ** 2 + x2 ** 2 + 1)
                v = v / np.sum(v)
                for i in range(28):
                    x[:,:,i]= bm3d_deblurring(np.atleast_3d(x[:,:,i]), sigma, v)
            else:
                raise ValueError('Unsupported denoiser {}!'.format(denoiser))
            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
            if show_iqa and X_orig is not None:
                ssim_all.append(calculate_ssim(X_orig, x))
                psnr_all.append(psnr(X_orig, x))
                if (k+1)%1 == 0:
                    if not noise_estimate and nsig is not None:
                        if nsig < 1:
                            print('  GAP-{0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                              'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                               k+1, nsig*255, psnr_all[k]),
                              'SSIM:{}'.format(ssim_all[k]))
                        else:
                            print('  GAP-{0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                                k+1, nsig, psnr_all[k]),
                              'SSIM:{}'.format(ssim_all[k]))
                    else:
                        print('  GAP-{0} iteration {1: 3d}, ' 
                              'PSNR {2:2.2f} dB.'.format(denoiser.upper(), 
                               k+1, psnr_all[k]),
                              'SSIM:{}'.format(ssim_all[k]))
            x = shift(x,step=1)
            if k==123:
                break
            k = k+1

    return x, psnr_all

def admm_denoise(y, Phi, A, At, _lambda=1, gamma=0.01,
                denoiser='tv', iter_max=50, noise_estimate=True, sigma=None, 
                tv_weight=0.1, tv_iter_max=5, multichannel=True, x0=None, 
                X_orig=None, show_iqa=True):
    '''
    Alternating direction method of multipliers (ADMM)[1]-based denoising 
    regularization for snapshot compressive imaging (SCI).

    Parameters
    ----------
    y : two-dimensional (2D) ndarray of ints, uints or floats
        Input single measurement of the snapshot compressive imager (SCI).
    Phi : three-dimensional (3D) ndarray of ints, uints or floats, omitted
        Input sensing matrix of SCI with the third dimension as the 
        time-variant, spectral-variant, volume-variant, or angular-variant 
        masks, where each mask has the same pixel resolution as the snapshot
        measurement.
    Phi_sum : 2D ndarray
        Sum of the sensing matrix `Phi` along the third dimension.
    A : function
        Forward model of SCI, where multiple encoded frames are collapsed into
        a single measurement.
    At : function
        Transpose of the forward model.
    proj_meth : {'admm' or 'gap'}, optional
        Projection method of the data term. Alternating direction method of 
        multipliers (ADMM)[1] and generalizedv alternating projection (GAP)[2]
        are used, where ADMM for noisy data, especially real data and GAP for 
        noise-free data.
    gamma : float, optional
        Parameter in the ADMM projection, where more noisy measurements require
        greater gamma.
    denoiser : string, optional
        Denoiser used as the regularization imposing on the prior term of the 
        reconstruction.
    _lambda : float, optional
        Regularization factor balancing the data term and the prior term, 
        where larger `_lambda` imposing more constrains on the prior term. 
    iter_max : int or uint, optional 
        Maximum number of iterations.
    accelerate : boolean, optional
        Enable acceleration in GAP.
    noise_estimate : boolean, optional
        Enable noise estimation in the denoiser.
    sigma : one-dimensional (1D) ndarray of ints, uints or floats
        Input noise standard deviation for the denoiser if and only if noise 
        estimation is disabled(i.e., noise_estimate==False). The scale of sigma 
        is [0, 255] regardless of the the scale of the input measurement and 
        masks.
    tv_weight : float, optional
        weight in total variation (TV) denoising.
    x0 : 3D ndarray 
        Start point (initialized value) for the iteration process of the 
        reconstruction.

    Returns
    -------
    x : 3D ndarray
        Reconstructed 3D scene captured by the SCI system.

    References
    ----------
    .. [1] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, 
           "Distributed Optimization and Statistical Learning via the 
           Alternating Direction Method of Multipliers," Foundations and 
           Trends® in Machine Learning, vol. 3, no. 1, pp. 1-122, 2011.
    .. [2] X. Yuan, "Generalized alternating projection based total variation 
           minimization for compressive sensing," in IEEE International 
           Conference on Image Processing (ICIP), 2016, pp. 2539-2543.
    .. [3] Y. Liu, X. Yuan, J. Suo, D. Brady, and Q. Dai, "Rank Minimization 
           for Snapshot Compressive Imaging," IEEE Transactions on Pattern 
           Analysis and Machine Intelligence, doi:10.1109/TPAMI.2018.2873587, 
           2018.
    '''
    # [0] initialization
    if x0 is None:
        x0 = At(y,Phi) # default start point (initialized value)
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)
    # [1] start iteration for reconstruction
    x = x0 # initialization
    theta = x0
    Phi_sum = np.sum(Phi,2)
    Phi_sum[Phi_sum==0]=1
    b = np.zeros_like(x0)
    psnr_all = []
    ssim_all=[]
    k = 0
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = net()
    model.load_state_dict(
        torch.load(r'/home/dgl/zhengsiming/self_train/check_points/deep_denoiser.pth'))
    model.eval()
    for q, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    for idx, nsig in enumerate(sigma): # iterate all noise levels
        for it in range(iter_max[idx]):
            # Euclidean projection
            yb = A(theta+b,Phi)
            x = (theta+b) + _lambda*(At((y-yb)/(Phi_sum+gamma),Phi)) # ADMM
            x1 = shift_back(x-b,step=2)
            #x1=x-b
            # switch denoiser 
            if denoiser.lower() == 'tv': # total variation (TV) denoising
                #theta = denoise_tv_chambolle(x1, nsig/255, n_iter_max=tv_iter_max, multichannel=multichannel)
                theta = TV_denoiser(x1, tv_weight, n_iter_max=tv_iter_max)
            elif denoiser.lower() == 'hsicnn':
                if k>=89:
                    tem = None
                    for i in range(28):
                        net_input = None
                        if i < 3:
                            if i==0:
                                net_input = np.dstack((x1[:, :, i], x1[:, :, i], x1[:, :, i], x1[:, :, i:i + 4]))
                            elif i==1:
                                net_input = np.dstack((x1[:, :, i-1], x1[:, :, i-1], x1[:, :, i-1], x1[:, :, i:i + 4]))
                            elif i==2:
                                net_input = np.dstack((x1[:, :, i-2], x1[:, :, i-2], x1[:, :, i-1], x1[:, :, i:i + 4]))
                            net_input = torch.from_numpy(np.ascontiguousarray(net_input)).permute(2, 0,1).float().unsqueeze(0)
                            net_input = net_input.to(device)
                            Nsigma = torch.full((1, 1, 1, 1), 10 / 255.).type_as(net_input)
                            output = model(net_input, Nsigma)
                            output = output.data.squeeze().float().cpu().numpy()
                            if i == 0:
                                tem = output
                            else:
                                tem = np.dstack((tem, output))
                        elif i > 24:
                            if i == 25:
                                net_input = np.dstack((x1[:, :, i - 3:i + 1], x1[:, :, i + 1], x1[:, :, i + 2], x1[:, :, i + 2]))
                            elif i == 26:
                                net_input = np.dstack((x1[:, :, i - 3:i + 1], x1[:, :, i + 1], x1[:, :, i + 1], x1[:, :, i + 1]))
                            elif i == 27:
                                net_input = np.dstack((x1[:, :, i - 3:i + 1], x1[:, :, i], x1[:, :, i], x1[:, :, i]))
                            net_input = torch.from_numpy(np.ascontiguousarray(net_input)).permute(2, 0, 1).float().unsqueeze(0)
                            net_input = net_input.to(device)
                            Nsigma = torch.full((1, 1, 1, 1),10 / 255.).type_as(net_input)
                            output = model(net_input, Nsigma)
                            output = output.data.squeeze().float().cpu().numpy()
                            tem = np.dstack((tem, output))
                        else:
                            net_input = x1[:, :, i - 3:i + 4]
                            net_input = torch.from_numpy(np.ascontiguousarray(net_input)).permute(2, 0,1).float().unsqueeze(0)
                            net_input = net_input.to(device)
                            Nsigma = torch.full((1, 1, 1, 1), 10 / 255.).type_as(net_input)
                            output = model(net_input, Nsigma)
                            output = output.data.squeeze().float().cpu().numpy()
                            tem = np.dstack((tem, output))
                    theta = tem
                else:
                    #print('theta:', np.max(theta))
                    theta = denoise_tv_chambolle(x1, tv_weight, n_iter_max=tv_iter_max, multichannel=multichannel)
            else:
                raise ValueError('Unsupported denoiser {}!'.format(denoiser))
            
            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
            if show_iqa and X_orig is not None:
                psnr_all.append(psnr(X_orig, theta))
                ssim_all.append(calculate_ssim(X_orig,theta))
                if (k+1)%1 == 0:
                    if not noise_estimate and nsig is not None:
                        if nsig < 1:
                            print('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                              'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                               k+1, nsig*255, psnr_all[k]),
                              'SSIM:{}'.format(ssim_all[k]))
                        else:
                            print('  ADMM-{0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                                k+1, nsig, psnr_all[k]),
                                'SSIM:{}'.format(ssim_all[k]))
                    else:
                        print('  ADMM-{0} iteration {1: 3d}, ' 
                              'PSNR {2: 2.2f} dB.'.format(denoiser.upper(), 
                               k+1, psnr_all[k]),
                              'SSIM:{}'.format(ssim_all[k]))
            theta = shift(theta,step=2)
            b = b - (x-theta) # update residual
            k = k+1
    return theta, psnr_all,ssim_all

def GAP_TV_rec(y,Phi,A, At,Phi_sum, maxiter, step_size, weight, row, col, ColT, X_ori):
    y1 = np.zeros((row,col))
    begin_time = time.time()
    f = At(y,Phi)
    for ni in range(maxiter):
        fb = A(f,Phi)
        y1 = y1+ (y-fb)
        f  = f + np.multiply(step_size, At( np.divide(y1-fb,Phi_sum),Phi ))
        f = denoise_tv_chambolle(f, weight,n_iter_max=30,multichannel=True)
    
        if (ni+1)%5 == 0:
            # mse = np.mean(np.sum((y-A(f,Phi))**2,axis=(0,1)))
            end_time = time.time()
            print("GAP-TV: Iteration %3d, PSNR = %2.2f dB,"
              " time = %3.1fs."
              % (ni+1, psnr(f, X_ori), end_time-begin_time))
    return f

def ADMM_TV_rec(y,Phi,A, At,Phi_sum, maxiter, step_size, weight, row, col, ColT, eta,X_ori):
    #y1 = np.zeros((row,col))
    begin_time = time.time()
    theta = At(y,Phi)
    v =theta
    b = np.zeros((row,col,ColT))
    for ni in range(maxiter):
        yb = A(theta+b,Phi)
        #y1 = y1+ (y-fb)
        v  = (theta+b) + np.multiply(step_size, At( np.divide(y-yb,Phi_sum+eta),Phi ))
        #vmb = v-b
        theta = denoise_tv_chambolle(v-b, weight,n_iter_max=30,multichannel=True)
        
        b = b-(v-theta)
        weight = 0.999*weight
        eta = 0.998 * eta
        
        if (ni+1)%5 == 0:
            # mse = np.mean(np.sum((y-A(v,Phi))**2,axis=(0,1)))
            end_time = time.time()
            print("ADMM-TV: Iteration %3d, PSNR = %2.2f dB,"
              " time = %3.1fs."
              % (ni+1, psnr(v, X_ori), end_time-begin_time))
    return v
