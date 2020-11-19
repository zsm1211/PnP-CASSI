# PnP-CASSI
Python(pytorch) code for the paper **Deep Plug-and-Play Priors for Spectral Snapshot Compressive Imaging**

<p align="left">
<img src="https://github.com/zsm1211/PnP-CASSI/blob/main/img/Simu_colorchecker_256-1.png?height="600" width="466"raw=true">
</p>

Figure 1.Simulation results of **color-checker** with size of **256×256** from KAIST dataset compared with the ground truth. PSNR and SSIM results are also shown for each algorithm.

## Abstract
We propose a plug-and-play (PnP) method, which uses deep-learning-based denoisersas regularization priors for spectral snapshot compressive imaging (SCI). Our method is efficient in terms of reconstruction quality and speed trade-off, and flexible to be ready-to-use for differentcompressive coding mechanisms.

<p align="center">
<img src="https://github.com/zsm1211/PnP-CASSI/blob/main/img/fig01_flow_chart-1.png?height="300" width="500" raw=true">
</p>

Figure 2.Image formation process of a typical spectral SCI system,i.e., SD-CASSI and the reconstruction process using the proposed deep plug-and-play (PnP) prior algorithm.

## Usage
1. Download this repository via git or download the [zip file](https://codeload.github.com/zsm1211/PnP-CASSI/zip/main) manually.
```
git clone https://github.com/zsm1211/PnP-CASSI
```
2. Run the file **main.py** to test the data.

3. We provide masks at 3 different resolutions, you can generate your own mask for HSI at different resolutions. 

## Test spectral data in the paper

<p align="center">
<img src="https://github.com/zsm1211/PnP-CASSI/blob/main/img/Sim_data_RGB-1.png?height="300" width="1100" raw=true">
</p>
