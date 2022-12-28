# PnP-CASSI
Python(pytorch) code for the paper: **Siming Zheng, Yang Liu, Ziyi Meng, Mu Qiao, Zhishen Tong, Xiaoyu Yang, Shensheng Han, and Xin Yuan, "Deep plug-and-play priors for spectral snapshot compressive imaging," Photon. Res. 9, B18-B29 (2021)**[[pdf]](https://opg.optica.org/prj/viewmedia.cfm?uri=prj-9-2-B18&seq=0) [[doi]](https://doi.org/10.1364/PRJ.411745)

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

## Test result

<p align="center">
<img src="https://github.com/zsm1211/PnP-CASSI/blob/main/img/table.png?height="300" width="1100" raw=true">
</p>

<p align="center">
<img src="https://github.com/zsm1211/PnP-CASSI/blob/main/img/Sim_data_RGB-1.png?height="300" width="1100" raw=true">
</p>

Figure 3.Test spectral data from ICVL (a) and KAIST (b) datasets used in simulation. The reference RGB images with pixel resolution **256×256** are shown here. We crop similar regions of the whole image for spatial sizes of 512×512 and 1024×1024. 
## Citation
```
@article{Zheng:21,
author = {Siming Zheng and Yang Liu and Ziyi Meng and Mu Qiao and Zhishen Tong and Xiaoyu Yang and Shensheng Han and Xin Yuan},
journal = {Photon. Res.},
keywords = {Compressive imaging; Hyperspectral imaging; Multispectral imaging; Reconstruction algorithms; Remote sensing; Spatial light modulators},
number = {2},
pages = {B18--B29},
publisher = {OSA},
title = {Deep plug-and-play priors for spectral snapshot compressive imaging},
volume = {9},
month = {Feb},
year = {2021},
url = {http://www.osapublishing.org/prj/abstract.cfm?URI=prj-9-2-B18},
doi = {10.1364/PRJ.411745},
abstract = {We propose a plug-and-play (PnP) method that uses deep-learning-based denoisers as regularization priors for spectral snapshot compressive imaging (SCI). Our method is efficient in terms of reconstruction quality and speed trade-off, and flexible enough to be ready to use for different compressive coding mechanisms. We demonstrate the efficiency and flexibility in both simulations and five different spectral SCI systems and show that the proposed deep PnP prior could achieve state-of-the-art results with a simple plug-in based on the optimization framework. This paves the way for capturing and recovering multi- or hyperspectral information in one snapshot, which might inspire intriguing applications in remote sensing, biomedical science, and material science. Our code is available at: https://github.com/zsm1211/PnP-CASSI.},
}
```
