# TENet [[PDF]](https://arxiv.org/abs/1905.02538)
### Trinity of Pixel Enhancement: a Joint Solution for Demosaicing, Denoising and Super-Resolution
By [Guocheng Qian](https://guochengqian.github.io/), [Jinjin Gu](http://www.jasongt.com/), [Jimmy S. Ren](http://www.jimmyren.com/), [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=en), Furong Zhao, Juan Lin

### Citation 
Please cite the following paper if you feel TENet is useful to your research


## Qucik Test
### Dependencies
- Python >= 3
- [PyTorch 0.4.1](https://pytorch.org/) (CUDA version >= 7.5 if installing with CUDA. [More details](https://pytorch.org/get-started/previous-versions/))
- [Tensorflow](https://www.tensorflow.org/install)  (cpu version is enough, only used for visualization in training)
- Python packages:  `pip install opencv-python scipy scikit-image`

### Test Models
1. Clone this github repo.

    ```
    git clone https://github.com/guochengqian/TENet
    cd TENet
    ```
2. Place your own **input images** in `$YourInputPath` folder.  You will save output in `$YourSavePath` folder.

3. Run test.
   1. test model trained by simulation datasets 
        ```
        sh ./scrip/test_tennet2-dn.sh  
        ```  
 
   2. test model trained by PixelShift200 datasets
        ```
        sh ./scrip/test_tennet2-dn-pixelshift200.sh  
        ```  
        Don't forget to change $YourInputPath and $YourSavePath

 

## TENet
<p align="center">
  <img height="600" src="figures/Network.png">
</p>


### Pixel Shift Technology
We employ advanced pixel shift technology to perform a full color sampling of the image.
Pixel shift technology takes four samples of the same image, and physically controls the camera sensor to move one pixel horizontally or vertically at each sampling to capture all color information at each pixel.
The pixel shift technology ensures that the sampled images follow the distribution of natural images sampled by the camera, and the full information of the color is completely obtained.
In this way, the collected images are artifacts-free, which leads to better training results for demosaicing related tasks.

<p align="center">
  <img height="600" src="figures/PixelShift.png">
</p>

### Results on simulation datasets


### Results on Real Images
<p align="center">
  <img height="600" src="figures/Surf.png">
</p>

### Train Network
1. Train code
    
    ```
    sh scripy\run_tenet2-deno.sh
    ```

## Ablation Study 


### Pretrained models location
    /mnt/lustre/DATAshare2/qianguocheng/senseSR/TENet_datasets/pretrained_models