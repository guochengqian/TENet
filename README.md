# TENet <a href="https://arxiv.org/abs/1905.02538" target="_blank">[PDF]</a> 
<!--<a href="https://drive.google.com/open?id=1FPELQupnGR750EoUWTY_0owkEnlAGVYH">[Checkpoints]</a>-->
<!--<a href="https://arxiv.org/abs/1905.02538" target="_blank">[datasets]</a> -->
### Trinity of Pixel Enhancement: a Joint Solution for Demosaicing, Denoising and Super-Resolution
By [Guocheng Qian](https://guochengqian.github.io/), [Jinjin Gu](http://www.jasongt.com/), [Jimmy S. Ren](http://www.jimmyren.com/), [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=en), Furong Zhao, Juan Lin

### Citation 
Please cite the following paper if you feel TENet is useful to your research
```
@article{qian2019trinity,
  title={Trinity of Pixel Enhancement: a Joint Solution for Demosaicking, Denoising and Super-Resolution},
  author={Qian, Guocheng and Gu, Jinjin and Ren, Jimmy S and Dong, Chao and Zhao, Furong and Lin, Juan},
  journal={arXiv preprint arXiv:1905.02538},
  year={2019}
}
```


# Resources Collection
### Pretrained models
[GoogleDrive](https://drive.google.com/open?id=1FPELQupnGR750EoUWTY_0owkEnlAGVYH) 

### Test data
[GoogleDrive](https://drive.google.com/open?id=1PtpOo7U-J-IuttZHeduE5ZyHlMW-7s1R)

### PixelShift200 dataset 
[Pixelshift200 website](http://guochengqian.com/pixelshift200)
   

## Quick Test
### Dependencies
- Python >= 3
- [PyTorch 0.4.1](https://pytorch.org/) (CUDA version >= 7.5 if installing with CUDA. [More details](https://pytorch.org/get-started/previous-versions/))
- [Tensorflow](https://www.tensorflow.org/install)  (cpu version is enough, only used for visualization in training)
- Python packages:  `pip install opencv-python scipy scikit-image`

```
conda create --name pytorch04
conda activate pytorch04
conda install pytorch=0.4.1 cuda90 torchvision tensorflow -c pytorch   
pip install opencv-python scipy scikit-image  
```

### Test Models
1. Clone this github repo.

    ```
    git clone https://github.com/guochengqian/TENet
    cd TENet
    ```
2. Place your own **input images** in `$YourInputPath` folder.  You will save output in `$YourSavePath` folder.
   Input images should be Bayer Raw images (**[bayer pattern](https://en.wikipedia.org/wiki/Bayer_filter) is rggb**). 
   
3. Run test.
   1. test model trained by synthesis datasets 
        ```
        sh ./script/test_tennet2-dn-df2k.sh  
        ```  
 
   2. test model trained by PixelShift200 datasets
        ```
        sh ./script/test_tenet2-dn-ps200.sh  
        ```  
        Don't forget to change $YourInputPath and $YourSavePath in .sh file.


## How to Train
We train our model both on synthesis datasets([DF2k](https://github.com/xinntao/BasicSR)) and our proposed full color sampled real word
4k dataset PixelShift200 (to be released soon).

1. Data preparation
    1. Synthesis data preparation
        1. Download ([DF2k](https://github.com/xinntao/BasicSR)) dataset, which is a combination dataset of DIV2K and Flickr2K
        2. Crop images into resolution 256*256, using followed code:
            ```
            python ./dataset/crop_images.py
            ```
        3. generate txt file used for training
            ```
            python ./dataset/generate_train_df2k.py
            ```       
        
    2. PixelShift200 data preparation 
        1. Download [Pixelshift200](http://guochengqian.com/pixelshift200). They are .mat format, having 4 channels (R, Gr, Gb, B).
        2. Crop images into 512*512, using followed code:
            ```
            python ./dataset/crop_mats.py
            ```
        3. generate txt file used for training
            ```
            python ./dataset/generate_train_mat.py
            ```
                         
2. Train model on synthesis dataset
    
    ```
    sh script/run_tenet2-dn-df2k.sh
    ```

3. Train model on PixelShift200 dataset
    ```
    sh script/run_tenet2-dn-ps200.sh    
    ```


## TENet

Our approach can be divided into two parts, the first part is a mapping of joint denoising and SR, 
and the second part converts the SR mosaic image into a full color image.
The two parts can be trained and performed jointly.
The network structure is illustrated as follows.

<p align="center">
  <img height="300" src="figures/Network.png">
</p>



## PixelShift200 dataset
We employ advanced pixel shift technology to perform a full color sampling of the image.
Pixel shift technology takes four samples of the same image, and physically controls the camera sensor to move one pixel horizontally or vertically at each sampling to capture all color information at each pixel.
The pixel shift technology ensures that the sampled images follow the distribution of natural images sampled by the camera, and the full information of the color is completely obtained.
In this way, the collected images are artifacts-free, which leads to better training results for demosaicing related tasks.

<p align="center">
  <img height="200" src="figures/PixelShift.png">
</p>


## Result
<!--### Results on simulated datasets-->


### Results on Real Images
<p align="center">
  <img width="800" src="figures/Surf.png">
</p>


