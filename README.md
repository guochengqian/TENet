# TENet 
### Trinity of Pixel Enhancement: a Joint Solution for Demosaicing, Denoising and Super-Resolution
By Guocheng Qian, [Jinjin Gu](http://www.jasongt.com/), [Jimmy S. Ren](http://www.jimmyren.com/), [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=en), Furong Zhao, Juan lin


### Dependencies
- Python 3.6
- [PyTorch 0.4.1](https://pytorch.org/) (CUDA version >= 7.5 if installing with CUDA. [More details](https://pytorch.org/get-started/previous-versions/))
- Python packages:  `pip install opencv-python libtiff`

### Qucik Test
1. Clone this github repo.

    ```
    git clone https://github.com/guochengqian/TENet
    cd TENet
    ```

2. Place your own **input images** in `$YourInputPath` folder.  You will save output in `$YourSavePath` folder.

3. Run test. We provide TENet2, TENet1, Demo, RawDeno, RGBDeno, RawSR, RGBSR models.    
    ```
    Demo: 
    python -u test.py --pretrained_model checkpoints/pretrained_models/demo-df2kx6-6-3-64-2-rrdb_checkpoint_1096.0k.path \
               --model demo --block_type rrdb --n_resblock 6 --channels 64 \
               --datatype uint8 --crop_scale 1 --img_type raw --test_path $YourInputPath --save_path $YourSavePath
    
    Joint Demo-deno: 
    python -u test.py --pretrained_model checkpoints/pretrained_models/demo-df2kx6-6-3-64-2-rrdb_checkpoint_1096.0k.path \
               --model demo --block_type rrdb --n_resblock 6 --channels 64 \
               --datatype uint8 --crop_scale 1 --img_type raw --denoise --sigma 10 --test_path $YourInputPath --save_path $YourSavePath
    
               
    TENet denoise with 2 label:
    python -u test.py --pretrained_model checkpoints/pretrained_models/tri2-dn-df2kx6-6-6-64-2-rrdb_checkpoint_1490.0k.path\
        --model tri2 --block_type rrdb --sr_n_resblocks 6 --dm_n_resblocks 6 --scale 2 --bias --channels 64 \
        --datatype uint8 --crop_scale 1 --img_type raw  --denoise --sigma 10 --test_path $YourInputPath --save_path $YourSavePath              
    ```
    <!--
    just change --model parameter to run all other models, cis shown above
    
        TENet2: tri2
        
        TENet1: tri1
        
        Demo: demo
        
        RawDeno: denoraw
        
        RGBDeno: denorgb
         
        RawSR: srraw
        
        RGBSR: srrgb
    -->
    
4. Run ablation study code:

    ```
    python ablation_study.py
    ```
    
### Train Network
1. Train code
    
    ```
        sh scripy\run_tri2-deno.sh
    ```
    
### Datasets location

    /mnt/lustre/DATAshare2/qianguocheng/senseSR/TENet_datasets
    
### Pretrained models location
    /mnt/lustre/DATAshare2/qianguocheng/senseSR/TENet_datasets/pretrained_models