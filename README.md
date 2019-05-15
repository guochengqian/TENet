# TENet 
### Trinity of Pixel Enhancement: a Joint Solution for Demosaicing, Denoising and Super-Resolution
By [Guocheng Qian](https://guochengqian.github.io/), [Jinjin Gu](http://www.jasongt.com/), [Jimmy S. Ren](http://www.jimmyren.com/), [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=en), Furong Zhao, Juan lin


### Dependencies
- Python >= 3
- [PyTorch 0.4.1](https://pytorch.org/) (CUDA version >= 7.5 if installing with CUDA. [More details](https://pytorch.org/get-started/previous-versions/))
- Python packages:  `pip install opencv-python scipy scikit-image`

### Qucik Test
1. Clone this github repo.

    ```
    git clone https://github.com/guochengqian/TENet
    cd TENet
    ```

2. Place your own **input images** in `$YourInputPath` folder.  You will save output in `$YourSavePath` folder.

3. Run test
   1. Model trained by simulation datasets 
        ```
        python -u test.py --pretrained_model checkpoints/pretrained_models/tri2-dn-df2kx6-6-6-64-2-rrdb_checkpoint_1490.0k.path\
            --model tenet2 --block_type rrdb --sr_n_resblocks 6 --dm_n_resblocks 6 --scale 2 --bias --channels 64 \
            --crop_scale 1 --denoise --sigma 10 --test_path $YourInputPath --save_path $YourSavePath  
        ```   
   2. Model trained by PixelShift200 datasets
        ```
        python -u test.py --pretrained_model checkpoints/pretrained_models/tri2-dn-matx6-6-6-64-2-rrdb_checkpoint_2500.0k.path\
            --model tenet2 --block_type rrdb --sr_n_resblocks 6 --dm_n_resblocks 6 --scale 2 --bias --channels 64 \
            --crop_scale 4 --denoise --sigma 5 --test_path $YourInputPath --save_path $YourSavePath  
        ```
   if Run out of CUDA memory, just set --crop_scale 2 (or higher)           
4. Run ablation study code:

    ```
    python ablation_study.py
    ```
5. Evaluate PSNR
    ```
    python cmp_psnr.py
    ```
    
### Train Network
1. Train code
    
    ```
    sh scripy\run_tri2-deno.sh
    ```
    
### Pretrained models location
    /mnt/lustre/DATAshare2/qianguocheng/senseSR/TENet_datasets/pretrained_models