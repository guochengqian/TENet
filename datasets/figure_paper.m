clear; clc; 
%% The script for cropping images for paper figure 
src_path = '/home/qiang/codefiles/low_level/ISP/ispnet/data/testdata/result/KAUST_2'; 
pretrain_dataset = 'pixelshift';
img_name = 'KAUST_2'; 
% pred_dir = fullfile(fullfile('/home/qiang/codefiles/low_level/ISP/ispnet/pretrain/', pretrain_dataset, 'pipeline'), ['result_', dataset]);

tasks = {'noisy_lr_raw-raw-linrgb-tenet-pixelshift', 'noisy_lr_raw-None-linrgb-tenet-pixelshift', 'noisy_lr_raw-raw-linrgb-tenet-div2k', 'noisy_lr_raw-raw-linrgb-jdsr-pixelshift', 'noisy_lr_raw-raw-linrgb-jdndmsr-pixelshift'}; 


% tasks = {'jdsr-dn+dm+sr-SR2', 'jdsr-dn+sr-dm-SR2', 'jdndmsr-dn+dm+sr-SR2', 'jdndmsr-dn+sr-dm-SR2', 'nlsa-dn+dm+sr-SR2', 'nlsa-dn+sr-dm-SR2', 'e2e-dn+dm+sr-SR2', 'e2e-dn+sr-dm-SR2'}; 

crop_size = [64-1, 64-1];
% crop_size = [1024-1, 1024-1];
% crop_size = [3072-1, 2480-1];
exts = {'*.jpg', '*.png', '*.bmp'};

% save_dir = fullfile(src_path, img_name); 
save_dir = src_path; 

if ~exist(save_dir, 'dir')
    mkdir(save_dir)
end

%% Crop the PIXLESHIFT at first
filepaths = [];
for idx_ext = 1:length(exts)
    filepaths = cat(1, filepaths, dir(fullfile(src_path, [img_name, '-', tasks{1}, exts{idx_ext}])));
end
src_name = filepaths(1).name;
im_gt = imread(fullfile(src_path, src_name));

imshow(im_gt);
h = imrect(gca, [100 100 crop_size(1) crop_size(2)]); % create rectangle on the image
position = wait(h); % get position
position = floor(position); 
[I_crop, rect] = imcrop(im_gt,position); % crop image
imwrite(I_crop, fullfile(save_dir, [src_name, '_crop.png']));
% I_rescale = imresize(im_gt, 0.25); 
% imwrite(I_rescale, fullfile(save_dir, [src_name, '_resize.png']));
% copyfile(fullfile(src_path, src_name), fullfile(save_dir, src_name)); 

for task_id = 2:length(tasks)
    task = tasks{task_id};
    
    filepaths = [];
    for idx_ext = 1:length(exts)
        filepaths = cat(1, filepaths, dir(fullfile(src_path, [img_name, '-', tasks{task_id}, exts{idx_ext}])));
    end
    
    imgc_name = filepaths(1).name;
    img = imread(fullfile(src_path, imgc_name));
    [I_crop, rect] = imcrop(img,position); % crop image
    imwrite(I_crop, fullfile(save_dir, [imgc_name, '_', task, '_crop.png']));
%     I_rescale = imresize(img, 0.25); 
%     imwrite(I_rescale, fullfile(save_dir, [imgc_name, '_', task, '_resize.png']));
end


%% Crop the PIXLESHIFT at first
