clear; clc; 
%% The script for cropping the PixelShift200 
% src_path = '/data/image/ISPNet/pixelshift200/mat';
% save_path = '/data/image/ISPNet/pixelshift200/cropped_mat';

% src_path_mat = '/data/image/ISPNet/pixelshift200/RAW_RGGB/train_raw_rggb';
src_path_mat = '/data/image/ISPNet/pixelshift200/RAW_RGGB/test_raw_rggb';
src_path_img = '/data/image/ISPNet/pixelshift200/RAW_RGGB/img_preview';
debug_path = '/data/image/ISPNet/pixelshift200/debug';
debug = false; 

enable_crop_size=false; % true for test split, false for train split 
crop_size = 1024;

dst_path_mat = '/data/image/ISPNet/pixelshift200/crop_mat';
dst_path_cropinfo = '/data/image/ISPNet/pixelshift200/crop_info';
dst_path_img = '/data/image/ISPNet/pixelshift200/crop_img';
post_len = length('rggb.mat');

if ~exist(dst_path_mat, 'dir')
    mkdir(dst_path_mat);
end
if ~exist(dst_path_cropinfo, 'dir')
    mkdir(dst_path_cropinfo);
end
if ~exist(dst_path_img, 'dir')
    mkdir(dst_path_img);
end

BIT = 14; 


list_mat = dir(fullfile(src_path_mat, '*mat'));
list_mat_names = {list_mat.name};
list_mat_names = sort(list_mat_names);
for img_no=3:length(list_mat_names)
    imgName = list_mat_names{img_no};
    imgNO = imgName(1:end-post_len);
    mat_path = fullfile(src_path_mat, [imgNO,'rggb.mat']);
    load(mat_path);
    img_path = fullfile(src_path_img, [imgNO, 'srgb.jpg']);
    rgb = imread(img_path);

    %% generate rgb img
    imshow(rgb);
    if enable_crop_size
        x=[]; y=[];
        for i = 1:20
            h = imrect(gca, [50 50 1024 1024]); % create rectangle on the image
            position = wait(h); % get position
            if isempty(position) 
                break 
            end
            [I_crop, rect] = imcrop(rgb,position); % crop image
            x = [x, rect(1), rect(1)+crop_size]; y = [y, rect(2), rect(2)+crop_size];
        end
    else
        [x, y] = ginput(10); 
    end

    %% crop imgs and mat
    n_crops = floor(length(x)/2);
    if n_crops>0
        for i = 1:n_crops
            y(i*2-1) = floor(y(i*2-1)/2)*2+1;
            y(i*2) = floor(y(i*2)/2)*2;
            x(i*2-1) = floor(x(i*2-1)/2)*2+1;
            x(i*2) = floor(x(i*2)/2)*2;

            img_crop = rgb(y(i*2-1):y(i*2), x(i*2-1):x(i*2), :); 
            raw = rggb(y(i*2-1):y(i*2), x(i*2-1):x(i*2), :);

            save_path_mat = fullfile(dst_path_mat, [imgNO, 'crop_', num2str(i), '.mat']);
            save_path_img = fullfile(dst_path_img, [imgNO, 'crop_', num2str(i), '.jpg']);

            save(save_path_mat, 'raw', '-v7.3');
            imwrite(img_crop, save_path_img);
        end
    else
        fprintf('discard %s \n', mat_path);
    end

    %% save crop info
    save_cropinfo_mat = fullfile(dst_path_cropinfo, [imgNO, 'crop_info.mat']);
    save(save_cropinfo_mat, 'x', 'y'); 
end
