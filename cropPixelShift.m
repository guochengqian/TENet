clear; clc; 
%% The script for cropping the results
src_dir = '/home/qiang/codefiles/low_level/ISP/ispnet/pretrain/pixelshift/pipeline_resblock_pretrain/pipeline_result';
img_name = '010_output.png'; 


crop_size = 100; 

%% 
img_prefix = split(img_name, '.'); 
img_prefix = img_prefix{1}; 

result_folders = dir(src_dir);
n_folders = length(result_folders); 
i_folder_name = result_folders(3).name;
img_path = fullfile(src_dir, i_folder_name, 'result', img_name); 
rgb = imread(img_path);
imshow(rgb);
h = imrect(gca, [50 50 crop_size crop_size]); % create rectangle on the image
position = wait(h); % get position

[I_crop, rect] = imcrop(rgb,position); % crop image
rect=int32(rect)
x = [rect(1), rect(1)+crop_size]; y = [rect(2), rect(2)+crop_size];

for i_folder=3:n_folders
    i_folder_name = result_folders(i_folder).name;
    img_dir = fullfile(src_dir, i_folder_name, 'result'); 
    img_path = fullfile(img_dir, img_name); 
    rgb = imread(img_path);
    cropped = rgb(y(1):y(2), x(1):x(2), :);
    
    % crop other images. 
    img_save_path = fullfile(img_dir, [img_prefix, '_crop_0', '.png']);
    imwrite(cropped, img_save_path);
end


%% resize
img_prefix = split(img_name, '.'); 
img_prefix = img_prefix{1}; 

result_folders = dir(src_dir);
n_folders = length(result_folders); 

for i_folder=3:n_folders
    i_folder_name = result_folders(i_folder).name;
    img_dir = fullfile(src_dir, i_folder_name, 'result'); 
    img_path = fullfile(img_dir, img_name); 
    rgb = imread(img_path);
    rgb_resize = imresize(rgb, [crop_size*3, crop_size*3]);
    % crop other images. 
    img_save_path = fullfile(img_dir, [img_prefix, '_resize', '.png']);
    imwrite(rgb_resize, img_save_path);
end

%% move results.
dst_dir = '/home/qiang/codefiles/low_level/ISP/ispnet/pretrain/pixelshift/selected_result';
img_prefix = split(img_name, '.'); 
img_prefix = img_prefix{1}; 

result_folders = dir(src_dir);
n_folders = length(result_folders); 

for i_folder=3:n_folders
    i_folder_name = result_folders(i_folder).name;
    img_dir = fullfile(src_dir, i_folder_name, 'result'); 
%     movefile(fullfile(img_dir, img_name), fullfile(dst_dir, [img_prefix, '_', i_folder_name, '.png']));
    movefile(fullfile(img_dir, [img_prefix, '_crop_0', '.png']), fullfile(dst_dir, [img_prefix, '_', i_folder_name, '_crop_0', '.png']));
    movefile(fullfile(img_dir, [img_prefix, '_resize', '.png']), fullfile(dst_dir, [img_prefix, '_', i_folder_name, '_resize', '.png']));
end