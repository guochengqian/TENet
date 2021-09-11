function Evaluate_PSNR_SSIM()
clear all; close all; clc
tic

%%
% div2k 
% pretrain_dataset = 'pixelshift'; 
% dataset = 'pixelshift'; 
pretrain_dataset = 'div2k'; 
dataset = 'urban100'; 
% dataset = 'cbsd68'; 
% dataset = 'set14'; 
% dataset = 'div2k'; 

% pipeline comprasion
% tasks = {'dm-dn-sr-SR2', 'dm-sr-dn-SR2', 'dn-dm-sr-SR2', 'dn-sr-dm-SR2', 'sr-dm-dn-SR2', 'sr-dn-dm-SR2', 'dn-dm+sr-SR2', 'dn+dm-sr-SR2', 'dn+sr-dm-SR2', 'e2e-dn+dm+sr-SR2', 'e2e-dn-dm+sr-SR2', 'e2e-dn+dm-sr-SR2', 'e2e-dn+sr-dm-SR2'}; 

% sota model
% tasks = {'jdsr-dn+dm+sr-SR2', 'jdsr-dn+sr-dm-SR2', 'jdndmsr-dn+dm+sr-SR2', 'jdndmsr-dn+sr-dm-SR2', 'nlsa-dn+dm+sr-SR2', 'nlsa-dn+sr-dm-SR2'}; 


% % tasks = {'resnet-dn+dm+sr-SR2', 'resnet-dn+sr-dm-SR2'}; 
tasks = {'e2e-dn+dm+sr-SR2', 'e2e-dn+sr-dm-SR2'}; 

gt_path = fullfile('/home/qiang/codefiles/low_level/ISP/ispnet/data/benchmark/', dataset, 'gt');
pred_dir = fullfile(fullfile('/home/qiang/codefiles/low_level/ISP/ispnet/pretrain/', pretrain_dataset, 'pipeline'), ['result_', dataset]);
% tasks = {'dn-sr-dm-SR2', 'sr-dm-dn-SR2', 'sr-dn-dm-SR2', }; 
% tasks = {'dm-dn-sr-SR2', 'dm-sr-dn-SR2', 'dn-dm-sr-SR2', 'dn-sr-dm-SR2', 'sr-dn-dm-SR2', 'dn-dm+sr-SR2', 'dn+dm-sr-SR2', 'dn+sr-dm-SR2'}; 
% tasks = {'dn-dm+sr-SR2', 'dn+dm-sr-SR2', 'dn+sr-dm-SR2'}; 
% tasks = {'sr-dm-dn-SR2', 'dn+sr-dm-SR2'}; 
% tasks = {'jdndmsr-dn+sr-dm-SR2'}; 
% tasks = {'jdndmsr-dn+sr-dm-SR2'}; 
% tasks = {'e2e-dn+dm+sr-SR2', 'e2e-dn-dm+sr-SR2', 'e2e-dn+dm-sr-SR2', 'e2e-dn+sr-dm-SR2'}; 

% %% for demosaicking test
% gt_path = '/home/qiang/codefiles/low_level/ISP/ispnet/data/benchmark/pixelshift200/srgb';
% pred_dir = '/home/qiang/codefiles/low_level/ISP/ispnet/pretrain/pixelshift/no_end2end_pretrain/pipe_result';

% tasks = {'dm-ps-ps'}; % DM trained on PS200 and test on PS200.  
% tasks = {'dm-div2k-ps'};

%%
% gt_path = '/home/qiang/Downloads/TENet_sim_test/gt/urban100';


%%
exts = {'*.jpg', '*.png', '*.bmp'};

for task_id = 1:length(tasks)
    task = tasks{task_id};
    
    % for each task
    pred_path = fullfile(pred_dir, task, 'result');

    %% 
    filename = [task, '.text'];
    file_path = fullfile(pred_dir, task,filename);
    fileID = fopen(file_path,'w');
    
    % print on file
    fprintf(fileID, '**********************\n');
    fprintf(fileID, 'task: %s \n', task);
    
    % print on terminal
    fprintf('**********************\n');
    fprintf('task: %s \n', task);
    
    filepaths = [];
    for idx_ext = 1:length(exts)
        filepaths = cat(1, filepaths, dir(fullfile(gt_path, exts{idx_ext})));
    end
    PSNR_all = zeros(1, length(filepaths));
    SSIM_all = zeros(1, length(filepaths));
    for idx_im = 1:length(filepaths)
        gt_name = filepaths(idx_im).name;
        ext = strsplit(gt_name, '.');
        ext = ext{2};
        pred_name = [sprintf('%03d', idx_im), '_output', '.png'];
        im_gt = imread(fullfile(gt_path, gt_name));
        im_pred = imread(fullfile(pred_path, pred_name));
        [H, W, ~] = size(im_pred);

        im_gt = double(im_gt);
        im_gt = im_gt(1:H, 1:W, :);
        im_pred = double(im_pred);
        % calculate PSNR, SSIM
        PSNR_all(idx_im) = csnr(im_gt, im_pred, 0, 0);
        [~, SSIM_all(idx_im)] = Cal_PSNRSSIM(im_gt, im_pred, 0, 0);
        fprintf('%d %s: PSNR= %f SSIM= %f\n', idx_im, gt_name, PSNR_all(idx_im), SSIM_all(idx_im));
        fprintf(fileID, '%d %s: PSNR= %f SSIM= %f\n', idx_im, gt_name, PSNR_all(idx_im), SSIM_all(idx_im));

    end
    fprintf(fileID, '--------Avg PSNR %s--------\n', task);
    fprintf(fileID, 'PSNR= %f , SSIM= %f \n', mean(PSNR_all), mean(SSIM_all));
    fclose(fileID);
    fprintf('--------Avg PSNR %s--------\n', task);
    fprintf('PSNR= %f , SSIM= %f \n\n\n', mean(PSNR_all), mean(SSIM_all));
end

end % add for function 

%% sub functions 
function s=csnr(A,B,row,col)

[n,m,ch]=size(A);

if ch==1
   e=A-B;
   e=e(row+1:n-row,col+1:m-col);
   me=mean(mean(e.^2));
   s=10*log10(255^2/me);
else
   e=A-B;
   e=e(row+1:n-row,col+1:m-col,:);
   e1=e(:,:,1);e2=e(:,:,2);e3=e(:,:,3);
   me1=mean(mean(e1.^2));
   me2=mean(mean(e2.^2));
   me3=mean(mean(e3.^2));
   mse=(me1+me2+me3)/3;
   s  = 10*log10(255^2/mse);
end
end

function [psnr_cur, ssim_cur] = Cal_PSNRSSIM(A,B,row,col)


[n,m,ch]=size(B);
A = A(row+1:n-row,col+1:m-col,:);
B = B(row+1:n-row,col+1:m-col,:);
A=double(A); % Ground-truth
B=double(B); %

e=A(:)-B(:);
mse=mean(e.^2);
psnr_cur=10*log10(255^2/mse);

if ch==1
    [ssim_cur, ~] = ssim_index(A, B);
else
    ssim_cur = (ssim_index(A(:,:,1), B(:,:,1)) + ssim_index(A(:,:,2), B(:,:,2)) + ssim_index(A(:,:,3), B(:,:,3)))/3;
end
end

function [mssim, ssim_map] = ssim_index(img1, img2, K, window, L)

%========================================================================
%SSIM Index, Version 1.0
%Copyright(c) 2003 Zhou Wang
%All Rights Reserved.
%
%The author is with Howard Hughes Medical Institute, and Laboratory
%for Computational Vision at Center for Neural Science and Courant
%Institute of Mathematical Sciences, New York University.
%
%----------------------------------------------------------------------
%Permission to use, copy, or modify this software and its documentation
%for educational and research purposes only and without fee is hereby
%granted, provided that this copyright notice and the original authors'
%names appear on all copies and supporting documentation. This program
%shall not be used, rewritten, or adapted as the basis of a commercial
%software or hardware product without first obtaining permission of the
%authors. The authors make no representations about the suitability of
%this software for any purpose. It is provided "as is" without express
%or implied warranty.
%----------------------------------------------------------------------
%
%This is an implementation of the algorithm for calculating the
%Structural SIMilarity (SSIM) index between two images. Please refer
%to the following paper:
%
%Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
%quality assessment: From error measurement to structural similarity"
%IEEE Transactios on Image Processing, vol. 13, no. 1, Jan. 2004.
%
%Kindly report any suggestions or corrections to zhouwang@ieee.org
%
%----------------------------------------------------------------------
%
%Input : (1) img1: the first image being compared
%        (2) img2: the second image being compared
%        (3) K: constants in the SSIM index formula (see the above
%            reference). defualt value: K = [0.01 0.03]
%        (4) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%        (5) L: dynamic range of the images. default: L = 255
%
%Output: (1) mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.
%        (2) ssim_map: the SSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size:
%            size(img1) - size(window) + 1.
%
%Default Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255
%
%   [mssim ssim_map] = ssim_index(img1, img2);
%
%Advanced Usage:
%   User defined parameters. For example
%
%   K = [0.05 0.05];
%   window = ones(8);
%   L = 100;
%   [mssim ssim_map] = ssim_index(img1, img2, K, window, L);
%
%See the results:
%
%   mssim                        %Gives the mssim value
%   imshow(max(0, ssim_map).^4)  %Shows the SSIM index map
%
%========================================================================


if (nargin < 2 || nargin > 5)
    ssim_index = -Inf;
    ssim_map = -Inf;
    return;
end

if (size(img1) ~= size(img2))
    ssim_index = -Inf;
    ssim_map = -Inf;
    return;
end

[M N] = size(img1);

if (nargin == 2)
    if ((M < 11) || (N < 11))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    window = fspecial('gaussian', 11, 1.5);	%
    K(1) = 0.01;								      % default settings
    K(2) = 0.03;								      %
    L = 255;                                  %
end

if (nargin == 3)
    if ((M < 11) || (N < 11))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    window = fspecial('gaussian', 11, 1.5);
    L = 255;
    if (length(K) == 2)
        if (K(1) < 0 || K(2) < 0)
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
    else
        ssim_index = -Inf;
        ssim_map = -Inf;
        return;
    end
end

if (nargin == 4)
    [H W] = size(window);
    if ((H*W) < 4 || (H > M) || (W > N))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    L = 255;
    if (length(K) == 2)
        if (K(1) < 0 || K(2) < 0)
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
    else
        ssim_index = -Inf;
        ssim_map = -Inf;
        return;
    end
end

if (nargin == 5)
    [H W] = size(window);
    if ((H*W) < 4 || (H > M) || (W > N))
        ssim_index = -Inf;
        ssim_map = -Inf;
        return
    end
    if (length(K) == 2)
        if (K(1) < 0 || K(2) < 0)
            ssim_index = -Inf;
            ssim_map = -Inf;
            return;
        end
    else
        ssim_index = -Inf;
        ssim_map = -Inf;
        return;
    end
end

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));
img1 = double(img1);
img2 = double(img2);

mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

if (C1 > 0 & C2 > 0)
    ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
    numerator1 = 2*mu1_mu2 + C1;
    numerator2 = 2*sigma12 + C2;
    denominator1 = mu1_sq + mu2_sq + C1;
    denominator2 = sigma1_sq + sigma2_sq + C2;
    ssim_map = ones(size(mu1));
    index = (denominator1.*denominator2 > 0);
    ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
    index = (denominator1 ~= 0) & (denominator2 == 0);
    ssim_map(index) = numerator1(index)./denominator1(index);
end

mssim = mean2(ssim_map);

end