import os
import torch
from PIL import Image
import numpy as np 
# from libtiff import TIFF
# import pdb
import cv2


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tiff', '.TIFF', '.tif']


def _is_image_file(filename):
    """
    judge if the file is an image file
    :param filename: path
    :return: bool of judgement
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def _image_file(path):  # TODO: wrong function
    """
    return list of images in the path
    :param path: path to Data Folder, absolute path
    :return: 1D list of image files absolute path
    """
    abs_path = os.path.abspath(path)
    image_files = os.listdir(abs_path)
    for i in range(len(image_files)):
        if (not os.path.isdir(image_files[i])) and (_is_image_file(image_files[i])):
            image_files[i] = os.path.join(abs_path, image_files[i])
    return image_files


def _all_images(path):
    """
    return all images in the folder
    :param path: path to Data Folder, absolute path
    :return: 1D list of image files absolute path
    """
    # TODO: Tail Call Elimination
    abs_path = os.path.abspath(path)
    image_files = list()
    # num = 0
    if os.path.isfile(abs_path):
        return [abs_path]
    else:
        for subpath in os.listdir(abs_path):
            if os.path.isdir(os.path.join(abs_path, subpath)):
                image_files = image_files + _all_images(os.path.join(abs_path, subpath))
            else:
                if _is_image_file(subpath):
                    # num = num +1
                    image_files.append(os.path.join(abs_path, subpath))
        image_files.sort()
        return image_files


def _read_image(path):
    """
    :param path:
    :return: read PIL Image and change channel sequence(same as torch) in np format
    """
    img = np.asarray(Image.open(path))
    # size: w*h*3
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.
    else:
        raise SystemExit('dtype is not supported')

    return img


# def _read_tiff(path, bits):
#     img = TIFF.open(path, mode='r')
#     img = img.read_image()
#     img = img.astype(np.float32) / 2 **bits
#     return img


def _cv2_read_images(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.
    else:
        raise SystemExit('dtype is not supported')
    return img


def _uint2float(I):
    if I.dtype == np.uint8:
        I = I.astype(np.float32)
        I = I*0.00390625
    elif I.dtype == np.uint16:
        I = I.astype(np.float32)
        I = I/65535.0
    else:
        raise ValueError("not a uint type {}".format(I.dtype))

    return I


def _float2uint(I, dtype):
    # pdb.set_trace()
    if dtype == np.uint8:
        # I /= 0.00390625
        I = I*255.
        I += 0.5
        I = np.clip(I,0,255)
        I = I.astype(np.uint8)
    elif dtype == np.uint16:
        I = I*65535.
        I += 0.5
        I = np.clip(I,0,65535)
        I = I.astype(np.uint16)
    else:
        raise ValueError("not a uint type {}".format(dtype))

    return I


def _tensor2cvimage(tensor, dtype):
    tensor = torch.clamp(tensor.cpu(), 0, 1).detach()
    img = tensor[0, :, :, :] if len(tensor.shape) == 4 else tensor
    if img.shape[0] == 3:
        return cv2.cvtColor(_float2uint(img.numpy().transpose(1, 2, 0), dtype), cv2.COLOR_RGB2BGR)
    else:
        return _float2uint(img[0, :, :].numpy(), dtype)


def _tensor2image(tensor, dtype):
    tensor = torch.clamp(tensor.cpu(), 0, 1).detach()
    img = tensor[0, :, :, :] if len(tensor.shape) == 4 else tensor
    if img.shape[0] == 3:
        return _float2uint(img.numpy().transpose(1, 2, 0), dtype)
    else:
        return _float2uint(img[0, :, :].numpy(), dtype)


def _pil2cv(img):
    im = np.asarray(img)
    if im.shape[2] == 3:
        return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    else:
        return im


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    Save tensor to cv2 format
         :param input_tensor: tensor to save
         :param filename: saved file name
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # Make a copy
    input_tensor = input_tensor.clone().detach()
    # To cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # Denormalization
    # input_tensor = unnormalize(input_tensor)
    # Remove batch dimension
    input_tensor = input_tensor.squeeze()
    # Convert from [0,1] to [0,255], then from CHW to HWC, and finally to cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB to BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


