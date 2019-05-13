# DataTools
###### JasonGUTU
This package provides some useful functions and DataSet classes for Image processing and Low-level Computer Vision.
### Structure
- `DataSets` contains some DataSet class, all the child class of torch.utils.data.Dataset.
- `FileTools` contains tools for file management
- `Loaders` contains Image loaders
- `Prepro` contains self-customized pre-processing functions or classes
### Docs
#### DataSets.py
All the classes inherited from torch.utils.data.Dataset are self-customized Dataset classes
```[Python]
# `TestDataset` is a Dataset classes
# Instantiation
dataset = TestDataset(*args, **kwargs)
# Use index to retrieve
first_data = dataset[0]
# Number of samples
length = len(dataset)
```
In this file, Datasets contain:
```
class SRDataSet(torch.utils.data.Dataset)
"""
:param data_path: Path to data root
:param lr_patch_size: the Low resolution size, by default, the patch is square
:param scala: SR scala, default is 4
:param interp: interpolation for resize, default is Image.BICUBIC, optional [Image.BILINEAR, Image.BICUBIC]
:param mode: 'RGB' or 'Y'
:param sub_dir: if True, then all the images in the `data_path` directory AND child directory will be use
:parem prepro: function fo to ``PIL.Image``!, will run this function before crop and resize
"""
```
This Dataset is for loading small images like image-91 and image-191.
The images are small, direct loading has little effect on performance.
In this dataset, every image will be returned once in one epoch.
Every time one image is return will be pre-processing and then random crop a patch.
If the patch size is bigger than image size, the image will be resize to a 'cropable' size and random crop a patch.

```
class SRDataLarge(torch.utils.data.Dataset)
"""
:param data_path: Path to data root
:param lr_patch_size: the Low resolution size, by default, the patch is square
:param scala: SR scala, default is 4
:param interp: interpolation for resize, default is Image.BICUBIC, optional [Image.BILINEAR, Image.BICUBIC]
:param mode: 'RGB' or 'Y'
:param sub_dir: if True, then all the images in the `data_path` directory AND child directory will be use
:parem prepro: function fo to ``PIL.Image``!, will run this function before crop and resize
:param buffer: how many patches cut from one image
"""
```
This Dataset is for loading large images like DIV2K.
The images are large, direct loading has effect on performance.
In this dataset, every image will be returned buffer number of patches in one epoch.
Every time one image is return will be pre-processing and then random crop a patch.

#### FileTools.py
```[Python]
def _is_image_file(filename):
    """
    judge if the file is an image file
    :param filename: path
    :return: bool of judgement
    """
```
```[Python]
def _video_image_file(path):
    """
    Data Store Format:

    Data Folder
        |
        |-Video Folder
        |   |-Video Frames (images)
        |
        |-Video Folder
            |-Video Frames (images)

        ...

        |-Video Folder
            |-Video Frames (images)

    :param path: path to Data Folder, absolute path
    :return: 2D list of str, the path is absolute path
            [[Video Frames], [Video Frames], ... , [Video Frames]]
    """
```
```[Python]
def _sample_from_videos_frames(path, time_window, time_stride):
    """
    Sample from video frames files
    :param path: path to Data Folder, absolute path
    :param time_window: number of frames in one sample
    :param time_stride: strides when sample frames
    :return: 2D list of str, absolute path to each frames
            [[Sample Frames], [Sample Frames], ... , [Sample Frames]]
    """
```
```[Python]
def _image_file(path):
    """
    return list of images in the path
    :param path: path to Data Folder, absolute path
    :return: 1D list of image files absolute path
    """
```
```[Python]
def _all_images(path):
    """
    return all images in the folder, include child folder.
    :param path: path to Data Folder, absolute path
    :return: 1D list of image files absolute path
    """
```

#### Loaders.py

```[Python]
def _add_batch_one(tensor):
    """
    Return a tensor with size (1, ) + tensor.size
    :param tensor: 2D or 3D tensor
    :return: 3D or 4D tensor
    """
```
```[Python]
def _remove_batch(tensor):
    """
    Return a tensor with size tensor.size()[1:]
    :param tensor: 3D or 4D tensor
    :return: 2D or 3D tensor
    """
```
```[Python]
def PIL2Tensor(img):
    """
    Converts a PIL Image or numpy.ndarray (H, W, C) in the range [0, 255] to a torch.FloatTensor of shape (1, C, H, W) in the range [0.0, 1.0].

    :param img: PIL.Image or numpy.ndarray (H, W, C) in the range [0, 255]
    :return: 4D tensor with size [1, C, H, W] in range [0, 1.]
    """
```
```[Python]
def Tensor2PIL(tensor, mode=None):
    """
    Convert a 4D tensor with size [1, C, H, W] to PIL Image
    :param tensor: 4D tensor with size [1, C, H, W] in range [0, 1.]
    :param mode: (`PIL.Image mode`_): color space and pixel depth of input data (optional).
                 PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes
    :return: PIL.Image
    """
```
```[Python]
def PIL2VAR(img, norm_function=_id):
    """
    Convert a PIL.Image to Variable directly, add batch dimension (can be use to test directly)
    :param img: PIL.Image
    :param norm_function: The normalization to the tensor
    :return: Variable
    """
```
```[Python]
def VAR2PIL(img, non_norm_function=_id):
    """
    Convert a Variable to PIL.Image
    :param img: Variable
    :param non_norm_function: according to the normalization function, the `inverse` normalization
    :return: PIL.Image
    """
```
```[Python]
def pil_loader(path, mode='RGB'):
    """
    open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    :param path: image path
    :return: PIL.Image
    """
```
```[Python]
def load_to_tensor(path, mode='RGB'):
    """
    Load image to tensor, 3D tensor
    :param path: image path
    :param mode: 'Y' returns 1 channel tensor, 'RGB' returns 3 channels, 'RGBA' returns 4 channels, 'YCbCr' returns 3 channels
    :return: 3D tensor
    """
```



