from torch.autograd import Variable

from .FileTools import _is_image_file
from .Prepro import _id
from ..Functions.functional import *


def _add_batch_one(tensor):
    """
    Return a tensor with size (1, ) + tensor.size
    :param tensor: 2D or 3D tensor
    :return: 3D or 4D tensor
    """
    return tensor.view((1, ) + tensor.size())


def _remove_batch(tensor):
    """
    Return a tensor with size tensor.size()[1:]
    :param tensor: 3D or 4D tensor
    :return: 2D or 3D tensor
    """
    return tensor.view(tensor.size()[1:])


def PIL2Tensor(img):
    """
    Converts a PIL Image or numpy.ndarray (H, W, C) in the range [0, 255] to a torch.FloatTensor of shape (1, C, H, W) in the range [0.0, 1.0].

    :param img: PIL.Image or numpy.ndarray (H, W, C) in the range [0, 255]
    :return: 4D tensor with size [1, C, H, W] in range [0, 1.]
    """
    return _add_batch_one(to_tensor(img))


def Tensor2PIL(tensor, mode=None):
    """
    :param tensor: 4D tensor with size [1, C, H, W] in range [0, 1.]
    :param mode: (`PIL.Image mode`_): color space and pixel depth of input data (optional).
                 PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes
    :return: PIL.Image
    """
    if len(tensor.size()) == 3:
        return to_pil_image(tensor, mode=mode)
    elif len(tensor.size()) == 4:
        return to_pil_image(_remove_batch(tensor))


def PIL2VAR(img, norm_function=_id, volatile=False):
    """
    Convert a PIL.Image to Variable directly
    :param img: PIL.Image
    :param norm_function: The normalization to the tensor
    :return: Variable
    """
    return Variable(norm_function(PIL2Tensor(img)), volatile=volatile)


def VAR2PIL(img, non_norm_function=_id):
    """
    Convert a Variable to PIL.Image
    :param img: Variable
    :param non_norm_function: according to the normalization function, the `inverse` normalization
    :return: PIL.Image
    """
    return Tensor2PIL(non_norm_function(img.data))


def pil_loader(path, mode='RGB'):
    """
    open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    :param path: image path
    :return: PIL.Image
    """
    assert _is_image_file(path), "%s is not an image" % path
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert(mode)


def load_to_tensor(path, mode='RGB'):
    """
    Load image to tensor
    :param path: image path
    :param mode: 'Y' returns 1 channel tensor, 'RGB' returns 3 channels, 'RGBA' returns 4 channels, 'YCbCr' returns 3 channels
    :return: 3D tensor
    """
    if mode != 'Y':
        return to_tensor(pil_loader(path, mode=mode))
    else:
        return to_tensor(pil_loader(path, mode='YCbCr'))[:1]


