import argparse
import numpy as np
import torch
from PIL import Image
import torch_dct as dct


class ToCustomTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self):
        pass

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # handle numpy arrays
        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]

            pic = pic/255

            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class TransCropHorizon(object):
    """
    This transformation crops the horizon and fills the cropped area with black
    pixels or delets them completely.
    Args:
        crop_value (float) [0,1]: Percentage of the image to be cropped from
        the total_step
        set_black (bool): sets the cropped pixels black or delets them completely
    """
    def __init__(self, crop_value, set_black=False):
        assert isinstance(set_black, (bool))
        self.set_black = set_black

        if crop_value >= 0 and crop_value < 1:
            self.crop_value = crop_value
        else:
            print('One or more Arg is out of range!')

    def __call__(self, image):
        crop_value = self.crop_value
        set_black = self.set_black
        image_height = image.size[1]
        crop_pixels_from_top = int(round(image_height*crop_value, 0))

        # convert from PIL to np
        image = np.array(image)

        if set_black==True:
            image[:][0:crop_pixels_from_top-1][:] = np.zeros_like(image[:][0:crop_pixels_from_top-1][:])
        else:
            image = image[:][crop_pixels_from_top:-1][:]

        # reconvert again to PIL
        image = Image.fromarray(image)

        return image


class SpectralTransform(torch.nn.Conv2d):
    pass


class SpectralTransformInverse(torch.nn.Conv2d):
    pass


def to_spectral(x):
    return dct.dct_2d(x)


def to_spatial(x):
    return dct.idct_2d(x)


def get_parser():
    # Parser for training settings
    parser = argparse.ArgumentParser()

    parser.add_argument("--host",
                        type=str,
                        default='rudolf',
                        help="local, rudolf or leonhard ")

    parser.add_argument("--gpu",
                        type=int,
                        default=False,
                        help="CUDA:n where n = [0, 1, 2, 3, 4, 5, 6 ,7]")
    parser.add_argument("--workers",
                        type=int,
                        default=1,
                        help="Number of cpu workers [1:8]")
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="Model to be trained: [VanillaCNN, SpectralDropoutCNN, SpectralDropoutEasyCNN]")
    parser.add_argument("--epochs",
                        type=int,
                        default=1,
                        help="Training epochs")
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="Batch size")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="Learning rate")
    parser.add_argument("--dataset",
                        type=str,
                        default='real',
                        help="Sim: 'sim'; real: 'real'; both='sim_real'")
    parser.add_argument("--validation_split",
                        type=float,
                        default=0.2,
                        help="Validation split size")
    parser.add_argument("--image_res",
                        type=int,
                        default=64,
                        help="Resolution of image (not squared, just used for rescaling")
    return parser
