import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torch
import cv2
import random
import numpy as np

class Compose:
    """
    Composes several transformations together from a list of transforms.

    Args:
        transforms (list): list of transforms to compose
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

class Resize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.operation = transforms.Resize(size=(size, size), interpolation=interpolation)

    def __call__(self, sample):
        image, mask = sample['rgb'], sample['mask']

        image = self.operation(image)
        mask = self.operation(mask)

        return {'rgb': image, 'mask': mask}

class CentroidCrop:
    """
    Crops the given image and mask at centroid computed from a binary mask.
    It expects a dict as input containing the image and its corresponding mask
    (both np.ndarray). This function only considers squared cropping.

    Args:
        crop_size (int): defines the height and width of the output image.
        thresh ([int, int]): defines a threshold to manually adjust the centroid position
                             if necessary.

        Returns a dictionary containing the cropped image and its corresponding mask.
    """
    def __init__(self, crop_size, thresh=[20, 100]):
        self.crop_size = crop_size
        self.thresh = thresh

    def __call__(self, sample):
        image, mask = sample['rgb'], sample['mask']

        # Pad image and mask
        border = self.crop_size // 2 + max(self.thresh)
        image = cv2.copyMakeBorder(image, border, border, border, border,
                                   cv2.BORDER_CONSTANT, value=[255, 255, 255])
        mask = cv2.copyMakeBorder(mask, border, border, border, border,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Calculate moments for the binary mask
        m = cv2.moments(mask)
        # Calculate the x, y coordinates of center
        c_x = int(m["m10"] / m["m00"])
        c_y = int(m["m01"] / m["m00"])

        x = c_x - self.crop_size // 2 + self.thresh[0]
        y = c_y - self.crop_size // 2 - self.thresh[1]

        # Cropping the image
        image = image[y:y+self.crop_size, x:x+self.crop_size]
        mask = mask[y:y+self.crop_size, x:x+self.crop_size]

        return {'rgb': image, 'mask': mask}


class CenterCrop:
    """
    Crops the given image and mask at the center. It expects a dict as input containing
    the image and its corresponding mask (both np.ndarray). This function only considers
    squared cropping.

    Args:
        crop_size (int): defines the height and width of the output image.

        Returns a dictionary containing the cropped image and its corresponding mask.
    """
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        image, mask = sample['rgb'], sample['mask']

        center = [image.shape[0]// 2, image.shape[1] // 2] # crop center
        x = center[1] - self.crop_size // 2
        y = center[0] - self.crop_size // 2

        # Cropping the image
        image = image[y:y+self.crop_size, x:x+self.crop_size]
        mask = mask[y:y+self.crop_size, x:x+self.crop_size]

        return {'rgb': image, 'mask': mask}

class Normalize(transforms.Normalize):
    """
    As different from the original torchvision Normalize, this function takes as input
    a dict containing the image and its mask. There is no normalization for the mask.

    Args:
        mean (list): mean values per channel
        std (list): std values per channel

        Returns a dictionary containing the normalized image and its corresponding mask
        in its original state.
    """
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        super().__init__(mean=mean, std=std)

    def __call__(self, sample):
        image, mask = sample['rgb'], sample['mask']
        image = super().__call__(image.clone())

        return {'rgb': image, 'mask': mask}

class MaskedRandomCrop:
    """
    Crops the given image and mask at a random location in the mask. We use this function
    to ensure we are outputing a crop containing part of tha hand. The function expects a
    dict as input containing the image and its corresponding mask (both np.ndarray). This
    function only considers squared cropping.

    Args:
        crop_size (int): defines the height and width of the output image.

        Returns a dictionary containing the cropped image and its corresponding mask.
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.kernel = np.ones((5,5), np.uint8) # kernel used to erode the mask

    def __call__(self, sample):
        image, mask = sample['rgb'], sample['mask']

        # Pad image and mask
        border = self.crop_size // 2
        image = cv2.copyMakeBorder(image, border, border, border, border,
                                   cv2.BORDER_CONSTANT, value=[255, 255, 255])
        mask = cv2.copyMakeBorder(mask, border, border, border, border,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Erode the mask to ensure the crop center is not in the hand border
        mask_eroded = cv2.erode(mask, self.kernel, iterations=3)
        coords = np.argwhere(mask_eroded==255) # hand is defined by white pixels
        center = random.choice(coords) # we get a random centroid

        # Calculating the crop center
        x = center[1] - border
        y = center[0] - border

        # Cropping the image and its corresponding mask
        image = image[y:y+self.crop_size, x:x+self.crop_size]
        mask = mask[y:y+self.crop_size, x:x+self.crop_size]

        return {'rgb': image, 'mask': mask}

class ToTensor:
    """
    Just to convert a np.ndarray to a tensor. We added a flag to control if we
    want to scale the inputs to [0,1] range. The function expects a
    dict as input containing the image and its corresponding mask (both np.ndarray).

    Args:
        scale (boolean): if True, we scale the image in range [0,1]

        Returns a dict containing the image and masks as float32 tensors.
    """
    def __init__(self, scale=True):
        self.scale = scale

    def __call__(self, sample):
        image, mask = sample['rgb'], sample['mask']

        # masks usually have no channel dim
        if mask.ndim == 2:
            mask = mask[:, :, None]

        # put it from HWC to CHW format expected by pytorch
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        if self.scale:
            image = image / 255.0
            mask = mask / 255.0

        return {'rgb': torch.from_numpy(np.float32(image)),
                'mask': torch.from_numpy(np.float32(mask))}

class Denormalize(transforms.Normalize):
    """
    This is just to undo the normalization process.
    
    Args:
        mean (list): mean values per channel
        std (list): std values per channel

        Returns a dictionary containing the normalized image and its corresponding mask
        in its original state.
    """
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
