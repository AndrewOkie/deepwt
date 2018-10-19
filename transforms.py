import math
import random

import numpy as np
import cv2

import numbers
import types
import collections

import torch


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.RandomHorizontalFlip(),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):

    def __call__(self, image, mask):
        h, w = image.shape

        image_out = torch.from_numpy(image.copy()).view(1, h, w)
        mask_out = torch.from_numpy(mask.astype(np.int32).copy())

        return image_out.float().div(255), mask_out.long()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        """
        Args:
            image (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if not torch.is_tensor(image) or not image.ndimension() == 3:
            raise TypeError('image is not a torch image.')

        # TODO: make efficient
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)

        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize(object):
    """Resize the input sample to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
    """

    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, image, mask):
        if not (isinstance(self.size, int) or (isinstance(self.size, collections.Iterable) and len(self.size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(self.size))

        if isinstance(self.size, int):
            h, w = image.shape
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return image, mask
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
            else:
                oh = self.size
                ow = int(self.size * w / h)
            return cv2.resize(image, (ow, oh), interpolation=cv2.INTER_LINEAR), cv2.resize(mask, (oh, ow), interpolation=cv2.INTER_NEAREST)
        else:
            return cv2.resize(image, self.size[::-1], interpolation=cv2.INTER_LINEAR), cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, image, mask):
        return self.lambd(image, mask)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomTransforms(object):
    """Base class for a list of transformations with randomness

    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApply(RandomTransforms):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def __call__(self, image, mask):
        if self.p < random.random():
            return image, mask
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomOrder(RandomTransforms):
    """Apply a list of transformations in a random order
    """
    def __call__(self, image, mask):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            image, mask = self.transforms[i](image, mask)
        return image, mask


class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list
    """
    def __call__(self, image, mask):
        t = random.choice(self.transforms)
        return t(image, mask)


class RandomCrop(object):
    """Crop the given image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(image, output_size):
        h, w = image.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, mask):
        if isinstance(self.padding, collections.Sequence) and len(self.padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a {} element tuple".format(len(self.padding)))

        if self.padding > 0:
            image = np.pad(image, self.padding, 'constant', constant_values=0)
            mask = np.pad(mask, self.padding, 'constant', constant_values=0)

        i, j, h, w = self.get_params(image, self.size)

        return image[i:i + h, j:j + w], mask[i:i + h, j:j + w]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    """Horizontally flip the given image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            return np.flip(image, axis=1), np.flip(mask, axis=1)

        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            return np.flip(image, axis=0), np.flip(mask, axis=0)

        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCrop(object):
    """Crop the given image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(image, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            image (image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = image.shape[0] * image.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            h = int(round(math.sqrt(target_area / aspect_ratio)))
            w = int(round(math.sqrt(target_area * aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if h <= image.shape[0] and w <= image.shape[1]:
                i = random.randint(0, image.shape[0] - h)
                j = random.randint(0, image.shape[1] - w)
                return i, j, h, w

        # Fallback
        w = min(image.shape[0], image.shape[1])
        i = (image.shape[0] - w) // 2
        j = (image.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, image, mask):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)

        image = image[i:i + h, j:j + w]
        mask = mask[i:i + h, j:j + w]

        resize = Resize(self.size)

        return resize(image, mask)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(round(self.scale, 4))
        format_string += ', ratio={0})'.format(round(self.ratio, 4))
        return format_string


class AdditiveGaussianNoise(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        noise = torch.zeros_like(image).normal_(self.mean, self.std)
        return image + noise, mask


class MultiplicativeGaussianNoise(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        noise = torch.zeros_like(image).normal_(self.mean, self.std)
        return image * noise, mask
