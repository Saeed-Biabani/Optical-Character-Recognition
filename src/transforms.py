from torchvision.transforms import functional as trf
from torchvision.transforms import v2
import torch

class Transforms:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __normalize__(self, image):
        return (image - image.mean()) / image.std()

    def __resize__(self, image):
        return trf.resize(
            image,
            size = self.size,
            antialias = True,
            interpolation = trf.InterpolationMode.BICUBIC
        )

    def __call__(self, image):
        return self.__normalize__(
            self.__resize__(
                trf.to_tensor(
                    image
                ).float()
            )
        )

class ImageTransforms:
    def __init__(self, cfg):
        self.size = cfg.image_size

    def train(self):
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale = True),
            v2.Resize(self.size),
            v2.RandomRotation(1),
            v2.RandomApply([
                v2.GaussianBlur(9),
                v2.GaussianNoise(),
                v2.Lambda(lambd = lambda x : x)
            ]),
            v2.RandomPerspective(0.1),
            v2.ElasticTransform(alpha = 5),
            v2.Lambda(lambd = lambda image : (image - image.mean()) / image.std())
        ])

    def test(self):
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale = True),
            v2.Resize(self.size),
            v2.Lambda(lambd = lambda image : (image - image.mean()) / image.std())
        ])