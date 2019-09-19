import torchvision.transforms as T
from .RandomErasing import RandomErasing
from PIL import Image
import random, math

def transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        # transform = T.Compose([
        #     T.Resize(cfg.INPUT.SIZE_TRAIN),
        #     T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        #     T.Pad(cfg.INPUT.PADDING),
        #     T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        #     T.ColorJitter(brightness=cfg.INPUT.BRIGHTNESS,
        #       contrast=cfg.INPUT.CONTRAST,
        #       saturation=cfg.INPUT.SATURATION,
        #       hue=cfg.INPUT.HUE),
        #     T.ToTensor(),
        #     normalize_transform,
        #     RandomErasing(probability=cfg.INPUT.PROB, mean=cfg.INPUT.PIXEL_MEAN),
        # ])
        transform = T.Compose([
            RandomSizedRectCrop(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize_transform,
        ])
    else:
        # transform = T.Compose([
        #     T.Resize(cfg.INPUT.SIZE_TEST),
        #     T.ToTensor(),
        #     normalize_transform
        # ])
        transform = T.Compose([
            RectScale(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),
            T.ToTensor(),
            normalize_transform,
        ])

    return transform

class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)

class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)