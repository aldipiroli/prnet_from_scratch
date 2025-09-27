from torchvision import transforms


class MoCoAugmentations:
    def __init__(self, config):
        self.config = config
        self.crop_size = config["DATA"]["augm"]["crop_size"]

        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=self.crop_size, scale=(0.2, 1.0), antialias=False),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=int(0.1 * self.crop_size[0])),
            ]
        )

    def augment(self, x):
        return self.transforms(x)
