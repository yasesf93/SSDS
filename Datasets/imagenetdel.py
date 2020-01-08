"""Dataset class for loading imagenet data."""

import os
import sys
from torch.utils import data as data_utils
from torchvision import datasets as torch_datasets
from torchvision import transforms
import torch
import Datasets

def get_train_loader(imagenet_path, batch_size, num_workers):
    train_dataset = ImageNet(imagenet_path, is_train=True)
    return data_utils.DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
        num_workers=num_workers)


def get_val_loader(imagenet_path, batch_size, num_workers):
    val_dataset = ImageNet(imagenet_path, is_train=False)
    return data_utils.DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size, pin_memory=True,
        num_workers=num_workers)

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d) or 'deltas' in d:
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    if 'deltas' not in item[0]:
                        images.append(item)

    return images

class ImageNet(torch_datasets.VisionDataset):
    """Dataset class for ImageNet dataset.
    Arguments:
        root_dir (str): Path to the dataset root directory, which must contain
            train/ and val/ directories.
        is_train (bool): Whether to read training or validation images.
    """
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, root_dir, is_train, transform=None, target_transform=None, is_valid_file=None):
        if is_train:
            # root_dir = os.path.join(root_dir, 'train')
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(ImageNet.MEAN, ImageNet.STD),
            ])
        else:
            # root_dir = os.path.join(root_dir, 'val')
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(ImageNet.MEAN, ImageNet.STD),
            ])

        self.root = root_dir
        self.loader = torch_datasets.folder.default_loader
        self.transform = transform
        self.target_transform = target_transform
        self.classes, self.class_to_idx = self._find_classes(self.root)
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        self.samples = make_dataset(self.root, self.class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))
        self.targets = [s[1] for s in self.samples]
        self.imgs = self.samples
        self.invtrans = transforms.ToPILImage()
        self.trans = transforms.ToTensor()

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and 'delta' not in d.name]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        folder, file = os.path.split(path)
        basefolder, _ = os.path.split(folder)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if os.path.exists(os.path.join(basefolder,'deltas',file)):
            delta = self.trans(self.loader(os.path.join(basefolder,'deltas',file)))
        else:
            delta = torch.zeros_like(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (sample, delta), target

    def set_delta(self, index, delta):
        path, target = self.samples[index]
        folder, file = os.path.split(path)
        basefolder, _ = os.path.split(folder)
        # if os.path.exists(file+'_delta.JPEG'):
        #     print('overwriting')
        # else:
        #     print('creating')
        delta_img = self.invtrans(delta.cpu())
        if not os.path.exists(os.path.join(basefolder,'deltas')):
            os.makedirs(os.path.join(basefolder,'deltas'))
        delta_img.save(os.path.join(basefolder,'deltas',file))


    def __len__(self):
        return len(self.samples)