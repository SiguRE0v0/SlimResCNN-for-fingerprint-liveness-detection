import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging
from Utils import preprocess, traversal


class FingerDataset(Dataset):
    def __init__(self, img_dir, img_size=160, augmentations=True, transform=None):
        self.img_dir = img_dir
        self.img_size = img_size
        self.img_list = []
        self.label_list = []
        self.augmentations = augmentations
        self.transform = transform

        logging.info('Creating data list')
        self.img_list, self.label_list = traversal.file_traversal(img_dir)
        logging.info(f'Finished creating data list with {len(self.img_list)} images')

        logging.info('preloading and preprocessing images...')
        self.images = []
        self.labels = []
        for img_path, label in tqdm(zip(self.img_list, self.label_list), total=len(self.label_list), desc=f'preprocess', leave=False):
            img = Image.open(img_path).convert('L')
            patches = preprocess.patching(img, self.img_size)
            for patch in patches:
                self.images.append(patch)
                self.labels.append(label)
                if augmentations:
                    patch_aug, label_aug = preprocess.augmentation(patch, label)
                    self.images.extend(patch_aug)
                    self.labels.extend(label_aug)
            img.close()
        logging.info(f'Finished creating dataset with {len(self.images)} images')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = torch.from_numpy(image.copy()).unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)
        return image.float().contiguous(), label


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    dataset = FingerDataset(img_dir='../data/training')

# class FPDataset(Dataset):
#     def __init__(self, img_dir, transform=None, img_size=224):
#         self.img_dir = img_dir
#         self.img = []
#         self.label = []
#         self.transform = transform
#         self.img_size = img_size
#
#         logging.info('Creating dataset')
#         self.img, self.label = traversal.file_traversal(img_dir)
#         logging.info(f'Finished creating dataset with {len(self.img)} images')
#
#     def __len__(self):
#         return len(self.img)
#
#     def __getitem__(self, idx):
#         img_path = self.img[idx]
#         label = self.label[idx]
#         img = Image.open(img_path).convert('L')
#         if self.transform is not None:
#             img = self.transform(img)
#         image = preprocess.patching(img, self.img_size)
#         image = torch.from_numpy(image.copy()).unsqueeze(0)
#         return image.float().contiguous(), label
