import os
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class UIEBDataset(data.Dataset):
    def __init__(
        self,
        train_dataset,
        crop_size,
        test_dataset,
        mode='train'
    ):
        self.raw_path = os.path.join(train_dataset, 'raw-890')
        self.gt_path = os.path.join(train_dataset, 'reference-890')
        self.crop_size = crop_size

        train_list_path = os.path.join(train_dataset, 'uie_train_list.txt')
        test_list_path = os.path.join(test_dataset, 'uie_test_list.txt')

        self.mode = mode

        if self.mode == 'train':
            f = open(train_list_path)
        elif self.mode in ['test', 'valid']:
            f = open(test_list_path)

        self.filenames = f.readlines()
        f.close()

    def __getitem__(self, item):
        raw_item_path = os.path.join(self.raw_path, self.filenames[item].rstrip())
        gt_item_path = os.path.join(self.gt_path, self.filenames[item].rstrip())

        raw_img = Image.open(raw_item_path)
        gt_img = Image.open(gt_item_path)

        transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor()
        ])

        return {
            'in_img': transform(raw_img),
            'label': transform(gt_img),
            'filename': self.filenames[item].rstrip()
        }

    def __len__(self):
        return len(self.filenames)

class UIEBUnpairedDataset(data.Dataset):
    def __init__(
        self,
        train_dataset,
        crop_size
    ):
        self.raw_path = os.path.join(train_dataset, 'challenging-60')
        self.crop_size = crop_size
        non_reference_list_path = os.path.join(train_dataset, 'uie_non_reference.txt')

        f = open(non_reference_list_path)
        self.filenames = f.readlines()
        f.close()

    def __getitem__(self, item):
        raw_item_path = os.path.join(self.raw_path, self.filenames[item].rstrip())
        raw_img = Image.open(raw_item_path)

        transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor()
        ])

        transformed_img = transform(raw_img)

        return {
            'in_img': transformed_img,
            'filename': self.filenames[item].rstrip()
        }

    def __len__(self):
        return len(self.filenames) 