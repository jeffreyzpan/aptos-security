import os
import numpy as np
import pandas as pd
import cv2

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.utils import save_image

import torchvision.datasets as datasets
from PIL import Image


class AdjustContrast:

    def __init__(self, adjustment=1):
        self.adjustment = adjustment

    def __call__(self, x):
        return transforms.functional.adjust_contrast(x, self.adjustment)

class APTOSPreprocess:

    def __init__(self, sigmaX, tol):
        self.sigmaX = sigmaX
        self.tol = tol

    def __call__(self, x):
        return preprocess_image(x, self.sigmaX)

def get_data_statistics(data_loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for (data, labels) in data_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std

def crop_image_from_gray(img, tol=7):
    """
    Applies masks to the orignal image and
    returns the a preprocessed image with
    3 channels

    :param img: A NumPy Array that will be cropped
    :param tol: The tolerance used for masking

    :return: A NumPy array containing the cropped image
    """
    # If for some reason we only have two channels
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
        # If we have a normal RGB images
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def preprocess_image(image, sigmaX=10):
    """
    The whole preprocessing pipeline:
    1. Read in image
    2. Apply masks
    3. Resize image to desired size
    4. Add Gaussian noise to increase Robustness

    :param img: A NumPy Array that will be cropped
    :param sigmaX: Value used for add GaussianBlur to the image

    :return: A NumPy array containing the preprocessed image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)
    return image

class APTOSDataset(data.Dataset):
    def __init__(self, csv_path, img_path, transform):
        self.labels = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_path, self.labels.loc[idx][0]+'.png')
        image = Image.open(img_name, 'r')
        label = self.labels.loc[idx][1]

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'labels': label}

def generate_dataset(path, input_size, batch_size, num_workers, inc_contrast=1, **kwargs):

    print('generating APTOS dataset ===>')
    train_label_path = os.path.join(path, 'train.csv')

    test_label_path = os.path.join(path, 'sample_submission.csv')
    train_path = os.path.join(path, 'train_images')
    test_path = os.path.join(path, 'test_images')

    assert os.path.exists(train_path), train_path + ' not found'
    assert os.path.exists(test_path), test_path + ' not found'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_transform_list = [AdjustContrast(inc_contrast),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]

    test_transform_list = [AdjustContrast(inc_contrast),
            transforms.Resize(int(input_size / 0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize]

    train_loader = data.DataLoader(
            APTOSDataset(
                train_label_path, train_path, transforms.Compose(train_transform_list)),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

    val_loader = data.DataLoader(
            APTOSDataset(
                test_label_path, test_path, transforms.Compose(test_transform_list)),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    n_class = 5

    return train_loader, val_loader, n_class

