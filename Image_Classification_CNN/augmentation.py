import torch
from torchvision import transforms
import numpy as np
import random
from dataset import ImagesDataset
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

def augment_image(img_tensor: torch.Tensor, index: int) -> (torch.Tensor, str):
    v = index % 7
    if img_tensor.dtype == torch.uint8:
        img_tensor = img_tensor.float() / 255.0

    if v == 0:
        return img_tensor, "Original"

    transformation_dict = {
        1: (transforms.GaussianBlur(kernel_size=5), "GaussianBlur"),
        2: (transforms.RandomRotation(degrees=30), "RandomRotation"),
        3: (transforms.RandomVerticalFlip(p=1.0), "RandomVerticalFlip"),
        4: (transforms.RandomHorizontalFlip(p=1.0), "RandomHorizontalFlip"),
        5: (transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), "ColorJitter")
    }

    if v in transformation_dict:
        transform, transform_name = transformation_dict[v]
        transformed_img = transform(img_tensor)
        return transformed_img, transform_name

    if v == 6:
        transform_list = random.sample(list(transformation_dict.values()), 3)
        composed_transform = transforms.Compose([t[0] for t in transform_list])
        transformed_img = composed_transform(img_tensor)
        return transformed_img, "Compose"

    return img_tensor, "Unknown"

class TransformedImagesDataset(Dataset):
    def __init__(self, data_set: Dataset):
        self.data_set = data_set

    def __getitem__(self, index: int):
        original_index = index // 7
        img_np, class_id, class_name, img_path = self.data_set[original_index]
        trans_img, trans_name = augment_image(img_np, index)

        return trans_img, class_id

    def __len__(self):
        return len(self.data_set) * 7