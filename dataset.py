import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor
from PIL import Image
from pdb import set_trace

class RoboticsDataset(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]

        if self.mode == 'train':
            img_file_name_split = img_file_name.split(",")
            img_file_name_path = img_file_name_split[0] + "slide_tiles/" + img_file_name_split[1] + ".jpg"
            image = cv2.imread(str(img_file_name_path))
            mask = load_mask(img_file_name_path, self.problem_type)      
            data = {"image": image, "mask": mask}
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]
            if self.problem_type == 'binary':
                return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                return img_to_tensor(image), torch.from_numpy(mask).long()
        else:
            img_file_name_split = img_file_name.split(",")
            img_file_name_path = img_file_name_split[0] + img_file_name_split[1] + ".jpg"
            image = cv2.imread(str(img_file_name_path))
            data = {"image": image}
            augmented = self.transform(**data)
            image = augmented["image"]
            return img_to_tensor(image), str(img_file_name_split[1])


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path, problem_type):
    mask_name_path = str(path).replace('slide_tiles', 'label_tiles').replace('jpg', 'png')
    img = Image.open(mask_name_path).getdata()
    img_width, img_height = img.size
    
    np_img = np.array(img, np.uint8).reshape((img_height, img_width))
    
    return np_img
