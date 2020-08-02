import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from PIL import Image

class Open_Images_Instance_Seg(data.Dataset):

    def __init__(self, root_dir, transforms=None):

        self.root_dir = os.path.abspath(root_dir)
        self.transforms = transforms

        self.imgs = sorted(os.listdir(os.path.join(self.root_dir, 'images')))
        self.img_ids = [img.split('.')[0] for img in self.imgs]

        self.masks = [[mask for mask in os.listdir(os.path.join(self.root_dir, 'masks')) if img_id in mask] for img_id in self.img_ids]

        self.classes = pd.read_csv(os.path.join(self.root_dir, 'class-descriptions-boxable.csv'), names=['id', 'name'])

        bbox_path = os.path.join(self.root_dir, 'oidv6-train-annotations-bbox.csv')
        bbox_df = pd.read_csv(os.path.join(self.root_dir, bbox_path))
        self.bbox_df = bbox_df

    def __get_item__(self, idx):

        # load image and associated masks
        img_path = os.path.join(self.root_dir, 'images', self.imgs[idx])
        img_id = self.img_ids[idx]
        mask_path_list = [ os.path.join(self.root_dir, 'masks', mask) for mask in self.masks[idx] ]

        img = np.array(Image.open(img_path))
        
        mask_list = [np.array(Image.open(mask_path)) for mask_path in mask_path_list]

        # get bounding boxes and class labels for the image
        bboxes_df = self.bbox_df[self.bbox_df['ImageID']==img_id]
        #scale bounding boxes to image size
        height, width = img.shape[:2]
        bboxes_df['XMin'] = bboxes_df['XMin'] * width
        bboxes_df['XMax'] = bboxes_df['XMax'] * width
        bboxes_df['YMin'] = bboxes_df['YMin'] * height
        bboxes_df['YMax'] = bboxes_df['YMax'] * height
        # get the label index id
        labels = bboxes_df['LabelName'].to_numpy()
        labels = np.array([self.classes.index[self.classes['id']==label] for label in labels]).flatten()
        bboxes = bboxes_df[['XMin','YMin','XMax','YMax']].to_numpy()
        
        # make everything torch tensors
        img = torch.as_tensor(img)
        mask_list = torch.as_tensor(mask_list)
        labels = torch.as_tensor(labels)
        bboxes = torch.as_tensor(bboxes)

        # output a dictionary with all the info
        target = {}

        target['bboxes'] = bboxes
        target['labels'] = labels
        target['masks'] = mask_list
        target['image_id'] = img_id
        target['index'] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)