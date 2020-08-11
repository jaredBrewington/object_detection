import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from PIL import Image


class OpenImagesDataset(data.Dataset):
    def __init__(self, root_dir, transforms=None):
        """
        Args:
            root_dir (str): Directory containing training data
            transforms (optional): Augmentation performed on the data. Defaults to None.
        """

        self.root_dir = os.path.abspath(root_dir)
        self.transforms = transforms

        self.imgs = sorted(
            os.listdir(os.path.join(self.root_dir, "train_00_part"))
        )
        self.img_ids = [img.split(".")[0] for img in self.imgs]

        self.masks = [
            [
                mask
                for mask in os.listdir(
                    os.path.join(self.root_dir, "train-masks-f")
                )
                if img_id in mask
            ]
            for img_id in self.img_ids
        ]

        id_to_name_path = "../input/preprocessed-excerpt-from-open-images-2020/class_id_to_names.json"
        with open(id_to_name_path) as f:
            id_to_name = json.load(f)
        self.class_id_to_name = {int(k): v for k, v in id_to_name.items()}

        name_to_id_path = "../input/preprocessed-excerpt-from-open-images-2020/class_name_to_id.json"
        with open(name_to_id_path) as f:
            name_to_id = json.load(f)
        self.class_name_to_id = {k: int(v) for k, v in name_to_id.items()}

        bbox_path = "../input/preprocessed-excerpt-from-open-images-2020/processed_train_anno_bbox.csv"
        bbox_df = pd.read_csv(bbox_path, index_col="ImageID")
        self.bbox_df = bbox_df

    def __getitem__(self, idx):

        # load image and associated masks
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        img_id = self.img_ids[idx]
        mask_path_list = [
            os.path.join(self.root_dir, "masks", mask)
            for mask in self.masks[idx]
        ]

        img = np.array(Image.open(img_path))

        mask_list = [
            np.array(Image.open(mask_path)) for mask_path in mask_path_list
        ]

        # get bounding boxes and class labels with metadata for the image
        bboxes_series = self.bbox_df.loc[img_id]

        # get the label index id
        labels = bboxes_series["class_id"].to_numpy()
        # bboxes should be list of [Xmin, Ymin, Xmax, Ymax] for each bbox
        bboxes = bboxes_series["bbox"].to_numpy()

        # make everything torch tensors
        img = torch.as_tensor(img)
        mask_list = torch.as_tensor(mask_list, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        bboxes = torch.as_tensor(bboxes)

        # output a dictionary with all the info
        target = {}

        target["bboxes"] = bboxes
        target["labels"] = labels
        target["masks"] = mask_list
        target["index"] = torch.tensor([idx], dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
