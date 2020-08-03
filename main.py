from dataset import Open_Images_Instance_Seg
from visualize import add_bboxes
import matplotlib.pyplot as plt

import torch.utils.data as data
import utils

# load dataset
train_data = Open_Images_Instance_Seg("./train/")

# make data loader
data_loader = data.DataLoader(
    train_data,
    batch_size=20,
    shuffle=True,
    num_workers=0,
    collate_fn=utils.collate_fn,
)

# train model
