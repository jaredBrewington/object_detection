import torch
import torch.utils.data as data
from dataset import OpenImagesDataset
from utils import collate_fn
import albumentations as A

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate


def get_model(model_name):

    if model_name == "MaskRCNN":

        num_classes = 600

        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True
        )

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

    else:
        print("Model name not recognized.")
        end(1)

    return model


def train(num_epochs=1, batch_size=16):
    """
    Args:
        num_epochs (int, optional): Defaults to 1.
        batch_size (int, optional): Batch size of data loader. Defaults to 16.
    """
    print("initiating training...")

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # data augmentation for training set
    train_aug = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15
            ),
            A.HorizontalFlip(),
            A.Normalize(always_apply=True),
        ]
    )

    # get training dataset
    dataset = OpenImagesDataset(
        "./train/", transforms=None, preprocessed_dir="/Users/jared/Downloads",
    )
    dataset_valid = OpenImagesDataset(
        "./train/", transforms=None, preprocessed_dir="/Users/jared/Downloads"
    )

    # split the dataset in train and validation set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_valid = torch.utils.data.Subset(dataset_valid, indices[-50:])

    # make data loader for training and validation data
    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    data_loader_valid = data.DataLoader(
        dataset_valid,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model_name = "MaskRCNN"
    model = get_model(model_name)

    # move model to the correct device
    model.to(device)

    # make optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.2, weight_decay=0.0005
    )

    # give learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1
    )

    for epoch in range(num_epochs):
        print("Starting epoch {}...".format(epoch))
        # train model
        train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=10
        )

        # update the learning rate
        lr_scheduler.step()

        # test on validation set
        evaluate(model, data_loader_test, device=device)

    print("\n\nDone!\n")

