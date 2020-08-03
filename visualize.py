import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np


def add_bboxes(img, target, labels_df=None):
    new_img = img.numpy()
    bboxes = target["bboxes"].numpy()
    labels = target["labels"].numpy()
    # add each box
    for bbox, label in zip(bboxes, labels):
        c = tuple([int(x) for x in np.random.randint(0, high=255, size=3)])
        min_point = (int(bbox[0]), int(bbox[1]))
        max_point = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(new_img, min_point, max_point, color=c, thickness=2)

        if labels_df is not None:
            label = labels_df["name"].loc[int(label)]

            # annotate box
            text_pos = (int(bbox[0]) + 2, int(bbox[3]) - 3)
            cv2.putText(
                new_img,
                label,
                text_pos,
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=c,
            )

    return new_img


# def add_masks(image, masks):

