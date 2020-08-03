from dataset import Open_Images_Instance_Seg
from visualize import add_bboxes
import matplotlib.pyplot as plt

train_data = Open_Images_Instance_Seg("./train/")

img, target = train_data.__get_item__(0)

print(target["bboxes"][0])

# plt.imshow(img)

img_w_boxes = add_bboxes(img, target, labels_df=train_data.classes)
plt.imshow(img_w_boxes)
plt.show()
