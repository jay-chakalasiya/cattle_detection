import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToPILImage, ToTensor, Compose
from torchvision.utils import draw_bounding_boxes

from config import DATA_PREP_PATH, DATA_PREP_FILES, DATA_PREP_PATH
from config import SCORE_THRESHOILD, COW_LABEL_ID

class CattleDetection:
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.float().eval()
        self.tensor_transform =  Compose([ToTensor()])
        self.loaded = False
        self.resized_images = None


    def infer(self, img_path, plot=True):
        #read
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, C = image.shape

        #resize and normalize
        image = cv2.resize(image, (W//5, H//5), interpolation=cv2.INTER_AREA)
        X = [self.tensor_transform(image)]
        detected_boxes = self.detect_boxes(X)

        bounding_box_image = draw_bounding_boxes(torch.tensor(np.moveaxis(image, -1, 0), dtype=torch.uint8), detected_boxes, colors='red').numpy()
        if plot:
            plt.imshow(np.moveaxis(bounding_box_image, 0, -1))
        return detected_boxes.numpy()


    def detect_boxes(self, image_tensor):
        prediction = self.model(image_tensor)

        boxes = prediction[0]['boxes'].detach()
        scores = prediction[0]['scores'].detach()
        labels = prediction[0]['labels'].detach()

        seleted_boxes = boxes[labels==COW_LABEL_ID][scores[labels==COW_LABEL_ID]>SCORE_THRESHOILD]
        return seleted_boxes


    def infer_from_binaries(self, index):
        if not self.loaded:
            f = open(os.path.join(DATA_PREP_PATH, DATA_PREP_FILES['resized_full_images']), 'rb')
            self.resized_images = np.load(f)
            self.loaded = True
        index = index % len(self.resized_images)
        return self.detect_boxes([torch.tensor(self.resized_images[index]/255, dtype=torch.float)]).numpy()

        
class CattleMasking:
    def __init__(self):
        return