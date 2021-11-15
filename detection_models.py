import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import ToPILImage, ToTensor, Compose
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


from config import DATA_PREP_PATH, DATA_PREP_FILES, DATA_PREP_PATH
from config import SCORE_THRESHOILD, COW_LABEL_ID, MASK_COLORS

class CattleDetection:
    def __init__(self):
        self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        #self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
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
        detected_boxes = self.detect(X)

        if plot:
            bounding_box_image = draw_bounding_boxes(torch.tensor(np.moveaxis(image, -1, 0), dtype=torch.uint8), detected_boxes, colors='red').numpy()
            plt.imshow(np.moveaxis(bounding_box_image, 0, -1))
        return detected_boxes.numpy()


    def detect_boxes(self, image_tensor):
        prediction = self.model(image_tensor)

        boxes = prediction[0]['boxes'].detach()
        scores = prediction[0]['scores'].detach()
        labels = prediction[0]['labels'].detach()

        seleted_boxes = boxes[labels==COW_LABEL_ID][scores[labels==COW_LABEL_ID]>SCORE_THRESHOILD]
        return seleted_boxes


    def infer_from_binaries(self, index, plot = False, return_boxes=True):
        if not self.loaded:
            f = open(os.path.join(DATA_PREP_PATH, DATA_PREP_FILES['resized_full_images']), 'rb')
            self.resized_images = np.load(f)
            self.loaded = True
        index = index % len(self.resized_images)
        detected_boxes = self.detect_boxes([torch.tensor(self.resized_images[index]/255, dtype=torch.float)])
        if plot:
            bounding_box_image = draw_bounding_boxes(torch.tensor(self.resized_images[index], dtype=torch.uint8), detected_boxes, colors='red').numpy()
            plt.imshow(np.moveaxis(bounding_box_image, 0, -1))
        if return_boxes:
            return detected_boxes.numpy()

    


class CattleSegmentation:
    def __init__(self):
        self.loaded = False
        self.resized_images = None
        self.model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
        self.model.float().eval()
        self.cow_class_id =10  # it is id the model originally trained on

    def show(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    def infer_from_binaries(self, index, plot=True):
        if not self.loaded:
            f = open(os.path.join(DATA_PREP_PATH, DATA_PREP_FILES['resized_full_images']), 'rb')
            self.resized_images = np.load(f)
            self.loaded = True
        index = index % len(self.resized_images)
        image = self.resized_images[index]/255
        image_tf = torch.tensor(np.array([image]), dtype=torch.float)
        X = F.normalize(image_tf, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        masks = self.model(X)['out']
        
        normalized_masks = torch.nn.functional.softmax(masks, dim=1)
        #print(normalized_masks.shape, normalized_masks.min().item(), normalized_masks.max().item())
        cow_masks = normalized_masks[0][self.cow_class_id]
        cow_masks = torch.tensor(cow_masks>0.4, dtype=torch.float)
        if plot:
            masked_image = draw_segmentation_masks(torch.tensor(self.resized_images[index], dtype=torch.uint8), 
                                                    masks=cow_masks>0.4, alpha=0.5, 
                                                    colors = MASK_COLORS[:len(cow_masks)]).numpy()
            plt.imshow(np.moveaxis(masked_image, 0, -1))



"""
WORK IN PROGRESS BELOW THIS
"""

class CattleAll:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.float().eval()
        self.tensor_transform =  Compose([ToTensor()])
        self.loaded = False
        self.resized_images = None

    def infer(self, img_path, plot_box=True, plot_mask=True):
        #read
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, C = image.shape

        #resize and normalize
        image = cv2.resize(image, (W//5, H//5), interpolation=cv2.INTER_AREA)
        X = [self.tensor_transform(image)]
        detected_boxes, detected_masks = self.detect(X)

        if plot_box:
            bounding_box_image = draw_bounding_boxes(torch.tensor(np.moveaxis(image, -1, 0), dtype=torch.uint8), 
                                                     detected_boxes, colors='red').numpy()
            plt.imshow(np.moveaxis(bounding_box_image, 0, -1))
        elif plot_mask:
            masked_image = draw_segmentation_masks(torch.tensor(np.moveaxis(image, -1, 0), dtype=torch.uint8), 
                                                   masks=detected_masks, alpha=0.7, 
                                                   colors = MASK_COLORS[:len(detected_masks)]).numpy()
            plt.imshow(np.moveaxis(masked_image, 0, -1))


    def detect(self, image_tensor):
        prediction = self.model(image_tensor)

        boxes = prediction[0]['boxes'].detach()
        scores = prediction[0]['scores'].detach()
        labels = prediction[0]['labels'].detach()
        masks = prediction[0]['masks'].detach()


        selected_boxes = boxes[labels==COW_LABEL_ID][scores[labels==COW_LABEL_ID]>SCORE_THRESHOILD]
        selected_masks = masks[labels==COW_LABEL_ID][scores[labels==COW_LABEL_ID]>SCORE_THRESHOILD]
        return selected_boxes, selected_masks


    def infer_from_binaries(self, index, plot_box=True, plot_mask=True):
        if not self.loaded:
            f = open(os.path.join(DATA_PREP_PATH, DATA_PREP_FILES['resized_full_images']), 'rb')
            self.resized_images = np.load(f)
            self.loaded = True
        index = index % len(self.resized_images)
        detected_boxes, detected_masks = self.detect([torch.tensor(self.resized_images[index]/255, dtype=torch.float)])
        if plot_box:
            bounding_box_image = draw_bounding_boxes(torch.tensor(self.resized_images[index], dtype=torch.uint8), 
                                                     detected_boxes, colors='red').numpy()
            plt.imshow(np.moveaxis(bounding_box_image, 0, -1))
        elif plot_mask:
            print(torch.squeeze(detected_masks, axis=1).shape)
            detected_masks = torch.tensor(torch.squeeze(detected_masks, axis=1), dtype=torch.bool)
            masked_image = draw_segmentation_masks(torch.tensor(self.resized_images[index], dtype=torch.uint8), 
                                                   masks=detected_masks, alpha=0.7, 
                                                   colors = MASK_COLORS[:len(detected_masks)]).numpy()
            plt.imshow(np.moveaxis(masked_image, 0, -1))

        return detected_boxes.numpy()