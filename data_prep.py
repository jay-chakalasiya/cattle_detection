import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm

from utils import format_image_id, parse_annotations, get_box, expand_box
from config import ANNOTATION_PATH, IMAGE_PATH, DATA_PREP_PATH, DATA_PREP_FILES
from config import MAX_IMAGE_ID, MISSING_IMAGES, NO_OF_IMAGES
from config import KEY_POINT_LABELS
from config import IMAGE_SIZE, CROPPED_IMAGE_SIZE, BOX_MARGIN




def main():

    if not os.path.exists:
        try:
            os.mkdir(DATA_PREP_PATH)
            print('Data Directory creates ---> Data will be stored at {}'.format(DATA_PREP_PATH))
        except:
            print('Please rerun with Admin access')
            return
    
    h_scale = CROPPED_IMAGE_SIZE[0]/IMAGE_SIZE[0]
    w_scale = CROPPED_IMAGE_SIZE[1]/IMAGE_SIZE[1]

    images = np.zeros((NO_OF_IMAGES, 3, CROPPED_IMAGE_SIZE[0], CROPPED_IMAGE_SIZE[1]), dtype=np.uint8)
    labels = np.zeros((NO_OF_IMAGES, 1, len(KEY_POINT_LABELS), 3))
    cropped_images = []

    index = 0
    for i in tqdm(range(1, MAX_IMAGE_ID+1)):
        if i not in MISSING_IMAGES:

            # Get Images
            im = cv2.imread(os.path.join(IMAGE_PATH, format_image_id(i)+'.jpg'))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (CROPPED_IMAGE_SIZE[1], CROPPED_IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)


            # Get box
            annot = parse_annotations(os.path.join(ANNOTATION_PATH, format_image_id(i)+'.txt'))
            annot[:, 0] = annot[:, 0]*w_scale
            annot[:, 1] = annot[:, 1]*h_scale
            box = expand_box(get_box(annot), CROPPED_IMAGE_SIZE[0], CROPPED_IMAGE_SIZE[1], BOX_MARGIN)

            # Crop image
            cropped_im = im[box[1]:box[3]+1, box[0]:box[2]+1, :]
            

            # Transform key-point coords to new location
            tr_annot = annot.copy()
            tr_annot[:, 0] = tr_annot[:, 0] - box[0]
            tr_annot[:, 1] = tr_annot[:, 1] - box[1]
            labels[index][0] = tr_annot[2:]

            im = np.moveaxis(im, -1, 0)
            images[index] = im
            cropped_im = np.moveaxis(cropped_im, -1, 0)
            cropped_images.append(cropped_im)

            
            # Get Labels
            index+=1


    print('Data Extracted ----> Now saving, (Might take a few minutes)')
    f = open(os.path.join(DATA_PREP_PATH, DATA_PREP_FILES['resized_full_images']), 'wb')
    np.save(f, images)
    f.close()

    f = open(os.path.join(DATA_PREP_PATH, DATA_PREP_FILES['key_point_data']), 'wb')
    pickle.dump({'images': cropped_images, 'labels': labels}, f)
    f.close()

    print('Completed')
    return


if __name__ == "__main__":
    main()