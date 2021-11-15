import os

# Data path dependencies
ROOT = os.path.join('D:\\', 'projects', 'cattle_detection')
DATA_PATH = os.path.join(ROOT , 'NWAFU_CattleDataset')
IMAGE_PATH = os.path.join(DATA_PATH, 'images')
ANNOTATION_PATH = os.path.join(DATA_PATH, 'annotations')
DATA_PREP_PATH = os.path.join(ROOT, 'data')
DATA_PREP_FILES = {'resized_full_images': 'resized_images.npy', 'key_point_data': 'key_point_data.pkl'}

IMAGE_SIZE = (1080, 1920)
CROPPED_IMAGE_SIZE = (216, 384)
BOX_MARGIN = 5


# Realted to object detection
COW_LABEL_ID = 21
SCORE_THRESHOILD = 0.7


# Related to Annotations
KEY_POINT_LABELS_ALL = ['head_top', 'head_bottom', 'face', 'neck', 'upper_back', 
                        'front_right_leg_top', 
                        'front_right_leg_center','front_right_leg_bottom', 
                        'front_left_leg_top', 'front_left_leg_center','front_left_leg_bottom', 
                        'lower_back', 
                        'rear_right_leg_top', 'rear_right_leg_center','rear_right_leg_bottom',
                        'rear_left_leg_top', 'rear_left_leg_center','rear_left_leg_bottom']
#'head_top', 'head_bottom', -> the first two points, although not sure what they are 
KEY_POINT_LABELS = KEY_POINT_LABELS_ALL[2:]
INDEX_TO_LABEL = {l:i for i, l in enumerate(KEY_POINT_LABELS)}



# Related to raw datafiles
MAX_IMAGE_ID = 2213
MISSING_IMAGES = [107, 137, 147, 205, 222, 275, 337, 363, 369, 370, 372, 448, 456, \
                460, 479, 490, 495, 516, 560, 575, 597, 643, 676, 703, 749, 770, 781, \
                854, 876, 904, 929, 963, 984, 989, 1014, 1041, 1066, 1167, 1282, 1289, \
                1347, 1379, 1406, 1410, 1431, 1432, 1448, 1449, 1460, 1468, 1504, 1512, \
                1516, 1519, 1532, 1550, 1564, 1592, 1594, 1612, 1639, 1662, 1673, 1696, \
                1706, 1710, 1735, 1746, 1750, 1753, 1758, 1776, 1777, 1793, 1844, 1855, \
                1864, 1900, 1937]
NO_OF_IMAGES = MAX_IMAGE_ID - len(MISSING_IMAGES)



# Plot Colros
PLOT_COLORS = {-1: 'red', 0: 'blue', 1: 'red'}
MASK_COLORS = ['red', 'blue', 'green', 'yellowgreen', 'turquoise', 'gold', 'coral', 'purple',
               'teal', 'chartreuse', 'beige', 'cyan', 'ivory', 'orange', 'orchid', 'pink']