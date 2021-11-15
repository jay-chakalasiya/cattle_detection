import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from config import IMAGE_PATH, ANNOTATION_PATH, PLOT_COLORS

def format_image_id(n):
    """
    To reformat image name to 4 digit format
    1 -> im0001
    """
    return 'im'+'0'*(4-len(str(n)))+str(n)



def parse_annotations(annotation_path, all_points=True):
    """
    Takes annotation filename as input -> im0001.txt
    Outputs array or points in format [x, y, v]
    Where v is visibility of point -> -1 for head orientation, 1 for visible points, 0 for invisible points
    """
    raw_data = open(annotation_path, 'r').readlines()
    data = []
    for raw in raw_data:
        data.append([float(entry) for entry in raw.strip('\n').split(' ')])
    data[0].append(1)
    data[1].append(1)
    
    
    if all_points:
        return np.array(data)
    else:
        return np.array(data[2:]) # not sure what first two points are



def get_box(annotation_data):
    """
    In form of [x1, y1, x2, y2] -> as required by keypoint RCNN
    0 <= x1 < x2 <= W and 0 <= y1 < y2 < H
    """
    return np.array([np.min(annotation_data[:, 0]), 
                     np.min(annotation_data[:, 1]), 
                     np.max(annotation_data[:, 0]),
                     np.max(annotation_data[:, 1])])

def expand_box(box, h, w, margin):
    return np.array([max(0, box[0]-margin), 
                     max(0, box[1]-margin), 
                     min(w, box[2]+margin), 
                     min(h, box[3]+margin)], dtype=np.uint16)

def choose_posture_line_color(vsibility1, visibility2):
    if vsibility1 == 0 and visibility2 == 0:
        return 0 
    else:
        return 1

def plot_posture(annots):
    line_pairs = [[0, 1], [1, 2], [1, 3], [1, 6], [2, 3], [2, 6], [3, 6], [3, 4], [4, 5], 
                  [6, 7], [7, 8], [2, 9], [3, 10], [6, 13], [9, 10], [9, 13], [10, 13], 
                  [10, 11], [11, 12], [13, 14], [14, 15]]
    for p1, p2 in line_pairs:
        plt.plot([annots[p1][0], annots[p2][0]], 
                [annots[p1][1], annots[p2][1]], 
                color = PLOT_COLORS[choose_posture_line_color(annots[p1][2], annots[p2][2])])


def plot_image(image_id, plot_points = False, plot_box=False, plot_labels=False, plot_lines=False):
    i_path = os.path.join(IMAGE_PATH, image_id+'.jpg')
    a_path = os.path.join(ANNOTATION_PATH, image_id+'.txt')
    annotation_data = parse_annotations(a_path)
    

    im = Image.open(i_path)
    
    plt.figure(figsize=(12, 19))
    
    # Plot points
    if plot_points:
        for i, annot in enumerate(annotation_data):
            plt.plot(annot[0], annot[1], color=PLOT_COLORS[annot[2]], marker='o')
            if plot_labels:
                plt.text(annot[0], annot[1], str(i), color='red', fontsize=12)

    # plot posture
    if plot_lines:
        plot_posture(annotation_data[2:])
        
    # plot box
    if plot_box:
        box_coords = get_box(annotation_data)
        x_min, y_min, x_max, y_max = box_coords
        plt.plot([x_min, x_max], [y_min, y_min], color='yellow')
        plt.plot([x_max, x_max], [y_min, y_max], color='yellow')
        plt.plot([x_max, x_min], [y_max, y_max], color='yellow')
        plt.plot([x_min, x_min], [y_min, y_max], color='yellow')
    
    # plot image
    plt.imshow(im)
    plt.show()




#### RELATED TO KEY-POINT DETECTION

