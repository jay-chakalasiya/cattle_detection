# Cattle Detection

> This repositiry provides profides code for different CV tasks n Cattle dataset, these tasks are 1) `Object Detection`, 2) `Object Segmentation`, 3) `Cattle Key-point detection(in progress)`


## Directory Sturcture 

```
Root
|
|-------- config.py
|-------- utils.py
|-------- data_prep.py
|-------- detection_models.py
|-------- key_point_detetcion.py
|-------- Data Exploration.ipynb
|-------- Models.ipynb
|-------- NWAFU_CattleDataset
|         |--------- images
|                    |-------------- im0001.jpg
|                    |-------------- im0002.jpg
|                    |
|         |--------- annotations
|                    |-------------- im0001.txt
|                    |-------------- im0002.txt
|                    |
```

#### config.py
- contains various global configurations -> it is the entry point to rerun the code
- Before running on your system please change variable `ROOT`


#### utils.py
- This file contains all supporting funtions for this code, further description will be inside the file


#### data_prep.py
- this file creates data binaries, so that data can be accessed faster during runtime, it also resizes image 


#### key_point_detection.py (in progress)
- This file will have NN training furntions to achieve multi-instance key-point detetcion on Cattle images


#### Data Exploration.ipynb
- This notebook explores the dataset


#### Models.ipynb
- This notebook illustrates the object detetcion and segmentation models


## How to Run
- install dependencies
```
pip install -r requirements.txt
```
- install `pytorch` (preferably with GPU access)
- set config file, only changing `ROOT` variable is suffice
- Download the dataset and structure it according to directory structure presented before
