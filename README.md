# Cattle Detection

> This repositiry provides profides code for different CV tasks n Cattle dataset, these tasks are 1) Object Detection, 2) Object Segmentation, 3) Cattle Key-point detection(in progress)


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
- Before running on your system please change `ROOT`