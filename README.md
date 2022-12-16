# Optimizing Panoptic Segmentation for LiDAR Cloud data 

## *CS541: Deep Learning - [Worcester Polytechnic Institute](https://www.wpi.edu/), Fall 2022*

#### Members: [Deepak Harshal Nagle](https://github.com/deepaknagle), [Anuj Pai Raikar](https://github.com/22by7-raikar), [Soham Aserkar](https://github.com/ssaserkar), [Swapneel Wagholikar](https://github.com/swagholikar29)

Master's in Robotics Engineering

#### [Link to Report](./final_report.pdf)

--------------------------------------------------------------

## Requirements:

1. [Tensorflow](https://www.tensorflow.org/install) 

2. Numpy, Matplotlib

3. GPU drivers

4. CUDA Toolkit

5. Pillow

--------------------------------------------------------------

## SemanticKITTI dataset

Use the following links to download the Velodyne sensor data and the Label data folders. Place them in the `dataset` folder in the form mentioned as per: [Semantic KITTI website](http://www.semantic-kitti.org/dataset.html#overview).

1. [Download](http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip) 3-D Point Cloud Data

2. [Download](http://www.semantic-kitti.org/assets/data_odometry_labels.zip) Label Data

The path of this dataset is required as an argument to run the command.

--------------------------------------------------------------

## Steps required to execute the code:

Go to the parent folder of this repo, that is, [panoptic_segmentation](.) and enter the command:
  ```
  python3 scripts/main.py -d **path_to_the_dataset_folder**
  ```

----------------------
## References
1. Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick. [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf). Facebook AI Research (FAIR) (2018)
