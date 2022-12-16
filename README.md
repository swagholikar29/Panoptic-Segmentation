# Optimizing Panoptic Segmentation for LiDAR Cloud data 

### *Fall 2022: Deep Learning - [Worcester Polytechnic Institute](https://www.wpi.edu/)*
Master's in Robotics Engineering

#### Members: [Deepak Harshal Nagle](https://github.com/deepaknagle), [Anuj Pai Raikar](https://github.com/22by7-raikar), [Soham Aserkar](https://github.com/ssaserkar), [Swapneel Wagholikar](https://github.com/swagholikar29)

### [Link to Report](https://github.com/22by7-raikar/Panoptic_Segmentation/blob/main/CS541_Deep_Learning_Report.pdf)

--------------------------------------------------------------

### Dataset:

1. Use the following links to download the [Velodyne sensor data](https://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip) (LiDAR Point Cloud SemanticKITTI) and the [Label data folders](http://www.semantic-kitti.org/assets/data_odometry_labels.zip). 
2. Place them in the `dataset` folder in the form mentioned as per: [Semantic KITTI website](http://www.semantic-kitti.org/dataset.html#overview). The path of the dataset is required as an argument to run the command during execution.

### Requirements:

[Tensorflow](https://www.tensorflow.org/install), Numpy, Matplotlib, GPU drivers, CUDA Toolkit, Pillow

--------------------------------------------------------------

## Steps required to execute the code:

Go to the parent folder of this repo, that is, [panoptic_segmentation](.) and enter the command:
  ```
  python3 scripts/main.py -d **path_to_the_dataset_folder**
  ```
----------------------
## Results:
![Results](https://user-images.githubusercontent.com/116770046/208182395-3564786a-27b2-4e2d-b430-6ae313794ee9.jpeg)



## References
1. Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick. [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf). Facebook AI Research (FAIR) (2018)
2. Milioto, Andres, Vizzo, Ignacio, Behley, Jens, Stachniss, Cyrill. [RangeNet ++: Fast and Accurate LiDAR Semantic Segmentation](http://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf). IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (2019)
3. Yuwen Xiong, Renjie Liao, Hengshuang Zhao, Rui Hu, Min Bai, Ersin Yumer, Raquel Urtasun. [UPSNet: A Unified Panoptic Segmentation Network](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xiong_UPSNet_A_Unified_Panoptic_Segmentation_Network_CVPR_2019_paper.pdf). IEEE/CVF Conference (2019)

