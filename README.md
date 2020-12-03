# Detecting Defects in Steel Using Tensorflow

## Data
### Source
The dataset that I used in this project is the Northeastern University (NEU) surface defect database. http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

I am grateful to Drs. Song and Yan for making this dataset available. Scroll to the bottom of this page to see examples of their work.

### Organization
This database consists of 1,800 square images of hot-rolled steel.
There are six defect categories:

1. rolled-in scale
2. patches
3. crazing
4. pitted surface
5. inclusion
6. scratches

Note that there are no images of steel *without defects* in this database. For the purposes of this demo, the model will only be trained to differentiate between defects.

### How do I use it?
I did not want to host these images, as I do not own them. If you want to run the code yourself, download the NEU-CLS dataset here:
https://drive.google.com/open?id=1NGlXT9sIaQpyxUoT6MLKm1Pr6x8oxOvc

I used the classification version of the database (CLS), but there is a version for detection as well that has detailed annotations marking defect locations. I might update this repo later to do something with the detection.

The dataset will be zipped. Follow these instructions after downloading:

1. Unzip the archive.
2. In the IMAGES folder, create a folder for each of the six classes. You can give them any name you want. I just named them IMAGES/CRAZING, IMAGES/PATCHES, etc. based on the defect type.
3. Move all images into their corresponding sub-folders. There are 300 of each type. So you should end up with 300 in the crazing folder, 300 in the patches folder, and so on.
4. Set an environment variable called NEU_CLS_DIR pointing to the path of the IMAGES directory.

Now when you clone this repo and run main.py, it should be able to find the path without needing to do anything further.

## Related research
K. Song and Y. Yan, “A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects,” Applied Surface Science, vol. 285, pp. 858-864, Nov. 2013.

Yu He, Kechen Song, Qinggang Meng, Yunhui Yan, “An End-to-end Steel Surface Defect Detection Approach via Fusing Multiple Hierarchical Features,” IEEE Transactions on Instrumentation and Measuremente,  2020,69(4),1493-1504.

Hongwen Dong, Kechen Song, Yu He, Jing Xu, Yunhui Yan, Qinggang Meng, “PGA-Net: Pyramid Feature Fusion and Global Context Attention Network for Automated Surface Defect Detection,” IEEE Transactions on Industrial Informatics,  2020.
