  
  
# Updates
**[2022/1/7]**
1. Update yolov5 base version to [Releases v6.0](https://github.com/ultralytics/yolov5/releases/tag/v6.0).
2. Rebuild the obb-label pre/post-process code. That means Faster in training/validation/testing. 

Model| Dataset  | BatchSize | GPU | Time Cost |
----   | -----                              | ------ | ----- | ------ |
yolov5m| DOTAv1.5_subsize1024_gap200_rate1.0|75      |3090Ti |15h     | 

3. Some Bugs Fixed.

|Bug | Fixed | Describe 
|----                                |------ | ------  
|Don't support validation            | ✔     | Support hbb validation in training, which is faster than obb validation|
|Don't support single class training | ✔     | - 
|Image must meets Height=Width       | ✔     | - 

4. support obb_nms gpu version.
