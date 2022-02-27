  
  
# Updates
**[2022/1/7]**
1. Update yolov5 base version to [Releases v6.0](https://github.com/ultralytics/yolov5/releases/tag/v6.0).
2. Rebuild the obb-label pre/post-process code. That means **Faster and Stronger** in training/validation/testing. 

Model| Training Dataset  | BatchSize | epochs |GPU | Time Cost |OBB mAP<sup>test<br><sup>@0.5| fps |
----           | -----                                   | ------ | ----- | ----- | ----- | -----   | ----- |
yolov5m-old    | DOTAv1.5train_subsize1024_gap200_rate1.0|75      |300    |2080Ti |96h    |68.36    |20     |
**yolov5m-new**| DOTAv1.5train_subsize1024_gap200_rate1.0|75      |300    |2080Ti |**15h**|**73.19**|**59** |

3. Some Bugs Fixed.

|Bug | Fixed | Describe 
|----                                |------ | ------  
|Don't support validation            | ✔     | Support hbb validation in training, which is faster than obb validation|
|Don't support single class training | ✔     | But it will get weaker results than **'nc=2'** |
|Image must meets Height=Width       | ✔     | - |

4. support obb_nms gpu version.
