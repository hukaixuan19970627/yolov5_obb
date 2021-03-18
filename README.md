# YOLOv5_DOTA_OBB
YOLOv5 in DOTA_OBB dataset with CSL_label.(Oriented Object Detection)


## Datasets and pretrained checkpoint
* `Datasets` : [DOTA](https://link.zhihu.com/?target=http%3A//captain.whu.edu.cn/DOTAweb/)
* `Pretrained Checkpoint` : 

## Fuction
* `train.py`.  Train.

* `detect.py`. Detect and visualize the detection result. Get the detection result txt.

* `evaluation.py`.  Merge the detection result and visualize it. Finally evaluate the detector



## Installation
`1.` Python 3.8 or later with all requirements.txt dependencies installed, including torch>=1.7. To install run:
```
$   pip install -r requirements.txt
```
`2.` Install swig
```
$   cd  \.....\yolov5_DOTA_OBB\utils
$   sudo apt-get install swig
```
`3.` Create the c++ extension for python
```
$   swig -c++ -python polyiou.i
$   python setup.py build_ext --inplace
```



## More detailed explanation
想要了解相关实现的细节和原理可以看我的知乎文章;    
[YOLOv5_DOTAv1.5(遥感旋转目标检测，全踩坑记录)](https://zhuanlan.zhihu.com/p/357992219);    


## Usage Example
`1.` `'Get Dataset' `
 
* Split the DOTA_OBB image and labels. Trans DOTA format to YOLO longside format.

* You can refer to  [hukaixuan19970627/DOTA_devkit_YOLO](https://github.com/hukaixuan19970627/DOTA_devkit_YOLO).

* The Oriented YOLO Longside Format is:

```
$  classid    x_c   y_c   longside   shortside    Θ    Θ∈[0, 180)


* longside: The longest side of the oriented rectangle.

* shortside: The other side of the oriented rectangle.

* Θ: The angle between the longside and the x-axis(The x-axis rotates clockwise).x轴顺时针旋转遇到最长边所经过的角度
```
`WARNING: IMAGE SIZE MUST MEETS 'HEIGHT = WIDTH'`

`2.` `'train.py'` 

* All same as [ultralytics/yolov5](https://github.com/ultralytics/yolov5).

![train_batch_mosaic0](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB/tree/master/ReadMe_img/train_batch0.jpg)
![train_batch_mosaic1](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB/tree/master/ReadMe_img/train_batch1.jpg)
![train_batch_mosaic2](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB/tree/master/ReadMe_img/train_batch2.jpg)


`3.` `'detect.py'` 
    
* Download the demo files [DOTA_demo_view](). Then run the demo. Visualize the detection result and get the result txt files.

```
$  python detect.py
```
![detection_result_before_merge1](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB/tree/master/ReadMe_img/P0004__1__0___0.png)
![detection_result_before_merge2](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB/tree/master/ReadMe_img/P0004__1__0___440.png)
![draw_detection_result](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB/tree/master/ReadMe_img/P1478__1__853___824.png)



`4.` `'evaluation.py'` 

* Run the detect.py demo first. Then change the path with yours:
```
evaluation
(
        detoutput=r'/....../DOTA_demo_view/detection',
        imageset=r'/....../DOTA_demo_view/row_images',
        annopath=r'/....../DOTA_demo_view/row_DOTA_labels/{:s}.txt'
)
draw_DOTA_image
(
        imgsrcpath=r'/...../DOTA_demo_view/row_images',
        imglabelspath=r'/....../DOTA_demo_view/detection/result_txt/result_merged',
        dstpath=r'/....../DOTA_demo_view/detection/merged_drawed'
)
```

* Run the evaluation.py demo. Get the evaluation result and visualize the detection result which after merged.
```
$  python evaluation.py
```

![detection_result_after_merge](https://github.com/hukaixuan19970627/YOLOv5_DOTA_OBB/tree/master/ReadMe_img/P0004_.png)


## 有问题反馈
在使用中有任何问题，欢迎反馈给我，可以用以下联系方式跟我交流

* 知乎（@略略略）
* 代码问题提issues,其他问题请知乎上联系


## 感激
感谢以下的项目,排名不分先后

* [ultralytics/yolov5](https://github.com/ultralytics/yolov5).
* [Thinklab-SJTU/CSL_RetinaNet_Tensorflow](https://github.com/Thinklab-SJTU/CSL_RetinaNet_Tensorflow).

## 关于作者

```javascript
  Name  : "胡凯旋"
  describe myself："咸鱼一枚"
  
```
