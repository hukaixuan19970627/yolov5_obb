# yolov5_for_oriented_object_detection

The Pytorch implementation is [yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb )

actually ,This repo is based on [yolov5](https://github.com/ultralytics/yolov5/tree/v6.0/)  and only used at Linux PC (partial dependency install hardly at windows PC)

## How to Run, yolov5s as example

1. generate .wts from pytorch with .pt

```
// clone code according to above #Different versions of yolov5
train and gen best.pt at runs/train/exp[]/weights/
cp gen_wts.py {yolov5_obb}
python gen_wts.py -w runs/train/exp[]/weights/best.pt -o yolov5s.wts
// a file 'yolov5s.wts' will be generated.
```

2. build **tensorr engine **

```
// update CLASS_NUM in yololayer.h 
// caution:CLASS_NUM= your classes +180(180 for angle classes)
mkdir build
cd build
cp {yolov5_obb}/yolov5s.wts ../yolov5s.wts
cmake ..
make
sudo ./yolov5_gen -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to plan file
// For example yolov5s
sudo ./yolov5_gen -s ../yolov5s.wts ../yolov5s.engine s
```

3. use **tensorr engine**


```
sudo ./yolov5_use ../../yolov5s.engine ../../images/mytest.png
```

![image-20220402142552499](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20220402142552499.png)



