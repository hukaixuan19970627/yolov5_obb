# INSTAllation 
## Requirements
* Linux **(Recommend)**, Windows **(not Recommend, Please refer to this [issue](https://github.com/hukaixuan19970627/yolov5_obb/issues/224) if you have difficulty in generating utils/nms_rotated_ext.cpython-XX-XX-XX-XX.so)**
* Python 3.7+ 
* PyTorch ≥ 1.7 
* CUDA 9.0 or higher

I have tested the following versions of OS and softwares：
* OS：Ubuntu 16.04/18.04
* CUDA: 10.0/10.1/10.2/11.3

## Install 
**CUDA Driver Version ≥ CUDA Toolkit Version(runtime version) = torch.version.cuda**

a. Create a conda virtual environment and activate it, e.g.,
```
conda create -n Py39_Torch1.10_cu11.3 python=3.9 -y 
source activate Py39_Torch1.10_cu11.3
```
b. Make sure your CUDA runtime api version ≤ CUDA driver version. (for example 11.3 ≤ 11.4)
```
nvcc -V
nvidia-smi
```
c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), Make sure cudatoolkit version same as CUDA runtime api version, e.g.,
```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
nvcc -V
python
>>> import torch
>>> torch.version.cuda
>>> exit()
```
d. Clone the yolov5-obb repository.
```
git clone https://github.com/hukaixuan19970627/yolov5_obb.git
cd yolov5_obb
```
e. Install yolov5-obb.

```python 
pip install -r requirements.txt
cd utils/nms_rotated
python setup.py develop  #or "pip install -v -e ."
```

## Install DOTA_devkit. 
**(Custom Install, it's just a tool to split the high resolution image and evaluation the obb)**

```
cd yolov5_obb/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

