python -m torch.distributed.launch --nproc_per_node 3 train.py \
    --batch 32 \
    --data '/home/test/Persons/hukaixuan/yolov5/datasets/package/data.yaml' \
    --weights 'runs/train/exp_multi_aug_yolov5n/weights/last.pt' \
    --hyp 'data/hyps/hyp.finetune_package.yaml' \
    --device 2 \
    --epochs 50 \
    --img 512 \
    --cache \
    --sync-bn


tensorboard --logdir runs/train/exp

# 测试 train/val/test 数据集的情况
python val.py \
    --weights runs/train/exp_close_mosaic/weights/last.pt \
    --img 480 \
    --half \
    --batch 1 \
    --conf 0.001 --iou 0.65 \ # mAP  测精度的标准
    --conf 0.25 --iou 0.45 \ # speed 测速标准
    --task 'val' or 'train' or 'test'

# speed模式
python val.py \
    --weights runs/train/exp_close_mosaic/weights/best.pt runs/train/exp_close_mosaic/weights/last.pt \
    --img 480 \
    --half \
    --task 'speed' \
    --device 'cpu' \
    --batch 1

#mAP
python val.py --task 'test' --batch 16 --save-json --name 'yolov5t_dotav1_test_split'
python tools/TestJson2VocClassTxt.py \
    --json_path 'runs/val/yolov5t_DroneVehicle_val/best_obb_predictions.json' \
    --save_path 'runs/val/yolov5t_DroneVehicle_val/splited_obb_prediction_Txt'
python DOTA_devkit/ResultMerge_multi_process.py \
    --scrpath 'runs/val/yolov5t_dotav1_test_split/splited_obb_prediction_Txt' \
    --dstpath 'runs/val/yolov5t_dotav1_test_split/Merged_obb_prediction_Txt'
python DOTA_devkit/results_obb2hbb.py \
    --srcpath 'runs/val/yolov5t_dotav1_test_split/Merged_obb_prediction_Txt' \
    --dstpath 'runs/val/yolov5t_dotav1_test_split/Merged_hbb_prediction_Txt'
