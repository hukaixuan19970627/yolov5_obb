"""
Yolov5-obb检测结果Json 文件转Voc Class Txt
--json_path 输入的json文件路径
--save_path 输出文件夹路径
"""

import os 
import json
from tqdm import tqdm
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='runs/val/exp/last_predictions.json',type=str, help="input: coco format(json)")
parser.add_argument('--save_path', default='runs/val/exp/last_predictions_Txt', type=str, help="specify where to save the output dir of labels")
arg = parser.parse_args()

# For DOTA-v2.0
dotav2_classnames = [ 'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
         'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor',
         'swimming-pool', 'helicopter', 'container-crane', 'airport', 'helipad']
# For DOTA-v1.5
dotav15_classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']
# For DOTA-v1.0
datav1_classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
            'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

DOTA_CLASSES = dotav15_classnames
if __name__ == '__main__':
    json_file =  arg.json_path # COCO Object Instance 类型的标注
    ana_txt_save_path = arg.save_path  # 保存的路径

    data = json.load(open(json_file, 'r'))
    if os.path.exists(ana_txt_save_path):
        shutil.rmtree(ana_txt_save_path)  # delete output folderX
    os.makedirs(ana_txt_save_path)

    for data_dict in data:
        img_name = data_dict["file_name"]
        score = data_dict["score"]
        poly = data_dict["poly"]
        classname = DOTA_CLASSES[data_dict["category_id"]-1] # COCO's category_id start from 1, not 0

        lines = "%s %s %s %s %s %s %s %s %s %s\n" % (img_name, score, poly[0],poly[1],poly[2],poly[3],poly[4],poly[5],poly[6],poly[7])
        with open(str(ana_txt_save_path + '/Task1_' + classname) + '.txt', 'a') as f:
            f.writelines(lines)     
        pass
    print("Done!")