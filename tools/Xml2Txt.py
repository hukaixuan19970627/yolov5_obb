import xml.etree.ElementTree as ET
from DOTA_devkit import dota_utils as util
import os
import  shutil

def DroneVehicle2DOTA(xmlpath, txtpath, extractclassname, specified_class):
    """
    Trans DroneVehicle xml farmat to DOTA labels format
    Args:
        xmlpath: the path of xml with DroneVehicle2DOTA format
        txtpath: the path of txt with DOTA format
        extractclassname: the category 
        specified_class: only output single category
    """
    if os.path.exists(txtpath):
        shutil.rmtree(txtpath)  # delete output folder
    os.makedirs(txtpath)  # make new output folder
    filelist = util.GetFileFromThisRootDir(xmlpath)  # fileist=['/.../0001.xml', ...]
    for xmlfile in filelist:  # fullname='/.../0001.xml'
        name = os.path.splitext(os.path.basename(xmlfile))[0]  # name='0001'
        out_file = open(os.path.join(txtpath, name + '.txt'), 'w')
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        for obj in root.iter('object'):
            cls = obj.find('name')
            if cls == None:
                continue
            cls = cls.text
            diffcult = obj.find('difficult')
            diffcult = int(diffcult.text) if diffcult != None else 0

            if diffcult < 2:
                # cls = cls.replace(' ', '_')
                cls = specified_class
                polygon = obj.find('polygon')
                if polygon == None:
                    continue
                polygon = [int(polygon.find(x).text) for x in ('x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4')]
                out_file.write(" ".join([str(a) for a in (*polygon, cls, diffcult)]) + '\n')
            else:
                print(f'{cls} is not in the extractclassname or diffcult is {diffcult}')

if __name__ == "__main__":
    xmlpath = ['/media/test/DroneVehicle/val/raw/vallabel',
               '/media/test/DroneVehicle/val/rays/vallabelr',
               '/media/test/DroneVehicle/train/raw/trainlabel',
               '/media/test/DroneVehicle/train/rays/trainlabelr']

    txtpath = ['/media/test/DroneVehicle/val/raw/vallabel_txt',
               '/media/test/DroneVehicle/val/rays/vallabelr_txt',
               '/media/test/DroneVehicle/train/raw/trainlabel_txt',
               '/media/test/DroneVehicle/train/rays/trainlabelr_txt']
    extractclassname = {'car', 'truck', 'feright_car', 'feright car', 'bus', 'van'}
    specified_class = 'vehicle'
    for (xml, txt) in zip(xmlpath, txtpath):
        print(f"{xml} is converting to {txt}")
        DroneVehicle2DOTA(xml, txt, extractclassname, specified_class)