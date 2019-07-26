import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from os.path import join
from tqdm import tqdm, tqdm_notebook

from cam import CAM
import util


# ===========================================================
# settings
# ===========================================================
parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpu_number', type=str, default='1')
parser.add_argument('--th1',     type=float, default=0.4, help='threshold for the heatmap mean value')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--runs_dir', type=str, default='')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number

args.runs_dir = join('runs', f'{args.th1}')
if not os.path.exists(args.runs_dir):
    os.makedirs(args.runs_dir)


def write_log(string):
    with open(join(args.runs_dir,'log.log'), 'a') as lf:
        sys.stdout = lf
        print(string)


def write_json(bboxes):
    with open(join(args.runs_dir,'bbox.json'), 'a') as jf:
        json.dump(bboxes, jf)


# ===========================================================
# main
# ===========================================================
if __name__ == "__main__":
    map = CAM()

    data_dict = map.valid_dataset.data_dict
    input_files = map.valid_dataset.img_files
    img_dir = map.valid_dataset.img_dir

    bboxes_cam = {}

    write_log('Localization Accuracy')

    count = 0
    correct_cam = 0
    for data_idx in range(len(data_dict)):
        count += 1

        # get true bbox
        input_file = input_files[data_idx]
        img_origin = Image.open(join(img_dir, input_file)).convert('RGB')
        bboxes_true = data_dict[input_file][1]
        bboxes_true = util.bboxes_resize(img_origin, bboxes_true, size=224)

        # get input and target
        input, target = map.get_item(data_idx)
        target = target.cpu().item()

        ''' CAM origin version '''
        _, _, _, _, bbox_pred = map.get_values(data_idx, target, th1=0.2, phase='test')

        ''' get iou '''
        iou_preds = []
        for bbox_true in bboxes_true:
            iou_preds.append(util.get_iou(bbox_true, bbox_pred))
            correct_cam += max(np.array(iou_preds) >= 0.5).astype(np.int)

        ''' save bboxes for every 100 iteration '''
        bboxes_cam[input_file] = bbox_pred

        if (data_idx+1) % 100 == 0:
            # print the log
            write_log(f'iter {data_idx+1:05d} ===> Origin: {correct_cam/count}')
            
            # save bbox in json file
            write_json(bboxes_cam)
            
            bboxes_cam= {}


