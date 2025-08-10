import os
import argparse
import json
import re
from PIL import Image, ImageFile, ImageDraw
from tqdm import tqdm
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-file', type=str, default='playground/Instructions_slim/Grounding/test.json')
    parser.add_argument('--result-file', type=str, default='results/CoIN_normaltrain_testslim/Grounding/OCRVQA/merge.jsonl')
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()

def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

def change_bbox(bbox,im_w,im_h):
    x,y,w,h = bbox
    x1,y1,x2,y2 = x,y,x+w,y+h
    max_wh = max(im_w,im_h)
    if im_w == im_h:
        return [x/max_wh, y/max_wh, x2/max_wh, y2/max_wh]
    elif im_w > im_h:
        y1 = y1 + (im_w-im_h)/2
        y2 = y2 + (im_w-im_h)/2
        return [x1/max_wh, y1/max_wh, x2/max_wh, y2/max_wh]
    else:
        x1 = x1 + (im_h-im_w)/2
        x2 = x2 + (im_h-im_w)/2
        return [x1/max_wh, y1/max_wh, x2/max_wh, y2/max_wh]

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x21, y21, x22, y22 = bbox2
    intersection_area = max(0, min(x2, x22) - max(x1, x21)) * max(0, min(y2, y22) - max(y1, y21))
    union_area = (x2 - x1) * (y2 - y1) + (x22 - x21) * (y22 - y21) - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def eval_single(test_file, result_file):
    annotations = json.load(open(test_file))
    annotations = {grounding_test['question_id']: grounding_test for grounding_test in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    total = len(results)
    right = 0
    for result in results:
        grounding_gt = annotations[result['question_id']]
        bbox_string = grounding_gt['answer_bbox']
        bbox_string = bbox_string.replace('[', '').replace(']', '')
        bbox_groundtruth = [float(x) for x in bbox_string.split(',')]
        size = grounding_gt['size']

        pred_bbox = result['text']
        try:
            pred_bbox = pred_bbox.replace('[', '').replace(']', '')
            bbox_pred = [float(x) for x in pred_bbox[1:-1].split(',')]
            if len(bbox_pred) != 4:
                continue
        except:
            continue

        # bbox_pred = change_bbox((bbox_pred[0],bbox_pred[1],bbox_pred[2],bbox_pred[3]),size[0],size[1])

        max_wh = max(size)
        bbox_pred = [x * max_wh for x in bbox_pred]
        bbox_groundtruth = [x * max_wh for x in bbox_groundtruth]

        iou = calculate_iou(bbox_pred, bbox_groundtruth)
        right += iou > 0.5

        # pil_img = Image.open(os.path.join('./cl_dataset/COCO2014/train2014/',os.path.split(grounding_gt['image'])[-1])).convert('RGB')
        # image = expand2square(pil_img, tuple(int(x*255) for x in [0,0,0,0]))
        # draw = ImageDraw.Draw(image)
        
        # draw.rectangle([bbox_pred[0], bbox_pred[1], (bbox_pred[2]), (bbox_pred[3])],outline='red',width=2)
        # draw.rectangle([bbox_groundtruth[0], bbox_groundtruth[1], (bbox_groundtruth[2]), (bbox_groundtruth[3])],outline='green',width=2)
        # # save image
        # image.save(os.path.join('./llava/eval/Instruction_CC', 'test'+'.jpg'))
        # print(grounding_gt['text'])
        # print('1')

    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
    #将结果写入文件
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.test_file, args.result_file)
