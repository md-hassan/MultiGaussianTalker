import cv2, os
import numpy as np
from utils.loss_utils import l1_loss, ssim, lpips_loss
from utils.image_utils import psnr
import argparse
import torch
import json
import lpips
import subprocess
loss_fn_alex = lpips.LPIPS(net='alex') 
loss_fn_vgg = lpips.LPIPS(net='vgg') 

def evaluate(gt_path, pred_path, json_path):
    os.makedirs('temp/gt', exist_ok=True)
    os.makedirs('temp/pred', exist_ok=True)
    gt_video = cv2.VideoCapture(gt_path)
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    gt_frames = []
    start_idx = json_file['frames'][0]['img_id']
    end_idx = json_file['frames'][0]['img_id'] + len(json_file['frames']) - 1
    while True:
        gt_ret, gt_frame = gt_video.read()
        if not gt_ret:
            break
        if start_idx <= end_idx:
            cv2.imwrite(f'temp/gt/{end_idx-start_idx}.jpg', gt_frame)
            gt_frame = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2RGB) / 255.0
            gt_frame = np.transpose(gt_frame, (2, 0, 1))
            gt_frames.append(gt_frame)
        start_idx += 1

    pred_frames = []
    pred_idx = 0
    pred_video = cv2.VideoCapture(pred_path)
    while True:
        pred_ret, pred_frame = pred_video.read()
        if not pred_ret:
            break
        cv2.imwrite(f'temp/pred/{pred_idx}.jpg', pred_frame)
        pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_BGR2RGB) / 255.0
        pred_frame = np.transpose(pred_frame, (2, 0, 1))
        pred_frames.append(pred_frame)
        pred_idx += 1

    assert len(gt_frames) == len(pred_frames), f'gt_frames: {len(gt_frames)}, pred_frames: {len(pred_frames)}'

    gt_frames = np.array(gt_frames)
    pred_frames = np.array(pred_frames)
    gt_frames = torch.tensor(gt_frames).float()
    pred_frames = torch.tensor(pred_frames).float()

    with torch.no_grad():
        print('Calculating SSIM')
        ssim_val = ssim(gt_frames, pred_frames)
        print('Calculating PSNR')
        psnr_val = torch.mean(psnr(gt_frames, pred_frames))
        print('Calculating LPIPS')
        lpips_val_alex = loss_fn_alex(gt_frames, pred_frames)
        # lpips_val_vgg = loss_fn_vgg(2 * gt_frames - 1, 2 * pred_frames - 1)

        print('SSIM:', ssim_val)
        print('PSNR:', psnr_val)
        print('LPIPS ALEX:', torch.mean(lpips_val_alex))
        # print('LPIPS VGG:', torch.mean(lpips_val_vgg))
        cmd = f'python -m pytorch_fid temp/gt temp/pred'
        os.system(cmd)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        fid_output = result.stdout
        print(fid_output.strip().split())
        fid_value = float(fid_output.strip().split()[-1])
        print('FID:', fid_value)

        print('Calculating Sync')
        cmd = f'python demo_syncnet.py --videofile {pred_path} --tmp_dir temp'
        os.system(cmd)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        sync_output = result.stdout
        sync_min_dist = float(sync_output.split('Min dist: \t')[1].split('\n')[0])
        sync_conf = float(sync_output.split('Confidence: \t')[1].split('\n')[0])
    
    gt_video.release()
    pred_video.release()
    os.system('rm -rf temp')

    return {'ssim': ssim_val.item(), 'psnr': psnr_val.item(), 'lpips_alex': torch.mean(lpips_val_alex).item(), 'fid': fid_value, 'sync_min_dist': sync_min_dist, 'sync_conf': sync_conf}

def list_of_strings(arg):
    return arg.split(',')

parser = argparse.ArgumentParser()
# parser.add_argument('--gt_path', type=str)
# parser.add_argument('--pred_path', type=str)
# parser.add_argument('--json_path', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument("--persons", type=list_of_strings, default = [])
args = parser.parse_args()
args.persons = [p.strip() for p in args.persons]

metrics = {}
for person in args.persons:
    print(f'Evaluating {person}')
    # args.gt_path = f'{args.data_path}/two_stage_mm_5_25k/test/ours_25000/gt/{person}_test_25000iter_gt.mov'
    # args.pred_path = f'{args.data_path}/two_stage_mm_5_25k/test/ours_25000/renders/{person}_test_25000iter_renders.mov'
    args.gt_path = f'{args.data_path}/baseline_{person}/test/ours_40000/gt/vids_25_test_40000iter_gt.mov'
    args.pred_path = f'{args.data_path}/baseline_{person}/test/ours_40000/renders/vids_25_test_40000iter_renders.mov'
    
    args.json_path = f'{args.data_path}/{person}/transforms_val.json'
    metrics[person] = evaluate(args.gt_path, args.pred_path, args.json_path)

avg_metrics = {}
for k in metrics[args.persons[0]].keys():
    avg_metrics[k] = sum([m[k] for m in metrics.values()]) / len(metrics)
metrics['avg'] = avg_metrics

print(metrics)
with open(f'{args.data_path}/baseline.json', 'w') as f:
    json.dump(metrics, f)

# python evaluate.py --gt_path data/vids_25/obama_25/obama_25.mp4 --pred_path data/vids_25/ckpt/test/ours_25000/renders/obama_25_test_25000iter_renders.mov --json_path data/vids_25/obama_25/transforms_val.json

# python evaluate.py --data_path /workspace/code/MultiGaussianTalker/data/vids_25 --persons "black_man, blue_girl, purple_girl, pewdiepie, may, obama_25, male_25, female_25, cnn_25, cnn2_25"

# python evaluate.py --data_path /workspace/code/MultiGaussianTalker/data/vids_25 --persons "black_man, blue_girl, purple_girl, pewdiepie, may, obama_25, male_25, female_25, cnn_25, cnn2_25"

# python evaluate.py --gt_path data/vids_25/obama_25/obama_25.mp4 --pred_path data/vids_25/ckpt/test/ours_25000/renders/obama_25_test_25000iter_renders.mov --json_path data/vids_25/obama_25/transforms_val.json