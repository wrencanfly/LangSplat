#!/usr/bin/env python
from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm

from vector_quantize_pytorch import ResidualVQ, VectorQuantize

import sys
sys.path.append("..")
import colormaps
from autoencoder.model import Autoencoder
from openclip_encoder import OpenCLIPNetwork
from utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths


def activate_stream(scene_index,
                    sem_map, 
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    img_ann: Dict = None, 
                    thresh : float = 0.5, 
                    colormap_options = None,
                    lvl : int = 3, query_dim=-1): # here query dim represent the pos, and 4 neg. -1 represent pos.
    valid_map = clip_model.query_3d_gaussian(sem_map, query_dim=query_dim)                 # 3xkx832x1264 # lvl x pos_query x H x W - original
    
    if query_dim == -1:
        return torch.save(valid_map, f'{scene_index}_valid_map_level_{lvl}.pt')
    else:
        return torch.save(valid_map, f'{scene_index}_valid_map_level_{lvl}_neg_{query_dim}.pt')
    


def lerf_localization(sem_map, image, clip_model, image_name, img_ann):
    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)

    valid_map = clip_model.get_max_across(sem_map)                 # 3xkx832x1264 # n_levels, n_phrases, h, w
    n_head, n_prompt, h, w = valid_map.shape
    
    # positive prompts
    acc_num = 0
    positives = list(img_ann.keys()) # all the text labels
    for k in range(len(positives)):
        select_output = valid_map[:, k]
        
        # NOTE 平滑后的激活值图中找最大值点
        scale = 30
        kernel = np.ones((scale,scale)) / (scale**2)
        np_relev = select_output.cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev.transpose(1,2,0), -1, kernel)
        
        score_lvl = np.zeros((n_head,))
        coord_lvl = []
        for i in range(n_head):
            score = avg_filtered[..., i].max()
            coord = np.nonzero(avg_filtered[..., i] == score)
            score_lvl[i] = score
            coord_lvl.append(np.asarray(coord).transpose(1,0)[..., ::-1])

        selec_head = np.argmax(score_lvl)
        coord_final = coord_lvl[selec_head]
        
        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (cord_list[0] >= x_min and cord_list[0] <= x_max and 
                    cord_list[1] >= y_min and cord_list[1] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break
        
        # NOTE 将平均后的结果与原结果相加，抑制噪声并保持激活边界清晰
        avg_filtered = torch.from_numpy(avg_filtered[..., selec_head]).unsqueeze(-1).to(select_output.device)
        torch_relev = 0.5 * (avg_filtered + select_output[selec_head].unsqueeze(-1))
        p_i = torch.clip(torch_relev - 0.5, 0, 1)
        valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (torch_relev < 0.5).squeeze()
        valid_composited[mask, :] = image[mask, :] * 0.3
        
        save_path = output_path_loca / f"{positives[k]}.png"
        show_result(valid_composited.cpu().numpy(), coord_final,
                    img_ann[positives[k]]['bboxes'], save_path)
    return acc_num


def evaluate(feat_dir, output_path, ae_ckpt_path, json_folder, mask_thresh, encoder_hidden_dims, decoder_hidden_dims, logger, lvl):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    # FIXME
    with torch.no_grad():

        
        scene = "figurines"
        
        codebook_path = f"/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yingwei/LangSplat_vq/vq/{scene}/codebook_{lvl}.pt"
        vq_path = f"/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yingwei/LangSplat_vq/vq/{scene}/vq_{lvl}.pt"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vq_layer = VectorQuantize(
            dim=3,           
            codebook_size=640, 
            commitment_weight =0.1,      
            use_cosine_sim = True
        ).to(device)
        vq_layer.load_state_dict(torch.load(vq_path, map_location=torch.device('cpu')))


        
        codebook = torch.load(codebook_path)['codebook']
        #indices = torch.load(codebook_path)['indices']
        
        # decode to K x 512
        # ae_ckpt_path = f"autoencoder/ckpt/{scene}/best_ckpt.pth"
        # checkpoint = torch.load(ae_ckpt_path, map_location=device)
        
        
        lg_tensor_1 = codebook
        lg_stacked_tensor = torch.unsqueeze(lg_tensor_1, 0).to(device)
       
        # #print("lg_stacked_tensor shape", lg_stacked_tensor.shape)
        # lg_tensor_2 = torch.load('language_feats_dim3_tensor_2.pt')
        # lg_tensor_3 = torch.load('language_feats_dim3_tensor_3.pt')
        # lg_stacked_tensor = torch.stack([lg_tensor_1, lg_tensor_2, lg_tensor_3], dim=0).to(device)  # Choose dim=0 for a new first dimension
    
    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(json_folder), Path(output_path))
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    # compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), *image_shape, 3), dtype=np.float32)
    # print("compressed_sem_feats shape", compressed_sem_feats.shape) # compressed_sem_feats shape (3, 6, 730, 988, 3)
    # for i in range(len(feat_dir)):
    #     feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
    #                            key=lambda file_name: int(os.path.basename(file_name).split(".npy")[0]))
    #     for j, idx in enumerate(eval_index_list):
    #         compressed_sem_feats[i][j] = np.load(feat_paths_lvl[idx])


    # instantiate autoencoder and openclip
    clip_model = OpenCLIPNetwork(device)
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    chosen_iou_all, chosen_lvl_list = [], []
    acc_num = 0
    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / f'{idx+1:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)
        
        # sem_feat = compressed_sem_feats[:, j, ...]
        # sem_feat = torch.from_numpy(sem_feat).float().to(device)
        rgb_img = cv2.imread(image_paths[j])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        with torch.no_grad():
        #     lvl, h, w, _ = sem_feat.shape
        #     print("sem_feat.flatten(0, 2) shape", sem_feat.flatten(0, 2).shape)
        #     restored_feat = model.decode(sem_feat.flatten(0, 2))
        #     restored_feat = restored_feat.view(lvl, h, w, -1)           # 3x832x1264x512
            lg_stacked_tensor_decoded = model.decode(lg_stacked_tensor) # 3 x N x 512

        img_ann = gt_ann[f'{idx}']
        clip_model.set_positives(list(img_ann.keys()))
        # get keys
        # print(list(img_ann.keys()))

        # continue        
        print("lg_stacked_tensor shape", lg_stacked_tensor_decoded.shape)
        # print("current mem ussage", torch.cuda.mem_get_info())
        for query in range(-1, 4):
            activate_stream(j, lg_stacked_tensor_decoded, rgb_img, clip_model, image_name, img_ann,
                                            thresh=mask_thresh, colormap_options=colormap_options, lvl=lvl, query_dim=query)
        # chosen_iou_all.extend(c_iou_list)
        # chosen_lvl_list.extend(c_lvl)

        # acc_num_img = lerf_localization(restored_feat, rgb_img, clip_model, image_name, img_ann)
        # acc_num += acc_num_img
        print("scene done")
        assert False

    assert False
    # # iou
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f'trunc thresh: {mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")

    # # localization acc
    # total_bboxes = 0
    # for img_ann in gt_ann.values():
    #     total_bboxes += len(list(img_ann.keys()))
    # acc = acc_num / total_bboxes
    # logger.info("Localization accuracy: " + f'{acc:.4f}')


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    
    parser = ArgumentParser(description="prompt any label")
    
    parser.add_argument("--lvl", type=int, default=3)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument("--ae_ckpt_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--json_folder", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument('--encoder_dims',
                        nargs = '+',
                        type=int,
                        default=[256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs = '+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )
    args = parser.parse_args()

    # NOTE config setting
    dataset_name = args.dataset_name
    mask_thresh = args.mask_thresh
    # FIXME here for semi test
    feat_dir = [os.path.join(args.feat_dir, dataset_name+f"_{i}", "train/ours_None/renders_npy") for i in range(1,4)]
    output_path = os.path.join(args.output_dir, dataset_name)
    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "best_ckpt.pth")
    json_folder = os.path.join(args.json_folder, dataset_name)
    lvl = args.lvl
    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)

    evaluate(feat_dir, output_path, ae_ckpt_path, json_folder, mask_thresh, args.encoder_dims, args.decoder_dims, logger, lvl)