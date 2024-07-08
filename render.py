#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import time
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args):


    frame_lst =  [40, 104, 151, 194]
    frame_lst =  [40]
    positives_lst = [
['old camera', 'toy elephant', 'waldo', 'tesla door handle', 'porcelain hand', 'rubber duck with hat', 'rubber duck with buoy', 'pink ice cream', 'red toy chair', 'green apple', 'pikachu', 'red apple', 'spatula', 'jake', 'toy cat statue', 'pirate hat', 'miffy'],
['rubics cube', 'green apple', 'green toy chair', 'jake', 'old camera', 'pink ice cream', 'pumpkin', 'red apple', 'rubber duck with hat', 'tesla door handle', 'spatula', 'rubber duck with buoy', 'pirate hat'],
['rubber duck with hat', 'rubics cube', 'toy elephant', 'green apple', 'jake', 'toy cat statue', 'pikachu', 'porcelain hand', 'red apple', 'waldo', 'pirate hat'],
['toy elephant', 'pink ice cream', 'porcelain hand', 'green apple', 'green toy chair', 'old camera', 'rubics cube', 'spatula', 'toy cat statue', 'waldo', 'rubber duck with buoy', 'pirate hat', 'miffy', 'bag', 'rubber duck with hat']]

    
    # positives = ['stuffed bear', 'coffee mug', 'bag of cookies', 'sheep', 'apple', 'paper napkin', 'plate', 'tea in a glass', 'bear nose', 'three cookies', 'coffee']
    for scene_index, frame in enumerate(frame_lst):
        scene_folder = os.path.join(model_path, name, 'scene_{}'.format(frame))
        os.makedirs(scene_folder, exist_ok=True)
        
        positives = positives_lst[scene_index]
        for pos_index, positive in enumerate(positives):
            render_path = os.path.join(scene_folder, '{}'.format(positive), "ours_{}".format(iteration), "renders")
            gts_path = os.path.join(scene_folder, '{}'.format(positive), "ours_{}".format(iteration), "gt")
            render_npy_path = os.path.join(scene_folder, '{}'.format(positive), "ours_{}".format(iteration), "renders_npy")
            gts_npy_path = os.path.join(scene_folder, '{}'.format(positive), "ours_{}".format(iteration), "gt_npy")

            os.makedirs(render_npy_path, exist_ok=True)
            os.makedirs(gts_npy_path, exist_ok=True)
            os.makedirs(render_path, exist_ok=True)
            os.makedirs(gts_path, exist_ok=True)

            for idx, view in enumerate(tqdm(views, desc="Rendering progress for scene {}".format(frame))):
                if idx == frame:

                    start_time = time.time()
                    output = render(scene_index, view, gaussians, pipeline, background, pos_index, args)
                    end_time = time.time()
                    loading_time = end_time - start_time
                    print(f"Time taken to render: {loading_time:.4f} seconds")
                    
                    if not args.include_feature:
                        rendering = output["render"]
                    else:
                        rendering = output["language_feature_image"]

                    if not args.include_feature:
                        gt = view.original_image[0:3, :, :]
                    else:
                        gt, mask = view.get_language_feature(os.path.join(source_path, args.language_features_name), feature_level=args.feature_level)


                    np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + ".npy"), rendering.permute(1, 2, 0).cpu().numpy())
                    # np.save(os.path.join(gts_npy_path, '{0:05d}'.format(idx) + ".npy"), gt.permute(1, 2, 0).cpu().numpy())
                    
                    # print("rendering shape", rendering.shape) # rendering shape torch.Size([5, 730, 988]) [03/06 01:12:21]
                    
                    img = rendering[0, :, :].unsqueeze(0).repeat(3, 1, 1)
                    torchvision.utils.save_image(img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                    # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
                    assert False

                
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)