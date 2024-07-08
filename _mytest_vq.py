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
from vector_quantize_pytorch import ResidualVQ, VectorQuantize


from autoencoder.model import Autoencoder

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, args):
    pass
               
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        
        # print(np.shape(gaussians.get_language_feature)) # torch.Size([2147799, 3])
        # language_feature_precomp = gaussians.get_language_feature
        # language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)
        
        
        # ############  start here
        lvl = 1
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
        indices = torch.load(codebook_path)['indices']
        
        # decode to K x 512
        ae_ckpt_path = f"autoencoder/ckpt/{scene}/best_ckpt.pth"
        checkpoint = torch.load(ae_ckpt_path, map_location=device)
        model = Autoencoder([256,128,64,32,3], [16, 32, 64, 128, 256, 256, 512]).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # codebook size before decode
        print("codebook size before decode", codebook.shape)
        codebook_de = model.decode(codebook)
        print("codebook size after decode", codebook_de.shape)
        
        assert False

        # vq_layer.codebook = codebook

        # vq_layer.eval()

        # quantized_out = vq_layer.get_codes_from_indices(indices)
        # language_feature_precomp = quantized_out.view(-1, 3)  
        
        
        # torch.save(language_feature_precomp, 'language_feats_dim3_tensor_1.pt')
        #torch.save(language_feature_precomp, 'language_feats_dim3_tensor_2.pt')
        # torch.save(language_feature_precomp, f'language_feats_dim3_tensor_{lvl}.pt')
        
        # print("language_feature_precomp shape", np.shape(language_feature_precomp)) # torch.Size([2147799, 3])
        
        
        ###############  start here
        
        
        # compressed_sem_feats shape (3, 6, 730, 988, 3)
        # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        #      render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        # if not skip_test:
        #      render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

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
