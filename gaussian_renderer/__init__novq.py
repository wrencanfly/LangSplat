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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from vector_quantize_pytorch import ResidualVQ, VectorQuantize


def render(scene_idx, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, pos_index : int, opt, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        include_feature=opt.include_feature,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if opt.include_feature:

        # @torch.no_grad()
        # def threshold_and_enhance_similarity(similarity, threshold=0.9, scale_factor=100):
            
        #     # enhanced_similarity = torch.where(similarity < threshold, torch.ones_like(similarity), similarity * scale_factor)
        #     enhanced_similarity = torch.where(similarity < threshold, torch.ones_like(similarity), torch.full_like(similarity, 5))
        #     return enhanced_similarity




        #language_feature_precomp = pc.get_language_feature
        # print("language_feature_precomp shape", language_feature_precomp.shape)
        
        
        #language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)
        
        # self.negatives = ("object", "things", "stuff", "texture")
        # # language_feature_precomp = torch.sigmoid(language_feature_precomp)
        # normalized_similarity = torch.load(f'/datadrive/yingwei/LangSplat/eval/{scene_idx}_valid_map_level_2_neg_3.pt')
        
        # ****************
        level = 1
        
        valid_map_pos = f'/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yingwei/LangSplat/eval/{scene_idx}_valid_map_level_{level}.pt'
        
        valid_map_neg_0 = f'/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yingwei/LangSplat/eval/{scene_idx}_valid_map_level_{level}_neg_0.pt'
        valid_map_neg_1 = f'/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yingwei/LangSplat/eval/{scene_idx}_valid_map_level_{level}_neg_1.pt'
        valid_map_neg_2 = f'/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yingwei/LangSplat/eval/{scene_idx}_valid_map_level_{level}_neg_2.pt'
        valid_map_neg_3 = f'/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/yingwei/LangSplat/eval/{scene_idx}_valid_map_level_{level}_neg_3.pt'

        
        # normalized_similarity = torch.load(valid_map_pos)
        # selected_similarity = normalized_similarity[0, :,pos_index]
        # selected_similarity = selected_similarity.unsqueeze(1)
        
        start_time = time.time()

        normalized_similarity = torch.load(valid_map_pos)
        neg_0 = torch.load(valid_map_neg_0)
        neg_1 = torch.load(valid_map_neg_1)
        neg_2 = torch.load(valid_map_neg_2)
        neg_3 = torch.load(valid_map_neg_3)
        
        # Select the necessary slices
        selected_similarity_pos = normalized_similarity[0, :, pos_index].unsqueeze(1)
        selected_similarity_neg_0 = neg_0[0, :, pos_index].unsqueeze(1)
        selected_similarity_neg_1 = neg_1[0, :, pos_index].unsqueeze(1)
        selected_similarity_neg_2 = neg_2[0, :, pos_index].unsqueeze(1)
        selected_similarity_neg_3 = neg_3[0, :, pos_index].unsqueeze(1)
        

        selected_similarity = torch.cat((selected_similarity_pos, selected_similarity_neg_0, selected_similarity_neg_1, selected_similarity_neg_2, selected_similarity_neg_3), dim=1)
        end_time = time.time()
        
        print("selected_similarity shape", selected_similarity.shape)

        loading_time = end_time - start_time
        print(f"Time taken to load tensors: {loading_time:.4f} seconds")
         # ****************
        
        # # self.positives ['stuffed bear', 'coffee mug', 'bag of cookies', 'sheep', 'apple', 'paper napkin', 'plate', 'tea in a glass', 'bear nose', 'three cookies', 'coffee']
        # print("negative normalized_similarity shape", normalized_similarity.shape)
        
        # FIXME normal here???
        # normalized_similarity = normalized_similarity/ (normalized_similarity.norm(dim=-1, keepdim=True) + 1e-9)
        # selected_similarity = normalized_similarity[0, :,pos_index]
        # # Normalize similarity to [0, 1]
  
        # # Enhance the similarity values
        # enhanced_similarity = threshold_and_enhance_similarity(selected_similarity, threshold=0.45)  # Shape: (2147799,)

        # Expand enhanced similarity to match language_feature_precomp dimensions
        # Shape: (2147799,) -> (2147799, 3)
        # enhanced_similarity_expanded = enhanced_similarity.unsqueeze(1).expand(-1, 3)
        # selected_similarity = selected_similarity.unsqueeze(1).expand(-1, 3)
        # Multiply language_feature_precomp with the expanded enhanced similarity

        #language_feature_precomp = language_feature_precomp * enhanced_similarity_expanded  # Shape: (2147799, 3)
        
        # zeros_tensor = torch.zeros(2147799, 2, device=selected_similarity.device)
        # selected_similarity = torch.cat((selected_similarity, zeros_tensor), dim=1)
        
        # language_feature_precomp = pc.get_language_feature
        # language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)
        # print("selected_similarity shape", language_feature_precomp.shape)
        language_feature_precomp = selected_similarity  # Shape: (2147799, 3)

    else:
        language_feature_precomp = torch.zeros((1,), dtype=opacity.dtype, device=opacity.device)
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # start_time = time.time()

    rendered_image, language_feature_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        language_feature_precomp = language_feature_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # end_time = time.time()
    # print('render_init_rasterizer程序运行时间为: %s Seconds'%(end_time-start_time))
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    
    return {"render": rendered_image,
            "language_feature_image": language_feature_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
