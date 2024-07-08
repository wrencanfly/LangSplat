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


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, opt, scaling_modifier = 1.0, override_color = None, iteration = -1):
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
        language_feature_precomp = pc.get_language_feature
        language_feature_precomp = language_feature_precomp / (language_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)

        # print("current iteration", iteration)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vq = False
        rvq_layer = False
        if vq == True:
            if iteration == -1:
                # Define the vector quantization layer
                if(rvq_layer):
                    vq_layer = ResidualVQ(
                        dim=3,              
                        num_quantizers=8,    
                        codebook_size=25,  
                        commitment_weight=0.1
                    ).to(device)
                else:
                    vq_layer = VectorQuantize(
                        dim=3,           
                        codebook_size=640,
                        commitment_weight =0.1,      
                        use_cosine_sim = True
                    ).to(device)
                    
                vq_layer = vq_layer
              
                language_feature_precomp = language_feature_precomp.view(1, -1, 3)  # Shape: (1, 2147799, 3)


                quantized_features, indices, _ = vq_layer(language_feature_precomp)
                
                # save
                
                if(rvq_layer):
                    codebook = vq_layer.codebooks 
                else:
                    codebook = vq_layer.codebook 

                torch.save(dict(codebook = codebook, indices = indices.to(torch.int16)), 'codebook.pt')
                torch.save(vq_layer.state_dict(), "vq.pt")

                
                language_feature_precomp = quantized_features.view(-1, 3)  # Remove batch dimension for further processing

            # LOAD
            elif iteration == -2:
                residual_vq_layer_loaded = ResidualVQ(
                    dim=3,                 # Dimension of the embeddings (same as the language features)
                    num_quantizers=8,      # Number of residual quantizers
                    codebook_size=128,     # Number of embeddings in each codebook
                    commitment_weight=0.1, # Commitment cost
                ).to(device)
                residual_vq_layer_loaded.load_state_dict(torch.load("vq.pt", map_location=torch.device('cpu')))

                codebook = torch.load('codebook.pt')['codebook']
                indices = torch.load('codebook.pt')['indices']

                residual_vq_layer_loaded.codebook = codebook

                residual_vq_layer_loaded.eval()

                quantized_out = residual_vq_layer_loaded.get_codes_from_indices(indices)
                language_feature_precomp = quantized_out.view(-1, 3)  # Remove batch dimension if necessary
            else:
                language_feature_precomp = language_feature_precomp

        else:
            language_feature_precomp = language_feature_precomp

        
        
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
