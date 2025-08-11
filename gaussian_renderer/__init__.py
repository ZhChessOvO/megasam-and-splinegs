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

import torch
from gsplat.rendering import rasterization, fully_fused_projection
from scene.gaussian_model import GaussianModel
import cv2


# @torch.jit.script
def interpolate_cubic_hermite(signal, times, N):
    # start.record()
    times_scaled = times * (N - 1)[:, None]
    indices = torch.floor(times_scaled).long()
    # Clamping to avoid out-of-bounds indices

    indices = torch.clamp(
        indices, torch.zeros_like(N)[:, None].expand(-1, 3, -1), (N - 2)[:, None].expand(-1, 3, -1)
    ).long()
    left_indices = torch.clamp(
        indices - 1, torch.zeros_like(N)[:, None].expand(-1, 3, -1), (N - 1)[:, None].expand(-1, 3, -1)
    ).long()
    right_indices = torch.clamp(
        indices + 1, torch.zeros_like(N)[:, None].expand(-1, 3, -1), (N - 1)[:, None].expand(-1, 3, -1)
    ).long()
    right_right_indices = torch.clamp(
        indices + 2, torch.zeros_like(N)[:, None].expand(-1, 3, -1), (N - 1)[:, None].expand(-1, 3, -1)
    ).long()

    t = times_scaled - indices.float()
    p0 = torch.gather(signal, -1, left_indices)
    p1 = torch.gather(signal, -1, indices)
    p2 = torch.gather(signal, -1, right_indices)
    p3 = torch.gather(signal, -1, right_right_indices)

    # One-sided derivatives at the boundaries
    m0 = torch.where(left_indices == indices, (p2 - p1), (p2 - p0) / 2)
    m1 = torch.where(right_right_indices == right_indices, (p2 - p1), (p3 - p1) / 2)

    # Hermite basis functions
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)

    interpolation = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1
    # if len(signal.shape) == 3:  # remove extra singleton dimension
    interpolation = interpolation.squeeze(-1)
    # end.record()
    # torch.cuda.synchronize()
    # print('v1:', start.elapsed_time(end))
    return interpolation


# @torch.jit.script
def interpolate_cubic_hermite_infer(signal, times, N, index_offset):
    # start.record()
    times_scaled = times * (N - 1)
    indices = torch.floor(times_scaled).long()

    # Clamping to avoid out-of-bounds indices
    indices = torch.clamp(indices, torch.zeros_like(N).expand(-1, 3), (N - 2).expand(-1, 3)).long()
    left_indices = torch.clamp(indices - 1, torch.zeros_like(N).expand(-1, 3), (N - 1).expand(-1, 3)).long()
    right_indices = torch.clamp(indices + 1, torch.zeros_like(N).expand(-1, 3), (N - 1).expand(-1, 3)).long()
    right_right_indices = torch.clamp(indices + 2, torch.zeros_like(N).expand(-1, 3), (N - 1).expand(-1, 3)).long()

    t = times_scaled - indices.float()
    p0 = torch.gather(signal, 0, left_indices + index_offset)
    p1 = torch.gather(signal, 0, indices + index_offset)
    p2 = torch.gather(signal, 0, right_indices + index_offset)
    p3 = torch.gather(signal, 0, right_right_indices + index_offset)

    # One-sided derivatives at the boundaries
    m0 = torch.where(left_indices == indices, (p2 - p1), (p2 - p0) / 2)
    m1 = torch.where(right_right_indices == right_indices, (p2 - p1), (p3 - p1) / 2)

    # Hermite basis functions
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t**2 * (3 - 2 * t)
    h11 = t**2 * (t - 1)

    interpolation = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1
    # end.record()
    # torch.cuda.synchronize()
    # print('v2:', start.elapsed_time(end))
    return interpolation


def render(
    viewpoint_camera,
    stat_pc: GaussianModel,
    dyn_pc: GaussianModel,
    bg_color: torch.Tensor,
    get_static=False,
    get_dynamic=False,
):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    # Get dyn variables
    means3D = dyn_pc.get_xyz.detach()
    no_dyn_gs = means3D.shape[0]
    scales = dyn_pc._scaling
    rotations = dyn_pc._rotation
    opacity = dyn_pc.get_opacity

    # Get stat variables
    stat_means3D = stat_pc.get_xyz
    no_stat_gs = stat_means3D.shape[0]
    stat_opacity = stat_pc.get_opacity
    stat_colors_precomp = stat_pc.get_features_static
    stat_scales = stat_pc.get_scaling
    stat_rotations = stat_pc.get_rotation_stat

    pointtimes = (
        torch.ones((dyn_pc.get_xyz.shape[0], 1), dtype=dyn_pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    )  #

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device)
    K = viewpoint_camera.K.to(viewmat.device)
    bg_color = bg_color[:3]
    bg_color = torch.concat([bg_color, bg_color, bg_color], dim=-1)

    trbfdistanceoffset = viewpoint_camera.time * pointtimes
    tforpoly = trbfdistanceoffset.detach()
    rotations = dyn_pc.get_rotation_dy(rotations, tforpoly)

    control_xyz = dyn_pc.get_control_xyz.cuda()
    curr_time = torch.tensor(viewpoint_camera.time).cuda()
    deform_means3D = interpolate_cubic_hermite(
        control_xyz.permute(0, 2, 1),
        curr_time[None, None].expand(control_xyz.shape[0], 3, 1),
        N=dyn_pc.current_control_num,
    )
    means3D = deform_means3D * dyn_pc.deform_spatial_scale

    # Apply activations
    scales = dyn_pc.scaling_activation(scales)
    rotations = dyn_pc.rotation_activation(rotations)
    colors_precomp = dyn_pc.get_features(tforpoly)

    smeans3D_final, sscales_final, srotations_final, sopacity_final = (
        stat_means3D,
        stat_scales,
        stat_rotations,
        stat_opacity,
    )
    means3D_final, scales_final, rotations_final, opacity_final = means3D, scales, rotations, opacity
    
    # instance_mask = cv2.imread("/home/nerf/VIC-3DGS/data/nvidia_rodynrf/Balloon2/instance_mask_manual/000/001.png")
    # instance_mask = torch.from_numpy(instance_mask[...,0]).cuda()[None,None] / 255.0
    
    # _, means2D_final, _, _, _ = fully_fused_projection(
    #     means = means3D_final,
    #     covars=None,
    #     quats=rotations_final,
    #     scales=scales_final,
    #     viewmats = viewmat[None],
    #     Ks=K[None],  # [C, 3, 3]
    #     width=int(viewpoint_camera.image_width),
    #     height=int(viewpoint_camera.image_height),
    # ) # B, N, 2

    # means2D_final[..., 0] = means2D_final[...,0] / int(viewpoint_camera.image_width)
    # means2D_final[..., 1] = means2D_final[...,1] / int(viewpoint_camera.image_height)
    # means2D_final = 2*means2D_final[None] - 1.0  
    # pts_mask = torch.nn.functional.grid_sample(
    #     instance_mask,
    #     means2D_final,
    #     mode='nearest',
    #     padding_mode='zeros',
    #     align_corners=False
    # )

    # valid_index = (pts_mask > 0.0).squeeze()    

    d_alpha = None
    if get_dynamic:
        dmeans3D_final = means3D_final
        dscales_final = scales_final
        drotations_final = rotations_final
        dopacity_final = opacity_final
        dcolors_precomp = colors_precomp

        d_img, _, _ = rasterization(
            means=dmeans3D_final,
            quats=drotations_final,
            scales=dscales_final,
            opacities=dopacity_final.squeeze(-1),
            colors=dcolors_precomp,
            backgrounds=bg_color[None],
            viewmats=viewmat[None].detach(),  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB+ED",
        )

        d_depth = d_img[..., -1]
        d_img = d_img[..., :-1].permute(0, 3, 1, 2)
        d_image = dyn_pc.rgbdecoder(d_img, viewpoint_camera.cam_ray)
        d_image = d_image.squeeze(0)

    # Combine stat and dyn gaussians
    means3D_final = torch.cat((smeans3D_final, means3D_final), 0)
    scales_final = torch.cat((sscales_final, scales_final), 0)
    rotations_final = torch.cat((srotations_final, rotations_final), 0)
    opacity_final = torch.cat((sopacity_final, opacity_final), 0)
    colors_precomp_final = torch.cat((stat_colors_precomp, colors_precomp), 0)
    
    

    # means3D = dyn_pc.get_xyz.detach()
    # no_dyn_gs = means3D.shape[0]
    # scales = dyn_pc._scaling
    # rotations = dyn_pc._rotation
    # opacity = dyn_pc.get_opacity
    # pointtimes = (
    #     torch.ones((dyn_pc.get_xyz.shape[0], 1), dtype=dyn_pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    # )  #

    # viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device)
    # K = viewpoint_camera.K.to(viewmat.device)
    # bg_color = bg_color[:3]
    # bg_color = torch.concat([bg_color, bg_color, bg_color], dim=-1)

    # trbfcenter = dyn_pc.get_trbfcenter
    # trbfdistanceoffset = 1.0 * pointtimes - trbfcenter
    # tforpoly = trbfdistanceoffset.detach()
    # rotations = dyn_pc.get_rotation_dy(rotations, tforpoly)

    # control_xyz = dyn_pc.get_control_xyz.cuda()
    # curr_time = torch.tensor(1.0).cuda()
    # deform_means3D = interpolate_cubic_hermite(
    #     control_xyz.permute(0, 2, 1),
    #     curr_time[None, None].expand(control_xyz.shape[0], 3, 1),
    #     N=dyn_pc.current_control_num,
    # )
    # means3D = deform_means3D * dyn_pc.deform_spatial_scale

    # # Apply activations
    # scales = dyn_pc.scaling_activation(scales)
    # rotations = dyn_pc.rotation_activation(rotations)
    # colors_precomp_2 = dyn_pc.get_features(tforpoly)
    # means3D_final_2, scales_final_2, rotations_final_2, opacity_final_2 = means3D, scales, rotations, opacity
    
    # means3D_final_2 = means3D_final_2[valid_index]
    # scales_final_2 = scales_final_2[valid_index]
    # rotations_final_2 = rotations_final_2[valid_index]
    # opacity_final_2 = opacity_final_2[valid_index]
    # colors_precomp_2 = colors_precomp_2[valid_index]
    
    # means3D_final = torch.cat((means3D_final, means3D_final_2), 0)
    # scales_final = torch.cat((scales_final, scales_final_2), 0)
    # rotations_final = torch.cat((rotations_final, rotations_final_2), 0)
    # opacity_final = torch.cat((opacity_final, opacity_final_2), 0)
    # colors_precomp_final = torch.cat((colors_precomp_final, colors_precomp_2), 0)
    
    
    # means3D = dyn_pc.get_xyz.detach()
    # no_dyn_gs = means3D.shape[0]
    # scales = dyn_pc._scaling
    # rotations = dyn_pc._rotation
    # opacity = dyn_pc.get_opacity
    # pointtimes = (
    #     torch.ones((dyn_pc.get_xyz.shape[0], 1), dtype=dyn_pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    # )  #

    # viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device)
    # K = viewpoint_camera.K.to(viewmat.device)
    # bg_color = bg_color[:3]
    # bg_color = torch.concat([bg_color, bg_color, bg_color], dim=-1)

    # trbfcenter = dyn_pc.get_trbfcenter
    # trbfdistanceoffset = (3/11) * pointtimes - trbfcenter
    # tforpoly = trbfdistanceoffset.detach()
    # rotations = dyn_pc.get_rotation_dy(rotations, tforpoly)

    # control_xyz = dyn_pc.get_control_xyz.cuda()
    # curr_time = torch.tensor((3/11)).cuda()
    # deform_means3D = interpolate_cubic_hermite(
    #     control_xyz.permute(0, 2, 1),
    #     curr_time[None, None].expand(control_xyz.shape[0], 3, 1),
    #     N=dyn_pc.current_control_num,
    # )
    # means3D = deform_means3D * dyn_pc.deform_spatial_scale

    # # Apply activations
    # scales = dyn_pc.scaling_activation(scales)
    # rotations = dyn_pc.rotation_activation(rotations)
    # colors_precomp_2 = dyn_pc.get_features(tforpoly)
    # means3D_final_2, scales_final_2, rotations_final_2, opacity_final_2 = means3D, scales, rotations, opacity
    
    # means3D_final_2 = means3D_final_2[valid_index]
    # scales_final_2 = scales_final_2[valid_index]
    # rotations_final_2 = rotations_final_2[valid_index]
    # opacity_final_2 = opacity_final_2[valid_index]
    # colors_precomp_2 = colors_precomp_2[valid_index]
    
    # means3D_final = torch.cat((means3D_final, means3D_final_2), 0)
    # scales_final = torch.cat((scales_final, scales_final_2), 0)
    # rotations_final = torch.cat((rotations_final, rotations_final_2), 0)
    # opacity_final = torch.cat((opacity_final, opacity_final_2), 0)
    # colors_precomp_final = torch.cat((colors_precomp_final, colors_precomp_2), 0)
    

    rendered_image, _, info = rasterization(
        means=means3D_final,
        quats=rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=colors_precomp_final,
        backgrounds=bg_color[None],
        viewmats=viewmat[None].detach(),  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB+ED",
        absgrad=True,
    )

    depth = rendered_image[..., -1]
    rendered_image = rendered_image[..., :-1].permute(0, 3, 1, 2)
    radii = info["radii"].squeeze(0)

    # info["means2d"].retain_grad()
    try:
        info["means2d"].retain_grad()
    except:
        pass

    # rendered_image = torch.cat((rendered_image1, rendered_image2, rendered_image3), dim=0)
    rendered_image = dyn_pc.rgbdecoder(rendered_image, viewpoint_camera.cam_ray)
    rendered_image = rendered_image.squeeze(0)

    d_alpha, _, _ = rasterization(
        means=means3D_final,
        quats=rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=torch.cat([torch.zeros(no_stat_gs, 1), torch.ones(means3D_final.shape[0] - no_stat_gs, 1)], dim=0).cuda(),
        backgrounds=bg_color[0:1][None],
        viewmats=viewmat[None].detach(),  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB",
    )
    d_alpha = d_alpha[..., 0]

    s_rendered_image = None
    s_depth = None
    s_alpha = None
    if get_static:
        s_rendered_image, _, _ = rasterization(
            means=smeans3D_final,
            quats=stat_rotations,
            scales=stat_scales,
            opacities=stat_opacity.squeeze(-1),
            colors=stat_colors_precomp,
            backgrounds=bg_color[None],
            viewmats=viewmat[None],  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB+ED",
        )
        s_depth = s_rendered_image[..., -1]
        s_rendered_image = s_rendered_image[..., :-1].permute(0, 3, 1, 2)
        s_rendered_image = dyn_pc.rgbdecoder(s_rendered_image, viewpoint_camera.cam_ray)
        s_rendered_image = s_rendered_image.squeeze(0)

        s_alpha, _, _ = rasterization(
            means=smeans3D_final,
            quats=stat_rotations,
            scales=stat_scales,
            opacities=stat_opacity.squeeze(-1),
            colors=torch.ones(stat_colors_precomp.shape[0], 1).cuda(),
            backgrounds=bg_color[0:1][None],
            viewmats=viewmat[None].detach(),  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB",
        )
        s_alpha = s_alpha[..., 0]

    return {
        # "valid_index":valid_index,
        "render": rendered_image,
        "depth": depth,
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "s_render": s_rendered_image,
        "s_depth": s_depth,
        "s_alpha": s_alpha,
        "d_render": d_image if get_dynamic else None,
        "d_depth": d_depth if get_dynamic else None,
        "d_alpha": d_alpha,
        "d_means3d": dmeans3D_final if get_dynamic else None,
    }

def render_static(
    viewpoint_camera,
    stat_pc: GaussianModel,
    dyn_pc: GaussianModel,
    bg_color: torch.Tensor,
    get_static=False,
    get_dynamic=False,
):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    # Get stat variables
    stat_means3D = stat_pc.get_xyz
    no_stat_gs = stat_means3D.shape[0]
    stat_opacity = stat_pc.get_opacity
    stat_colors_precomp = stat_pc.get_features_static
    stat_scales = stat_pc.get_scaling
    stat_rotations = stat_pc.get_rotation_stat

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(stat_means3D.device)
    K = viewpoint_camera.K.to(viewmat.device)
    bg_color = bg_color[:3]
    bg_color = torch.concat([bg_color, bg_color, bg_color], dim=-1)

    smeans3D_final, sscales_final, srotations_final, sopacity_final = (
        stat_means3D,
        stat_scales,
        stat_rotations,
        stat_opacity,
    )

    # Combine stat and dyn gaussians
    means3D_final = smeans3D_final
    scales_final = sscales_final
    rotations_final = srotations_final
    opacity_final = sopacity_final
    colors_precomp_final = stat_colors_precomp

    rendered_image, _, info = rasterization(
        means=means3D_final,
        quats=rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=colors_precomp_final,
        backgrounds=bg_color[None],
        viewmats=viewmat[None].detach(),  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB+ED",
        absgrad=True,
    )

    depth = rendered_image[..., -1]
    rendered_image = rendered_image[..., :-1].permute(0, 3, 1, 2)
    radii = info["radii"].squeeze(0)

    # info["means2d"].retain_grad()
    try:
        info["means2d"].retain_grad()
    except:
        pass

    # rendered_image = torch.cat((rendered_image1, rendered_image2, rendered_image3), dim=0)
    rendered_image = dyn_pc.rgbdecoder(rendered_image, viewpoint_camera.cam_ray)
    rendered_image = rendered_image.squeeze(0)

    s_rendered_image = None
    s_depth = None
    s_alpha = None
    if get_static:
        s_rendered_image, _, _ = rasterization(
            means=smeans3D_final,
            quats=stat_rotations,
            scales=stat_scales,
            opacities=stat_opacity.squeeze(-1),
            colors=stat_colors_precomp,
            backgrounds=bg_color[None],
            viewmats=viewmat[None],  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB+ED",
        )
        s_depth = s_rendered_image[..., -1]
        s_rendered_image = s_rendered_image[..., :-1].permute(0, 3, 1, 2)
        s_rendered_image = dyn_pc.rgbdecoder(s_rendered_image, viewpoint_camera.cam_ray)
        s_rendered_image = s_rendered_image.squeeze(0)

        s_alpha, _, _ = rasterization(
            means=smeans3D_final,
            quats=stat_rotations,
            scales=stat_scales,
            opacities=stat_opacity.squeeze(-1),
            colors=torch.ones(stat_colors_precomp.shape[0], 1).cuda(),
            backgrounds=bg_color[0:1][None],
            viewmats=viewmat[None].detach(),  # [C, 4, 4]
            Ks=K[None],  # [C, 3, 3]
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            render_mode="RGB",
        )
        s_alpha = s_alpha[..., 0]

    return {
        "render": rendered_image,
        "depth": depth,
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "s_render": s_rendered_image,
        "s_depth": s_depth,
        "s_alpha": s_alpha,
        "d_render": None,
        "d_depth": None,
        "d_alpha":  None,
        "d_means3d": None,
    }


def render_infer(viewpoint_camera,
    stat_pc: GaussianModel,
    dyn_pc: GaussianModel,
    bg_color: torch.Tensor,
    ):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # Get dyn variables
    means3D = dyn_pc.get_xyz.detach()
    no_dyn_gs = means3D.shape[0]
    scales = dyn_pc._scaling
    rotations = dyn_pc._rotation
    opacity = dyn_pc.get_opacity

    # Get stat variables
    stat_means3D = stat_pc.get_xyz
    no_stat_gs = stat_means3D.shape[0]
    stat_opacity = stat_pc.get_opacity
    stat_colors_precomp = stat_pc.get_features_static
    stat_scales = stat_pc.get_scaling
    stat_rotations = stat_pc.get_rotation_stat

    pointtimes = (
        torch.ones((dyn_pc.get_xyz.shape[0], 1), dtype=dyn_pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    )  #

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device)
    K = viewpoint_camera.K.to(viewmat.device)
    bg_color = bg_color[:3]
    bg_color = torch.concat([bg_color, bg_color, bg_color], dim=-1)

    trbfdistanceoffset = viewpoint_camera.time * pointtimes
    tforpoly = trbfdistanceoffset.detach()
    rotations = dyn_pc.get_rotation_dy(rotations, tforpoly)

    control_xyz = dyn_pc.flat_control_xyz.cuda()
    # start.record()
    deform_means3D = interpolate_cubic_hermite_infer(control_xyz, torch.tensor(viewpoint_camera.time).cuda()[None].expand(means3D.shape[0], 3), N=dyn_pc.current_control_num, index_offset=dyn_pc.index_offset)
    # end.record()
    # torch.cuda.synchronize()
    # print('spline time:', start.elapsed_time(end) / means3D.shape[0])
    # duration = start.elapsed_time(end) / means3D.shape[0]

    means3D = deform_means3D * 1e-2
    # Apply activations
    scales = dyn_pc.scaling_activation(scales)
    rotations = dyn_pc.rotation_activation(rotations)
    colors_precomp = dyn_pc.get_features(viewpoint_camera.time)

    smeans3D_final, sscales_final, srotations_final, sopacity_final = stat_means3D, stat_scales, stat_rotations, stat_opacity
    means3D_final, scales_final, rotations_final, opacity_final = means3D, scales, rotations, opacity


    # Combine stat and dyn gaussians
    means3D_final = torch.cat((smeans3D_final, means3D_final), 0)
    scales_final = torch.cat((sscales_final, scales_final), 0)
    rotations_final = torch.cat((srotations_final, rotations_final), 0)
    opacity_final = torch.cat((sopacity_final, opacity_final), 0)
    colors_precomp_final = torch.cat((stat_colors_precomp, colors_precomp), 0)

    rendered_image, _, _ = rasterization(
        means=means3D_final,
        quats=rotations_final,
        scales=scales_final,
        opacities=opacity_final.squeeze(-1),
        colors=colors_precomp_final,
        backgrounds=bg_color[None],
        viewmats=viewmat[None].detach(),  # [C, 4, 4]
        Ks=K[None],  # [C, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        render_mode="RGB",
    )
    
    rendered_image = rendered_image.permute(0, 3, 1, 2)
    rendered_image = dyn_pc.rgbdecoder(rendered_image, viewpoint_camera.cam_ray)
    rendered_image = rendered_image.squeeze(0)

    return {
        "render": rendered_image,
    }
