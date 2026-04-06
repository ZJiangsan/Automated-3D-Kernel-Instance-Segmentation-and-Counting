#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 09:47:42 2025

@author: nibio
"""




import os


from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from pathlib import Path
import torch
from jaxtyping import Float
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
import os


# Nerfacto-Inspired Implementation in Python (Updated)
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nerfstudio.data.scene_box import SceneBox
import torch
import torch.optim as optim
import torch.nn.functional as F


## Nerfacto MLP based on nerfstudio components


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer, SemanticRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.data.scene_box import SceneBox

from nerfstudio.cameras.camera_utils import auto_orient_and_center_poses



@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 128
    """Dimension of hidden layers"""
    hidden_dim_color: int = 128
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 128
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    base_res: int = 16
    """Resolution of the base grid for the hashgrid."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    features_per_level: int = 2
    """How many hashgrid features per level"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_appearance_embedding: bool = True
    """Whether to use an appearance embedding."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = True
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    appearance_embed_dim: int = 128
    """Dimension of the appearance embedding."""
    average_init_density: float = 1.0
    """Average initial density output from MLP. """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Config of the camera optimizer to use"""
    pass_semantic_gradients: bool = False
    """Whether to pass semantic gradients."""
    use_semantics: bool = True
    """Whether to use semantics."""
    num_semantic_classes: int = 1
    """Number of semantic classes."""
    semantic_loss_weight: float = 1.0
    """Number of semantic classes."""





class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    average_init_density=self.config.average_init_density,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.step = 0
        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
        # print("field_outputs[FieldHeadNames.RGB] {}".format(field_outputs[FieldHeadNames.RGB].shape))
        # print("weights {}".format(weights.shape))
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        # print("rgb {}".format(rgb.shape))
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs
    
    def get_export_outputs(self, ray_bundle: RayBundle):
        outputs = {}
        
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        outputs["rgb"] = field_outputs[FieldHeadNames.RGB]

        outputs['point_location'] = ray_samples.frustums.get_positions()
        # outputs["semantics"] = field_outputs[FieldHeadNames.SEMANTICS][..., 0]
        outputs["density"] = field_outputs[FieldHeadNames.DENSITY][..., 0]
        ##
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        outputs["normals"] = self.normals_shader(normals)

        return outputs
    
    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)  # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        loss_dict["rgb_loss"] = 1*self.rgb_loss(gt_rgb, pred_rgb)
        
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict


from typing import Optional, Tuple, List, Union, Callable

## 5. Ray Sampling
from typing import Tuple

from dataclasses import dataclass
from typing import Optional, Union
from torch import Tensor
from typing import List
from dataclasses import dataclass
import torch
from torch import Tensor
import numpy as np


import torch



def load_checkpoint_multiple(checkpoint_path, model):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load the model's state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded successfully.")

    # Optionally, you can print or log other details from the checkpoint
    epoch = checkpoint["epoch"]
    print(f"Resuming training from epoch {epoch}.")

    # Return the epoch to continue training
    return epoch



import json

def load_transforms(file_path):
    """Load camera poses from a transforms.json file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data



from pathlib import Path
from PIL import Image
##
from pathlib import Path
import random
#
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig, ParallelDataManager
    

import cv2


base_dir = 'cob_new'
folder_ls = [
    name for name in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, name)) and name.endswith("_out")
]


for f_i in range(len(folder_ls)):
    folder_i = folder_ls[f_i]
    print("folder_i = {}".format(folder_i))
    folder_path = os.path.join(base_dir, folder_i)

    point_path = os.path.join(folder_path, '{}SEG_density_pc_001.ply'.format(folder_i))
    

    
    data_config = NerfstudioDataParserConfig()
    data_config.data = Path(folder_path)
    data_config.downscale_factor = 1
    data_config.train_split_fraction = 0.99
    # data_config.eval_mode = 'all'
    
    
    dataManager_config = ParallelDataManagerConfig()
    dataManager_config.dataparser.data = Path(folder_path)
    dataManager_config.dataparser = data_config
    dataManager_config.train_num_rays_per_batch = 1024*10
    dataManager_config.eval_num_rays_per_batch = 1024*10
    dataManager_config.camera_res_scale_factor = 1
    
    dataManager = ParallelDataManager(dataManager_config)
        
    ###
    img_num = len(os.listdir(os.path.join(folder_path, "images")))
    nerfcfig = NerfactoModelConfig()
    # Create the SceneBox
    scale_factor = 1
    ab = 1
    min_bound = torch.tensor([-ab*scale_factor, -ab*scale_factor, -ab*scale_factor])
    max_bound = torch.tensor([ab*scale_factor, ab*scale_factor, ab*scale_factor])
    
    scene_box = SceneBox(aabb=torch.stack([1*min_bound, 1*max_bound], dim=0))
    ##
    
    ##
    near_plane = 0.01
    far_plane = 100
    
    nerfmodel = NerfactoModel(nerfcfig, scene_box, num_train_data=img_num, training = True).cuda()
    
    ##
    #                      Load checkpoint if available
    checkpoint_path = os.path.join(base_dir, "best_checkpoint_nerfacto_cobNew{}Seg_RGB_pc_001.pth".format(folder_i))
    
    load_checkpoint_multiple(checkpoint_path, nerfmodel)
    ##    
    progress = Progress(
        TextColumn(":cloud: Computing Point Cloud :cloud:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
        console=CONSOLE,
    )
        
    points_sem = []
    points_only_sem = []
    points_den = []
    points_den_sem = []
    points_sem_colormap = []
    color_semantics = []
    color_only_semantics = []
    color_semantics_colormap = []
    densities = []
    densities_sem = []
    
    rgb_flag = True
    # sample_points_along_edge = num_points_per_side # num_points_per_side
    with progress as progress_bar:
        task = progress_bar.add_task("Generating Point Cloud", total=1e+6)
        while not progress_bar.finished:
            xx = 0
            with torch.no_grad():
                ray_bundle_train, batch_train = dataManager.next_train(xx)
                # Select the corresponding slice of rays for the current batch
                batch_nears = (torch.ones(ray_bundle_train.origins.shape[0])*near_plane).reshape(-1, 1).float().cuda()
                batch_fars = (torch.ones(ray_bundle_train.origins.shape[0])*far_plane).reshape(-1, 1).float().cuda()
                #
                # Create the ray bundle for the current batch
                ray_bundle = RayBundle(
                    origins=ray_bundle_train.origins.float().cuda(),
                    directions=ray_bundle_train.directions.float().cuda(),
                    nears=batch_nears,
                    fars=batch_fars,
                    pixel_area=ray_bundle_train.pixel_area.float().cuda(),
                    camera_indices=ray_bundle_train.camera_indices.long().cuda()  # All zeros for single camera
                )
                # Feed the batch into the model
                outputs = nerfmodel.get_export_outputs(ray_bundle)
                ##
                outputs.keys()
                # set a bounding box
                min_bound = torch.tensor([-0.5, -0.5, -0.7]).cuda()  # Adjust based on object
                max_bound = torch.tensor([0.5, 0.5, 0.9]).cuda()
                # Sampled volume points
                sampled_point_position = outputs['point_location']
                points_3d_0 = sampled_point_position.reshape((-1, 3))
                # Apply bounding box filtering
                mask_bbox = ((points_3d_0 >= min_bound) & (points_3d_0 <= max_bound)).all(dim=1)
                points_3d = points_3d_0[mask_bbox]
                
                # Semantic & Density value
                density_0 = outputs['density'].reshape((-1, 1)).repeat((1, 3))
                rgb_0 = outputs['rgb'].reshape((-1, 3))
                ###
                rgb = rgb_0[mask_bbox]
                density = density_0[mask_bbox]
                # Mask irrelevant semantic masks and density values
                mask_den = density >= 70  # 10

                points_3d_density = points_3d[mask_den.sum(dim=1).to(bool)]
                if rgb_flag:
                    density_color = rgb[mask_den.sum(dim=1).to(bool)]
                else:
                    density_color = density[mask_den.sum(dim=1).to(bool)]
                density_color = torch.hstack(
                    [density_color, torch.sigmoid(density[mask_den.sum(dim=1).to(bool)][:, 0]).unsqueeze(-1)])
                #
                points_den.append(points_3d_density.cpu())
    
                densities.append(density_color.cpu())
                #
                xx+=1
                #
                e_i_ls = [30, 0,0]
                a_i_ls = [0, 90, 180]
                for e_a_i in zip(e_i_ls, a_i_ls):
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')            
                    # Extract X, Y, Z coordinates
                    x, y, z = points_3d_density.cpu()[:, 0], points_3d_density.cpu()[:, 1], points_3d_density.cpu()[:, 2]
                    # Scatter plot
                    ax.scatter(x, y, z, c=density_color.cpu(), s=1)  # Color by Z value
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    plt.title("3D rgb Point Cloud")
                    ax.view_init(elev=e_a_i[0], azim=e_a_i[1])
                    plt.show()
    
                torch.cuda.empty_cache()
                progress.advance(task, sampled_point_position.shape[0])
    
        pcd_list = {}
        # Density
        points_den = torch.cat(points_den, dim=0)
        density_rgb = torch.cat(densities, dim=0)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_den.double().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(density_rgb.double().cpu().numpy()[:, :3])
    
        pcd_list.update({'density': {
                'pcd': pcd,
                'path': str(point_path)
            }})
    
    
    #####  save the point cloud
    for pcd_name in pcd_list.keys():
        pcd = pcd_list[pcd_name]['pcd']
        pcd_path = pcd_list[pcd_name]['path']
        o3d.io.write_point_cloud(pcd_path, pcd)












