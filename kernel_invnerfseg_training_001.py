#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 00:03:33 2025

@author: nibio
"""



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


"""
NeRF implementation that combines many recent advancements.
"""

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
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
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
        # Create a mask for object pixels (assumes background is exactly 0)
        # mask = (gt_rgb.sum(dim=-1, keepdim=True) > 0).float()  # Shape: [B, H, W, 1]
        # Compute masked RGB loss (only where mask == 1)
        # masked_loss = self.rgb_loss(gt_rgb, pred_rgb) * mask  # Element-wise loss
        # loss_dict["rgb_loss"] = masked_loss.sum() / (mask.sum() + 1e-8)  # Normalize by valid pixels
        loss_dict["rgb_loss"] = 1*self.rgb_loss(gt_rgb, pred_rgb)
        
        # loss_dict["rgb_loss"] = 1*self.rgb_loss(gt_rgb, pred_rgb)
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

# Camera setup
@dataclass
class Camera:
    camera_to_worlds: Tensor  # Shape: [num_cameras, 3, 4]
    fx: Tensor  # Shape: [num_cameras, 1]
    fy: Tensor  # Shape: [num_cameras, 1]
    cx: Tensor  # Shape: [num_cameras, 1]
    cy: Tensor  # Shape: [num_cameras, 1]

    def generate_rays(self, coords: Tensor, camera_index: int) -> dict:
        """
        Generate rays for a single image given its pixel coordinates and camera index.
        """
        c2w = self.camera_to_worlds[camera_index]  # [3, 4]
        fx, fy = self.fx[0].item(), self.fy[0].item()
        cx, cy = self.cx[0].item(), self.cy[0].item()

        # Normalize coordinates
        x = (coords[:, 0] - cx) / fx
        y = (coords[:, 1] - cy) / fy
        directions = torch.stack([x, y, torch.full_like(x, -1.0)], dim=-1)  # [num_rays, 3]

        # Rotate directions to world space
        rotation = c2w[:3, :3]  # [3, 3]
        directions = torch.matmul(directions, rotation.T)  # [num_rays, 3]
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)  # Normalize

        # Set origins to camera position in world space
        origins = c2w[:3, 3].expand_as(directions)  # [num_rays, 3]

        return {"origins": origins, "directions": directions}

    def generate_rays_batch(self, coords: Tensor, camera_indices: Tensor) -> dict:
        """
        Generate rays for a batch of images.

        Args:
            coords (Tensor): Pixel coordinates for all images in the batch. 
                             Shape: [batch_size, num_pixels, 2].
            camera_indices (Tensor): Indices of cameras for each image in the batch. 
                                      Shape: [batch_size].

        Returns:
            dict: Contains batched 'origins' and 'directions'.
                  - origins: [batch_size, num_pixels, 3]
                  - directions: [batch_size, num_pixels, 3]
        """
        batch_size, num_pixels, _ = coords.shape
        origins = []
        directions = []

        for i in range(batch_size):
            rays = self.generate_rays(coords[i], camera_indices[i].item())
            origins.append(rays["origins"])
            directions.append(rays["directions"])

        origins = torch.stack(origins, dim=0)  # [batch_size, num_pixels, 3]
        directions = torch.stack(directions, dim=0)  # [batch_size, num_pixels, 3]

        return {"origins": origins, "directions": directions}
# Utility to convert image dimensions to coordinates
def image_to_coords(batch_images: Tensor) -> Tensor:
    """
    Converts a batch of images into pixel coordinates.

    Args:
        batch_images (Tensor): Images of shape (b, h, w, c).

    Returns:
        Tensor: Pixel coordinates of shape (b, h*w, 2).
    """
    b, h, w, c = batch_images.shape
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).reshape(1, h * w, 2)  # Shape: [1, h*w, 2]
    coords = coords.expand(b, -1, -1)  # Shape: [b, h*w, 2]
    return coords.float()






import torch

def save_checkpoint_multiple(model, optimizers, epoch, checkpoint_path="checkpoint.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizers_state_dict": {
            key: (
                [optimizer.state_dict() for optimizer in optimizer_list]
                if isinstance(optimizer_list, list)
                else optimizer_list.state_dict()
            )
            for key, optimizer_list in optimizers.items()
        },
        "epoch": epoch,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

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

def extract_poses_only(transforms_data):
    poses = []
    for frame in transforms_data['frames']:
        # Load the 4x4 transformation matrix for the current frame
        transform_matrix = np.array(frame['transform_matrix'])[:3,:]
        poses.append(transform_matrix)
    return poses

def load_transforms(file_path):
    """Load camera poses from a transforms.json file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
def extract_camera_poses_and_directions(transforms_data):
    """
    Extract camera poses and viewing directions from transforms.json data.

    Parameters:
    - transforms_data (dict): Data loaded from transforms.json

    Returns:
    - poses (list): List of 4x4 camera-to-world transformation matrices.
    - directions (list): List of 3D unit vectors representing camera directions.
    """
    poses = []
    directions = []
    img_id_ls = []
    for frame in transforms_data['frames']:
        # Load the 4x4 transformation matrix for the current frame
        transform_matrix = np.array(frame['transform_matrix'])
        img_id = frame["file_path"].split("/")[-1]
        # Extract the camera's position (translation) and rotation part
        position = transform_matrix[:3, 3]
        rotation_matrix = transform_matrix[:3, :3]

        # Compute the camera direction (forward vector)
        # Camera's forward vector is the negative Z-axis in the camera coordinate system
        camera_forward = -rotation_matrix[:, 2]  # Take negative Z-axis

        # Normalize the direction vector to make it a unit vector
        camera_forward_normalized = camera_forward / np.linalg.norm(camera_forward)

        # Store pose and direction
        poses.append(transform_matrix)
        directions.append(camera_forward_normalized.tolist())

        img_id_ls.append(img_id)
    return poses, directions, img_id_ls



# Clear gradients for all optimizers
def zero_grad_all(optimizers):
    for key, optimizer in optimizers.items():
        if isinstance(optimizer, list):  # For proposal networks (list of optimizers)
            for sub_optimizer in optimizer:
                sub_optimizer.zero_grad()
        else:
            optimizer.zero_grad()


# Perform a step for all optimizers
def step_all_optimizers(optimizers):
    for key, optimizer in optimizers.items():
        if isinstance(optimizer, list):  # For proposal networks
            for sub_optimizer in optimizer:
                sub_optimizer.step()
        else:
            optimizer.step()


# Step all schedulers after optimizer update
def step_all_schedulers(schedulers):
    for key, scheduler in schedulers.items():
        if isinstance(scheduler, list):  # For proposal networks
            for sub_scheduler in scheduler:
                sub_scheduler.step()
        else:
            scheduler.step()


base_dir = 'cob_new'
folder_ls = [
    name for name in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, name)) and name.endswith("_out")
]

for f_i in range(len(folder_ls)):
    folder_i = folder_ls[f_i]
    folder_path = os.path.join(base_dir, folder_i)

    file_path = os.path.join(folder_path, 'transforms.json')  # Update with the path to your file
    transforms_data = load_transforms(file_path)
    # Extract camera poses and directions
    poses_out_0, directions_out_0, img_ls_out_0 = extract_camera_poses_and_directions(transforms_data)
    poses_out_0 = torch.tensor(np.array(poses_out_0)).float()
    directions_out = torch.tensor(np.array(directions_out_0)).float()
    #####
    poses_out, global_transform = auto_orient_and_center_poses(poses_out_0)
    img_ls_out = img_ls_out_0
    ##
    scale_factor = 1.0 
    scale_factor /= float(torch.max(torch.abs(poses_out[:,:3,3])))
    # Normalize positions
    poses_out[:,:3,3] *= scale_factor
    
    
    import cv2
    
    images_out = []
    for image_name in img_ls_out:
        img_path = os.path.join(folder_path, "images", image_name)
        # Convert quaternion to rotation matrix
        sc_size = 1
        img_i = cv2.imread(img_path)[:,:,[2,1,0]]
        img_i_n = cv2.resize(img_i, (int(img_i.shape[1]/sc_size), int(img_i.shape[0]/sc_size)),interpolation = cv2.INTER_LINEAR)
        #
        images_out.append(img_i_n)
        # masks_out.append(mask_i_n)
    #
    images_out = np.array(images_out)
    ###
    
    nerfcfig = NerfactoModelConfig()
    
    # Create the SceneBox
    scale_factor = 1
    ab = 1
    min_bound = torch.tensor([-ab*scale_factor, -ab*scale_factor, -ab*scale_factor])
    max_bound = torch.tensor([ab*scale_factor, ab*scale_factor, ab*scale_factor])
    
    scene_box = SceneBox(aabb=torch.stack([1*min_bound, 1*max_bound], dim=0))
    ##
    
    ##
    near_plane = 0.01#nerfcfig.near_plane  # e.g., 0.05
    far_plane = 100 #nerfcfig.far_plane    # e.g., 10.0
    
    nerfmodel = NerfactoModel(nerfcfig, scene_box, 
                              num_train_data=images_out.shape[0], training = True).cuda()
    
    ##
    num_rays = images_out.reshape(-1,3).shape[0]
    #                      Load checkpoint if available
    checkpoint_path = os.path.join(base_dir,
                                   "best_checkpoint_nerfacto_cobNew{}_RGB_pc_001.pth".format(folder_i))
    load_checkpoint_multiple(checkpoint_path, nerfmodel)
    ##  reduce the learning rate during fine tuning
    field_optimizer = torch.optim.Adam(
        params=nerfmodel.field.parameters(),  # All parameters in the field
        lr=1e-2,  # Default learning rate, 0.01
        eps=1e-8,  # Default epsilon
        weight_decay=0  # Default weight decay
    )

    ###
    proposal_optimizers = []    
    for proposal_net in nerfmodel.proposal_networks:
        optimizer = torch.optim.Adam(
            params=proposal_net.parameters(),
            lr=1e-4,  # Default learning rate, 0.01
            eps=1e-8,  # Default epsilon
            weight_decay=0  # Default weight decay
        )
        proposal_optimizers.append(optimizer)
    
    ###
    camera_optimizer = torch.optim.Adam(
        params=nerfmodel.camera_optimizer.parameters(),  # Parameters related to camera optimization
        lr=1e-4,  # Default learning rate for camera parameters, 0.001
        eps=1e-8,  # Default epsilon
        weight_decay=0  # Default weight decay
    )

    
    ##################################
    optimizers = {
        "field": field_optimizer,
        "camera": camera_optimizer,
        "proposal_networks": proposal_optimizers,
    }
    
    images = torch.tensor(images_out)
    height, width = images.shape[1:3]
    ##
    from pathlib import Path
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio
    
    data_config = NerfstudioDataParserConfig()
    data_config.data = Path(folder_path)
    data_config.downscale_factor = 1
    data_config.train_split_fraction = 0.8
    # data_config.eval_mode = "all"
    #
    shi = Nerfstudio(data_config)
    ###
    
    from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig, ParallelDataManager
    
    dataManager_config = ParallelDataManagerConfig()
    dataManager_config.dataparser.data = Path(folder_path)
    # dataManager_config.dataparser.data = 0.5
    dataManager_config.dataparser = data_config
    dataManager_config.train_num_rays_per_batch = 1024*5
    dataManager_config.eval_num_rays_per_batch = 1024*5
    dataManager_config.camera_res_scale_factor = 1
    
    dataManager = ParallelDataManager(dataManager_config)

    shi_val = shi._generate_dataparser_outputs('train')
    shi_val.image_filenames
    numbers_eval = [int(path.stem.split("_")[-1]) for path in shi_val.image_filenames]

    
    import random
    best_loss = np.inf
    best_loss_i = np.inf
    loss_out_final_i = 0
    # Process each batch
    for epoch in range(2200):
        # randomly sample rays and directions for model training
        nerfmodel.train()  # Set model to training mode
        print("current epoch {}".format(epoch))
        ray_bundle_train, batch_train = dataManager.next_train(epoch)
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
        batch = {}
        batch['image'] = batch_train['image'][:,:3].float().cuda()
        # Feed the batch into the model
        # output = nerfmodel(ray_bundle)
        outputs = nerfmodel.get_outputs(ray_bundle)
        # outputs
        # outputs["rgb"].shape
        # outputs["accumulation"].shape
        #
        metric_dic = nerfmodel.get_metrics_dict(outputs, batch)
        loss_out = nerfmodel.get_loss_dict(outputs, batch, metric_dic)
        #
        total_loss = sum(loss for loss in loss_out.values())
        
        # Perform backward pass to compute gradients
        total_loss.backward()
    
        #
        step_all_optimizers(optimizers)
        zero_grad_all(optimizers)
    
        #
        # torch.nn.utils.clip_grad_norm_(nerfmodel.parameters(), max_norm=1.0)
        #
        # Optionally print or log individual losses for monitoring
        for loss_name, loss_value in loss_out.items():
            print(f"{loss_name}: {loss_value.item()}")
        
        # Log the total loss as well
        print(f"Total Loss: {total_loss.item()}")
        ##
        loss_out_final_i_0 = loss_out['rgb_loss'].item()
        loss_out_final_i += loss_out_final_i_0
        ##
        if loss_out_final_i_0 < best_loss_i:
            best_loss_i = loss_out_final_i_0
            save_checkpoint_multiple(nerfmodel, optimizers, 
                                     epoch, "best_checkpoint_nerfacto_cobNew{}_RGB_pc_001.pth".format(folder_i))
            ## change the model name segmentation during fine tuning
            # save_checkpoint_multiple(nerfmodel, optimizers, 
            #                          epoch, "best_checkpoint_nerfacto_cobNew{}Seg_RGB_pc_001.pth".format(folder_i))
        # Append the output for the current batch to the results list
        if (epoch +1)%50 ==0:
            print("folder_path = {}".format(folder_path))

            cur_lr = optimizer.param_groups[0]['lr']
            print("current learning rate = {}".format(cur_lr))
            #
            nerfmodel.eval()
            with torch.no_grad():
                ray_bundle_test, batch_test = dataManager.next_eval_image(epoch)
                eval_rays = ray_bundle_test.generate_rays(0)#origins,directions,pixel area,camera indices
                print("eval_rays camera indice = {}".format(batch_test['image_idx']))
                new_images_n = batch_test['image'][:,:,:3].reshape(-1,3)
                test_positions = eval_rays.origins.reshape(-1, 3)
                test_directions = eval_rays.directions.reshape(-1,3)
                test_nears = torch.full((test_positions.shape[0], 1), near_plane)
                test_fars  =  torch.full((test_positions.shape[0], 1), far_plane)
                test_area = torch.full((test_positions.shape[0], 1), 1.0)
                test_indices = torch.full((test_positions.shape[0], 1), 0.0).long()
                #
                pred_rgb_out = []
                pred_depth_out = []
                psnr_val_0 = 0
                for w_ii in range(height):
                    test_bundle = RayBundle(
                        origins=test_positions[width*w_ii:(w_ii+1)*width,:].cuda(),
                        directions=test_directions[width*w_ii:(w_ii+1)*width,:].cuda(),
                        nears=test_nears[width*w_ii:(w_ii+1)*width,:].cuda(),
                        fars=test_fars[width*w_ii:(w_ii+1)*width,:].cuda(),
                        pixel_area=test_area[width*w_ii:(w_ii+1)*width,:].cuda(),
                        camera_indices=test_indices[width*w_ii:(w_ii+1)*width,:].cuda()  # All zeros for single camera
                    )
                    batch_test_i = {}
                    batch_test_i['image'] = new_images_n[width*w_ii:(w_ii+1)*width,:].cuda()
                    ##
                    outputs_test = nerfmodel.get_outputs(test_bundle)
                    metric_test = nerfmodel.get_metrics_dict(outputs_test, batch_test_i)
                    psnr_val_0 += metric_test['psnr'].item()
                    pred_rgb = outputs_test['rgb'].clone().detach().cpu().reshape(1,width,3)#.reshape(image_size[1], image_size[0], 3)
                    pred_depth = outputs_test['depth'].clone().detach().cpu().reshape(1,width)#.reshape(image_size[1], image_size[0], 3)
                    pred_rgb_out.append(pred_rgb)
                    pred_depth_out.append(pred_depth)
                #A
                pred_rgb_out_n = torch.cat(pred_rgb_out,0)
                pred_depth_out_n = torch.cat(pred_depth_out,0)
                pred_depth_out_n = (pred_depth_out_n-pred_depth_out_n.min())/(pred_depth_out_n.max() - pred_depth_out_n.min())
                gt_rgb = batch_test['image']#.clone().cpu().detach().numpy()
                ##
                fig, ax = plt.subplots(1, 3, figsize=(10,4))
                ax[0].imshow(pred_rgb_out_n)
                ax[1].imshow(pred_depth_out_n)
                ax[2].imshow(gt_rgb)
                plt.show()
                #
                # psnr_val = metric_test['psnr'].item()
                psnr_val = psnr_val_0/height
                print("psnr_val {}".format(psnr_val))








