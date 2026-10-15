import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.necks.fpn import FPN
from typing import List
from mmrotate.registry import MODELS


class DAFFusion(nn.Module):
    """
    Directional-Aware Frequency Fusion (DAFF).

    Key improvements for O²-RTDETR:
      - Vectorized FFT & Spatial Rotation (No for-loops, high performance).
      - Energy-guided Frequency Attention (Suppress background noise for Query-based decoder).
      - Residual Frequency Fusion (RFF) with post-refinement convolution.
    """

    def __init__(
            self,
            in_channels: int = 256,
            m: int = 7,
            c_mid: int = 16,
            eps: float = 1e-8,
            layer_scale_init_value: float = 1e-5,
            use_frequency_response: bool = True,
            use_direction_alignment: bool = True,
            direction_estimator: str = 'fft',
    ):
        super().__init__()
        self.m = m
        self.c_mid = c_mid
        self.eps = eps
        self.in_channels = in_channels
        self.use_frequency_response = use_frequency_response
        self.use_direction_alignment = use_direction_alignment
        if direction_estimator not in {'fft', 'global'}:
            raise ValueError("direction_estimator must be 'fft' or 'global'")
        self.direction_estimator = direction_estimator
        # Only the no-estimation ablation owns this extra scalar. The default
        # 'fft' path is state-dict compatible with existing DAFF checkpoints.
        if direction_estimator == 'global':
            self.global_theta = nn.Parameter(torch.zeros(1))

        # Learnable LayerScale: per-channel scalar, initialized small
        self.layer_scale = nn.Parameter(
            torch.full((1, in_channels, 1, 1), layer_scale_init_value),
            requires_grad=True
        )

        # Dimension reduction for frequency calculation
        self.proj_low = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.proj_high = nn.Conv2d(in_channels, c_mid, kernel_size=1, bias=False)
        self.recon = nn.Conv2d(c_mid, in_channels, kernel_size=1, bias=False)

        # RT-DETR Adapter: Refinement convolution for Query-based Transformer
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        )

        self._init_freq_grids(m)

    def _init_freq_grids(self, m: int):
        h_freq = torch.fft.fftfreq(m, d=1.0) * m
        w_freq = torch.fft.fftfreq(m, d=1.0) * m
        h_grid, w_grid = torch.meshgrid(h_freq, w_freq, indexing='ij')

        rho = torch.sqrt(h_grid ** 2 + w_grid ** 2)
        theta = torch.atan2(h_grid, w_grid)
        theta = (theta + 2 * math.pi) % (2 * math.pi)

        mask = rho > self.eps
        self.register_buffer('valid_thetas', theta[mask])
        self.register_buffer('valid_rhos', rho[mask])
        self.register_buffer('mask_flat', mask.view(-1))

    def _estimate_main_direction_and_energy(self, x_local: torch.Tensor):
        """Estimate dominant orientation and energy from magnitude spectrum.
        x_local: [K, 1, m, m]
        Returns:
            theta_e: [K] Dominant angles
            max_energy: [K] Maximum energy for attention gating
        """
        K, _, m, _ = x_local.shape
        device = x_local.device

        x_fft = torch.fft.fft2(x_local.squeeze(1), norm='ortho')
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
        mag = x_fft_shifted.abs() + self.eps

        mag_flat = mag.view(K, -1)
        mag_valid = mag_flat[:, self.mask_flat]
        rho_valid = self.valid_rhos.to(device)

        weighted_energy = mag_valid * rho_valid.unsqueeze(0)

        # Extract both max index (for angle) and max value (for attention gate)
        max_energy, max_idx = torch.max(weighted_energy, dim=1)
        theta_e = self.valid_thetas.to(device)[max_idx]

        return theta_e, max_energy

    def _rotate_spatial_patch(self, patch: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        K, _, m, _ = patch.shape
        device = patch.device

        cos_t = torch.cos(theta).view(K, 1, 1)
        sin_t = torch.sin(theta).view(K, 1, 1)

        center = (m - 1) / 2.0
        rot_mat = torch.zeros(K, 2, 3, device=device)
        rot_mat[:, 0, 0] = cos_t.squeeze()
        rot_mat[:, 0, 1] = -sin_t.squeeze()
        rot_mat[:, 1, 0] = sin_t.squeeze()
        rot_mat[:, 1, 1] = cos_t.squeeze()
        rot_mat[:, 0, 2] = center - cos_t.squeeze() * center + sin_t.squeeze() * center
        rot_mat[:, 1, 2] = center - sin_t.squeeze() * center - cos_t.squeeze() * center

        grid = F.affine_grid(rot_mat, patch.size(), align_corners=False)
        rotated = F.grid_sample(patch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return rotated

    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        B, C, H_l, W_l = x_low.shape
        _, _, H_h, W_h = x_high.shape
        device = x_low.device

        # Step 1: Upsample
        if (H_h, W_h) != (H_l, W_l):
            x_high_up = F.interpolate(x_high, size=(H_l, W_l), mode='bilinear', align_corners=False)
        else:
            x_high_up = x_high

        # Step 2: Project
        xl_proj = self.proj_low(x_low)      # [B, c_mid, H_l, W_l]
        xh_proj = self.proj_high(x_high_up) # [B, c_mid, H_l, W_l]

        pad = self.m // 2
        N = H_l * W_l

        # Step 3: Unfold & Vectorize (Eliminated loop)
        xl_unfold = F.unfold(xl_proj, kernel_size=self.m, stride=1, padding=pad) # [B, c_mid*m*m, N]
        xh_unfold = F.unfold(xh_proj, kernel_size=self.m, stride=1, padding=pad)

        # Reshape to parallel batch format: [B*N*c_mid, 1, m, m]
        xl_patches = xl_unfold.view(B, self.c_mid, self.m * self.m, N).permute(0, 3, 1, 2).reshape(-1, 1, self.m, self.m)
        xh_patches = xh_unfold.view(B, self.c_mid, self.m * self.m, N).permute(0, 3, 1, 2).reshape(-1, 1, self.m, self.m)

        # Step 4: Frequency-domain Direction & Energy Estimation
        theta_low, _ = self._estimate_main_direction_and_energy(xl_patches)
        theta_high, energy_high = self._estimate_main_direction_and_energy(xh_patches)

        if self.direction_estimator == 'fft':
            theta_low_norm = torch.remainder(theta_low, math.pi)
            theta_high_norm = torch.remainder(theta_high, math.pi)
            theta_ = theta_low_norm - theta_high_norm  # [B*N*c_mid]
        else:
            # Keeps the rotation operator but removes adaptive, local
            # frequency-direction estimation. A single trained global angle
            # is a controlled replacement, rather than a no-op rotation.
            theta_ = math.pi * torch.tanh(self.global_theta).expand_as(energy_high)

        # Step 5: Spatial Alignment
        if self.use_direction_alignment:
            xh_rotated = self._rotate_spatial_patch(xh_patches, theta_)
        else:
            xh_rotated = xh_patches

        # Step 6: Energy-guided Frequency Attention (Crucial for Query-Based detection)
        # Suppress noise in non-directional/background regions
        if self.use_frequency_response:
            gate = torch.sigmoid(energy_high).view(-1, 1, 1, 1)
        else:
            gate = 1.0
        xh_rotated = xh_rotated * gate

        # Step 7: Fold back
        xh_rotated_flat = xh_rotated.view(B, N, self.c_mid, self.m * self.m).permute(0, 2, 3, 1).reshape(B, -1, N)
        xh_aligned_map = F.fold(xh_rotated_flat, output_size=(H_l, W_l), kernel_size=self.m, stride=1, padding=pad)

        # Normalize by overlap
        ones = torch.ones(1, 1, H_l, W_l, device=device)
        ones_unfold = F.unfold(ones, kernel_size=self.m, stride=1, padding=pad)
        ones_fold = F.fold(ones_unfold, output_size=(H_l, W_l), kernel_size=self.m, stride=1, padding=pad)
        xh_aligned_map = xh_aligned_map / (ones_fold + self.eps)

        # Step 8: Reconstruct to C channels
        xh_recon = self.recon(xh_aligned_map)  # [B, C, H_l, W_l]

        # Step 9: Residual Frequency Fusion & Refinement
        # High-level upsampled features act as base, aligned features act as direction-aware residual
        x_high_modulated = x_high_up + self.layer_scale * xh_recon
        fused = x_low + x_high_modulated

        # Final refinement for Query Decoder
        out = self.refine_conv(fused)
        return out


@MODELS.register_module()
class DAFFusionFPN(FPN):
    """
    Directional-Aware Frequency Fusion FPN.
    Replaces standard Top-Down fusion with DAFF for rotation-sensitive feature alignment.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 fusion_modes: List[str],
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                 daff_cfg=dict(m=7, c_mid=64)):

        super(DAFFusionFPN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg)

        backbone_levels = self.backbone_end_level - self.start_level
        expected_fusion_steps = backbone_levels - 1
        assert len(fusion_modes) == expected_fusion_steps, \
            f"fusion_modes length ({len(fusion_modes)}) must be {expected_fusion_steps}"

        for mode in fusion_modes:
            assert mode in ['add', 'daff'], f"Invalid fusion mode: {mode}"

        self.fusion_modes = fusion_modes

        self.daff_modules = nn.ModuleList()
        for mode in fusion_modes:
            if mode == 'daff':
                # Pass out_channels as in_channels since FPN lateral convs standardize to out_channels
                self.daff_modules.append(DAFFusion(in_channels=out_channels, **daff_cfg))
            else:
                self.daff_modules.append(None)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            fusion_idx = used_backbone_levels - 1 - i
            mode = self.fusion_modes[fusion_idx]

            if mode == 'add':
                if 'scale_factor' in self.upsample_cfg:
                    upsampled = F.interpolate(laterals[i], **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    upsampled = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)
                laterals[i - 1] = laterals[i - 1] + upsampled
            elif mode == 'daff':
                # DAFF replaces the standard interpolation + addition
                fused_low = self.daff_modules[fusion_idx](laterals[i], laterals[i - 1])
                laterals[i - 1] = fused_low
            else:
                raise ValueError(f"Unknown fusion mode: {mode}")

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)
