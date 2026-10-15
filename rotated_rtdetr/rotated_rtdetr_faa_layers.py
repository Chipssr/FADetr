from typing import Tuple

from torch import Tensor, nn

from mmdet.models.layers.transformer import inverse_sigmoid
from projects.rotated_dino.rotated_dino import RotatedDinoTransformerDecoder

from .utils import MLP
from .aga import AngleGuidedAlignment


class RotatedRTDETRAGATransformerDecoder(RotatedDinoTransformerDecoder):
    """Transformer decoder with optional Angle Guided Alignment (AGA)."""

    def __init__(self, *args, use_aga: bool = True, **kwargs) -> None:
        self.use_aga = use_aga
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        super()._init_layers()

        self.ref_point_head = MLP(
            5,
            self.embed_dims * 2,
            self.embed_dims,
            2
        )

        self.norm = nn.Identity()

        if self.use_aga:
            self.aga = AngleGuidedAlignment(
                embed_dims=self.embed_dims,
                hidden_dims=64,
                scale=0.1
            )

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        key_padding_mask: Tensor,
        self_attn_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reg_branches: nn.ModuleList,
        cls_branches: nn.ModuleList,
        **kwargs
    ) -> Tuple[Tensor]:

        assert self.return_intermediate
        assert reg_branches is not None
        assert reference_points.shape[-1] == 5

        eval_idx = kwargs.pop('eval_idx', -1)

        if eval_idx < 0:
            eval_idx += self.num_layers
            assert eval_idx >= 0

        hidden_states = []

        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for lid, layer in enumerate(self.layers):

            num_levels = layer.cross_attn_cfg.num_levels

            reference_points_input = (
                reference_points
                .unsqueeze(2)
                .repeat(1, 1, num_levels, 1)
            )

            reference_points_input[..., -1] *= self.angle_factor

            query_pos = self.ref_point_head(reference_points)

            query = layer(
                query=query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs
            )

            # --------------------------------------------------
            # Regression Branch
            # --------------------------------------------------
            tmp = reg_branches[lid](query)

            unact_pred_boxes = (
                tmp +
                inverse_sigmoid(
                    reference_points,
                    eps=1e-3
                )
            )

            pred_boxes = unact_pred_boxes.sigmoid()

            # --------------------------------------------------
            # Angle Guided Alignment
            # --------------------------------------------------
            theta = pred_boxes[..., 4] * self.angle_factor

            cls_query = self.aga(
                query=query,
                theta=theta
            ) if self.use_aga else query

            # --------------------------------------------------
            # Output
            # --------------------------------------------------
            if self.training or lid == eval_idx:

                # 保持原始query用于回归
                hidden_states.append(query)

                # 分类使用AGA增强后的query
                all_layers_outputs_classes.append(
                    cls_branches[lid](cls_query)
                )

                all_layers_outputs_coords.append(
                    pred_boxes
                )

                if (
                    not self.training
                    or lid == self.num_layers - 1
                ):
                    break

            # --------------------------------------------------
            # Update reference points
            # --------------------------------------------------
            reference_points = (
                unact_pred_boxes
                .detach()
                .sigmoid()
            )

        return (
            hidden_states,
            (
                all_layers_outputs_classes,
                all_layers_outputs_coords
            )
        )
