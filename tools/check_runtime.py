"""Build/initialize the primary FADetr model inside the full upstream checkout."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys


def main():
    project_root = Path(__file__).resolve().parents[1]
    upstream_root = project_root.parent.parent
    if not (upstream_root / 'projects' / 'rotated_dino').is_dir():
        raise SystemExit(
            'Place this extension under <upstream>/projects/rotated_rtdetr first. '
            'The compatible full upstream projects/rotated_dino source is required.'
        )
    sys.path.insert(0, str(upstream_root))
    for name in ('torch', 'mmengine', 'mmcv', 'mmdet', 'mmrotate'):
        try:
            print(f'{name}: {version(name)}')
        except PackageNotFoundError:
            print(f'{name}: distribution metadata unavailable')

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmrotate.registry import MODELS
    from mmrotate.utils import register_all_modules

    register_all_modules(init_default_scope=False)
    init_default_scope('mmrotate')
    config_path = project_root / 'configs' / 'o2_rtdetr_r34vd_2xb4_72e_dota_daff_faahead.py'
    cfg = Config.fromfile(str(config_path))
    assert cfg.model.backbone.depth == 34, 'Unexpected backbone depth'
    assert list(cfg.model.neck.in_channels) == [128, 256, 512], 'Unexpected neck channels'
    assert cfg.model.encoder.fpn_cfg.expansion == 0.5, 'Unexpected FPN expansion'
    assert cfg.model.decoder.num_layers == 4, 'Unexpected decoder depth'
    assert cfg.train_dataloader.batch_size == 8, 'Unexpected batch size'
    model = MODELS.build(cfg.model)
    model.init_weights()
    assert type(model).__name__ == 'RotatedRTDETR'
    assert type(model.neck).__name__ == 'DAFFusionFPN'
    assert type(model.decoder).__name__ == 'RotatedRTDETRAGATransformerDecoder'
    assert model.decoder.use_aga, 'AGA is disabled'
    assert type(model.bbox_head).__name__ == 'RotatedRTDETRHead'
    print('PASS: primary configuration resolved; model built and initialized.')
    print('No forward pass, training or accuracy evaluation was performed.')


if __name__ == '__main__':
    main()
