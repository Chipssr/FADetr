# FADetr

FADetr is a research implementation of oriented object detection based on
[O²-RT-DETR](https://github.com/wokaikaixinxin/O2-RT-DETR). The included model
combines direction-aware frequency fusion (DAFF) in the feature pyramid with
angle-guided query alignment (AGA) in the decoder classification branch.

**Status:** development snapshot prepared for a future paper release. Training,
accuracy, latency and checkpoint reproduction have not been verified in this
release-preparation environment. Paper metadata, trained weights and FADetr
results are not yet included. Upstream benchmark results are not FADetr results.

## Model

The primary configuration is
[`configs/o2_rtdetr_r34vd_2xb4_72e_dota_daff_faahead.py`](configs/o2_rtdetr_r34vd_2xb4_72e_dota_daff_faahead.py).
Historical configuration names are retained for compatibility; the head is
`RotatedRTDETRHead`, not a separate FAA ROI head.

```text
Image -> ResNet34-vd -> DAFFusionFPN -> RTDETRHybridEncoder
      -> 4-layer rotated decoder with AGA -> RotatedRTDETRHead
```

| Component | Implementation | Primary configuration |
| --- | --- | --- |
| Backbone | `rotated_rtdetr/resnet.py`, `res_layer.py` | ResNet34-vd; output channels 128/256/512 |
| Neck | `rotated_rtdetr/daff.py` | `DAFFusionFPN`; 3 outputs of 256 channels |
| DAFF | `rotated_rtdetr/daff.py` | Fusion modes `['add', 'daff']`; window 7; intermediate channels 64 |
| Encoder | `rotated_rtdetr/rtdetr_layers.py` | 1 Transformer layer on the highest level; FPN expansion 0.5 |
| Decoder | `rotated_rtdetr/rotated_rtdetr_faa_layers.py` | 4 layers; 256 dimensions; 8 self-attention heads |
| AGA | `rotated_rtdetr/aga.py` | Angle-conditioned classification query gate; scale 0.1 |
| Head and losses | `rotated_rtdetr/rotated_rtdetr_head.py`, `varifocal_loss.py`, `prob_iou.py` | 15 classes; Varifocal loss with ProbIoU targets, L1 and KLD losses |

## Installation

This repository is a **project extension**, not a standalone detector framework.
It requires a compatible full O²-RT-DETR checkout containing
`projects/rotated_dino`, framework training tools, MMDetection and MMRotate.
The upstream code and external datasets are not vendored here.

1. Follow the installation instructions in the
   [upstream repository](https://github.com/wokaikaixinxin/O2-RT-DETR), preferably
   using the exact environment and commit from the author's training machine.
   Tested dependency versions and the upstream commit remain to be recorded;
   compatibility with the latest upstream revision is not established.
2. Clone FADetr separately. Back up the existing upstream
   `projects/rotated_rtdetr/` directory. Copy FADetr's `configs/`,
   `rotated_rtdetr/` and `tools/` directories into that upstream project
   directory, replacing the corresponding files.
3. Run the commands below **from the upstream repository root**. Installing
   MMDetection/MMRotate packages alone does not supply `projects.rotated_dino`.

## Data preparation

Obtain DOTA through its authorized distribution and prepare single-scale image
patches using the compatible upstream preparation procedure. The default paths are:

```text
data/split_ss_dota/
├── trainval/
│   ├── images/
│   └── annfiles/
└── test/
    ├── images/
    └── annfiles/
```

Inputs are resized/padded to 1024 × 1024. Configure paths and evaluators in
`configs/dota_rr.py`. The checked-in validation loader points to the **same
trainval paths as the training loader**. It does not establish held-out
validation performance. Use a separately defined split or official test
evaluation when reporting results. The current test evaluator expects
annotations; for an unlabeled official test set, configure the upstream
format/submission evaluator instead of evaluating against missing labels.

## Train and evaluate

From the upstream repository root, after installing the extension and preparing data:

```bash
# First check that the configuration and model can actually be built.
python projects/rotated_rtdetr/tools/check_runtime.py

# Example: 2 GPUs. Actual configured batch size: 8 per GPU.
bash tools/dist_train.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_dota_daff_faahead.py 2

# Replace the checkpoint path with a weight file produced by your run.
bash tools/dist_test.sh projects/rotated_rtdetr/configs/o2_rtdetr_r34vd_2xb4_72e_dota_daff_faahead.py /path/to/epoch_72.pth 2
```

The inherited configuration sets 72 epochs, 300 matching queries, 100 denoising
queries, AdamW with learning rate `1e-4`, backbone learning-rate multiplier 0.1,
and EMA. Despite `2xb4` in the filename, actual batch size is **8 per GPU**
(16 across 2 GPUs); automatic LR scaling is disabled. Record the effective
batch size and any overrides with each reported experiment.

## Validation and reproducibility

Run dependency-free static checks from this repository root:

```bash
python tools/check_release.py
```

The GitHub workflow runs the same checks: Python syntax, local imports, package
exports and the primary configuration inheritance. These checks do not execute
the model or verify numerical results. The runtime preflight above requires the
complete training environment, initializes the model (which may download
pretrained weights) and checks key resolved settings; it does not run a forward
pass or training experiment.

Only the R34 DOTA entry and its parent/data/runtime configurations are the
documented release path. Other configurations are retained research artifacts.
The old R50 baseline references the absent `RotatedRTDETRB`,
`o2_rtdetr_r50vd_all_1.py` references an external FAA project, and DIOR
configurations require additional upstream data configuration. They are not
validated release examples.

See [the release checklist](docs/RELEASE_CHECKLIST.md) and
[the Chinese publishing guide](docs/PUBLISHING.zh-CN.md).

## Acknowledgements and citation

FADetr builds on O²-RT-DETR, RT-DETR, MMDetection, MMRotate, MMCV and MMEngine.
Existing source comments and upstream attribution are retained. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Upstream results must not be
relabeled as FADetr results.

The FADetr paper title, authors and publication identifier will be added when
available. Please cite relevant upstream work when using this implementation:

```bibtex
@article{ding2026o2rtdetr,
  author = {Ding, Zeyu and Zhou, Yong and Zhao, Jiaqi and Du, Wen-Liang and Li, Xixi and Yao, Rui and Saddik, Abdulmotaleb El},
  title = {Real-Time Oriented Object Detection Transformer in Remote Sensing Images},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year = {2026},
  volume = {64},
  pages = {1--14},
  doi = {10.1109/TGRS.2026.3671683}
}
```

## License

The repository includes the [Apache License 2.0](LICENSE), consistent with the
license displayed by the upstream O²-RT-DETR repository at preparation time.
Third-party notices and licenses must be retained. A complete provenance and
notice comparison against the exact training-source revision remains a release
checklist item. Dataset and pretrained-checkpoint terms are separate.
