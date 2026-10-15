# Third-party sources and attribution

FADetr is derived from O²-RT-DETR rather than an independent implementation of
the complete detection framework.

| Source | Relationship |
| --- | --- |
| [O²-RT-DETR](https://github.com/wokaikaixinxin/O2-RT-DETR) | Detector/project foundation and required `projects.rotated_dino`; upstream displays Apache-2.0 |
| [RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Components referenced by existing source comments |
| [MMDetection](https://github.com/open-mmlab/mmdetection) | Backbone foundation, layers, losses and assignment |
| [MMRotate](https://github.com/open-mmlab/mmrotate) | Oriented detection structures, registry, losses and evaluation |
| [MMCV](https://github.com/open-mmlab/mmcv), [MMEngine](https://github.com/open-mmlab/mmengine) | Operators, configuration and training infrastructure |

The supplied tree contains DAFF, AGA, a decoder using AGA, a detector connecting
that decoder, and related configurations. Exact author-by-author change history
is not available in this directory, which arrived without Git history. Before
a paper release, compare against the exact upstream source revision, retain all
applicable copyright/license/NOTICE texts, and add prominent change notices to
adapted files. Do not infer authorship solely from a filename.

The upstream Apache-2.0 LICENSE was retrieved through the authenticated
GitHub connection on 2026-10-15 and is included verbatim as LICENSE.
License blob SHA: 261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64.

Inspected upstream reference commit: e15b5a6696b864c50624ec8eed891785c8150c79.
The inspected root listing contains no standalone NOTICE file. This reference
commit is for source/notice comparison; it is not claimed to be the original
training-source commit or a verified compatible runtime. Files that differ
from the inspected upstream model files carry FADetr adaptation notices.
The exact training-source revision and other third-party provenance still
need confirmation before a reproducible paper release.

The release-preparation edit to `rotated_rtdetr/__init__.py` removes references
to absent `RotatedRTDETRB` and undefined `FAARotatedRTDETR`, and exports the
existing `RotatedRTDETRHead`. Model behavior was not changed by that edit.

External data and checkpoints are not included. Referenced download links do
not grant redistribution rights over those artifacts.
