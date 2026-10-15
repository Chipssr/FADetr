from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r50vd_daff_dior import *

# Keep the spatial rotation operator, but replace adaptive local FFT direction
# estimation with one learnable global angle shared by all local patches.
model['neck']['daff_cfg'].update(
    dict(
        m=7,
        c_mid=64,
        use_frequency_response=True,
        use_direction_alignment=True,
        direction_estimator='global'))

work_dir = './work_dirs/o2_rtdetr_r50vd_daff_no_direction_estimation_dior'
load_from = None
resume = False
