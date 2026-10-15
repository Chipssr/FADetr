from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r50vd_daff_dior import *

# Retain frequency-energy response and FFT direction estimation, but disable
# the patch-wise spatial rotation that performs cross-scale alignment.
model['neck']['daff_cfg'].update(
    dict(
        m=7,
        c_mid=64,
        use_frequency_response=True,
        use_direction_alignment=False,
        direction_estimator='fft'))

work_dir = './work_dirs/o2_rtdetr_r50vd_daff_no_direction_alignment_dior'
load_from = None
resume = False
