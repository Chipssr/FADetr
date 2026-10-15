from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r50vd_daff_dior import *

# Remove only DAFF's energy-guided frequency response (spectral gate).
model['neck']['daff_cfg'].update(
    dict(
        m=7,
        c_mid=64,
        use_frequency_response=False,
        use_direction_alignment=True,
        direction_estimator='fft'))

work_dir = './work_dirs/o2_rtdetr_r50vd_daff_no_freq_response_dior'
load_from = None
resume = False
