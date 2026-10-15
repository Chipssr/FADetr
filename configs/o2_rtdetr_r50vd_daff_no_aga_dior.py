from mmengine.config import read_base

with read_base():
    from .o2_rtdetr_r50vd_daff_dior import *

model['decoder']['use_aga'] = False
work_dir = './work_dirs/grouped_eval/daff_no_aga'
load_from = None
resume = False
