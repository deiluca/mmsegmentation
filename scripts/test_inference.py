# %%

from mmcv import Config
import os.path as osp
import numpy as np
from PIL import Image
import mmcv
import matplotlib.pyplot as plt
from collections import Counter
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import os
from torch import nn


def all_eval_except_dropout(m):
    if not isinstance(m, nn.Dropout) and not isinstance(m, mmcv.cnn.bricks.drop.DropPath) and not isinstance(m, nn.Dropout2d):
        m.eval()
    else:
        if isinstance(m, nn.Dropout2d):
            m.p = 1.0
            m.inplace = True
            m.train()
        print('keep', m, 'active')

# %%

img_dir = '/home/ws/oc9627/mmseg_fork/data/if_3class_uncertainty_annot_adapted_splits/images/split0_validation'
labels = []

for img_loc in sorted(os.listdir(img_dir))[:1]:
    cfg = Config.fromfile(
        '/home/ws/oc9627/mmseg_fork/configs/segformer/segformer_b2_dlw-10.0_split-0_lr-0.00005_wd-0.01_GN_2class_correct_class_weights_and_ignore_index_mcd.py')
    #     cfg.load_from = '/home/ws/oc9627/mmseg_fork/work_dirs/segformer_b2_3class_seg_same_splits/dlw-10.0_split-0_lr-0.0001_wd-0.1/latest.pth'
    cfg.load_from = '/home/ws/oc9627/mmseg_fork/work_dirs/try_to_include_mcd/best_mDICE_iter_90.pth'
    cfg.model.decode_head.norm_cfg = dict(type='BN', requires_grad=True)

    img = os.path.join(img_dir, img_loc)
    model = init_segmentor(
        cfg, checkpoint='/home/ws/oc9627/mmseg_fork/work_dirs/try_to_include_mcd/best_mDice_iter_90.pth')
    model.apply(all_eval_except_dropout)
    result = inference_segmentor(model, img)
    print(img)
# %%
plt.imshow(result[0])
plt.show(block=True)

# plt.figure(figsize=(8, 6))
# print(result[0].shape)
# print(Counter(result[0].ravel()))
# show_result_pyplot(model, img, result, get_palette('if_dataset'))
# show_result_pyplot(model, img, result, palette)
