# # %%

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

# # %%

img_dir = '/home/ws/oc9627/mmseg_fork/data/2d_ms_org_seg_incl_annot_Julian/images/split1_validation'
labels = []
ckp = '/home/ws/oc9627/mmseg_fork/work_dirs/2d_ms_org_seg_incl_Julian_annotations/split-0/best_mDice_iter_110.pth'
cfg = Config.fromfile(
    '/home/ws/oc9627/mmseg_fork/work_dirs/2d_ms_org_seg_incl_Julian_annotations/split-0/split-0.py')
#     cfg.load_from = '/home/ws/oc9627/mmseg_fork/work_dirs/segformer_b2_3class_seg_same_splits/dlw-10.0_split-0_lr-0.0001_wd-0.1/latest.pth'
cfg.load_from = ckp
cfg.model.decode_head.norm_cfg = dict(type='BN', requires_grad=True)

model = init_segmentor(
    cfg, checkpoint=ckp)
model.apply(all_eval_except_dropout)

imgs, pred = [], []
for img_loc in sorted(os.listdir(img_dir)):
    img = os.path.join(img_dir, img_loc)

    result = inference_segmentor(model, img)
    print(img)
    imgs.append(Image.open(img))
    pred.append(result[0])

# %%
print(len(imgs))
# %%
fig, axs = plt.subplots(len(imgs), 2, figsize=(3, len(imgs)))
for i in range(len(imgs)):
    axs[i, 0].imshow(imgs[i])
    axs[i, 1].imshow(pred[i])

plt.show(block=True)


