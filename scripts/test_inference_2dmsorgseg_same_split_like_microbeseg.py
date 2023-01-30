
from mmcv import Config
import os.path as osp
import numpy as np
from PIL import Image
import mmcv
import matplotlib.pyplot as plt
from collections import Counter
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot, get_data
from mmseg.core.evaluation import get_palette
import os
from torch import nn
from pytorch_model_summary import summary
from os.path import join as opj

def all_eval_except_dropout(m):
    if not isinstance(m, nn.Dropout) and not isinstance(m, mmcv.cnn.bricks.drop.DropPath) and not isinstance(m, nn.Dropout2d):
        m.eval()
    else:
        if isinstance(m, nn.Dropout2d):
            m.p = 1.0
            m.inplace = True
            m.train()
        print('keep', m, 'active')



# create model
ckp = '/home/ws/oc9627/mmseg_fork/work_dirs/2d_ms_org_seg_incl_Julian_annotations/split_like_microbeseg/best_mDice_iter_690.pth'
cfg = Config.fromfile(
    '/home/ws/oc9627/mmseg_fork/work_dirs/2d_ms_org_seg_incl_Julian_annotations/split-0/split-0.py')
#     cfg.load_from = '/home/ws/oc9627/mmseg_fork/work_dirs/segformer_b2_3class_seg_same_splits/dlw-10.0_split-0_lr-0.0001_wd-0.1/latest.pth'
cfg.load_from = ckp
cfg.model.decode_head.norm_cfg = dict(type='BN', requires_grad=True)

model = init_segmentor(
    cfg, checkpoint=ckp)
model.apply(all_eval_except_dropout)

img_dir = '/home/ws/oc9627/mmseg_fork/data/2d_ms_org_seg_incl_annot_Julian/images/split_like_microbeseg_test'
outdir = '/home/ws/oc9627/mmseg_fork/work_dirs/2d_ms_org_seg_incl_Julian_annotations/split_like_microbeseg/inference_test'
os.makedirs(outdir, exist_ok=True)
for img_loc in sorted(os.listdir(img_dir)):
    img = os.path.join(img_dir, img_loc)
    img_id = os.path.basename(img).replace('.jpg', '')

    seg, encoder_features, attn_weights = inference_segmentor(model, img)

    print(img_id)

    # outdir2 = opj(outdir, img_id)
    # os.makedirs(outdir2, exist_ok=True)

    # np.save(opj(outdir2, f'img.npy'), np.asarray(Image.open(img)))
    np.save(opj(outdir, f'{img_id}.npy'), seg[0])
    # for i in range(4):
    #     np.save(opj(outdir2, f'f{i}.npy'), np.squeeze(encoder_features[i].cpu().detach().numpy()))
    # for i in range(len(attn_weights)):
    #     np.save(opj(outdir2, f'attn_weights{i}.npy'), np.squeeze(attn_weights[i].cpu().detach().numpy()))