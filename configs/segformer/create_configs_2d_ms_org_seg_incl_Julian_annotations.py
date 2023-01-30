import os

outdir_root = '/home/ws/oc9627/mmseg_fork/work_dirs/2d_ms_org_seg_incl_Julian_annotations'


for split in range(5):
    conf_id = f"split-{split}"
    outdir = os.path.join(outdir_root, conf_id)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "eval_imgs"), exist_ok=True)
    conf = "_base_ = [\
    \n'../../_base_/models/segformer_mit-b0.py', '../../_base_/datasets/2dMS.py',\
    \n'../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py'\
    \n]\
    \ncheckpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa\
    \n\
    \nmodel = dict(pretrained=checkpoint, \
    \n             decode_head=dict(num_classes=2, \
    \n                              loss_decode=[\
    \n                              dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),\
    \n                              dict(type='DiceLoss', loss_name='loss_dice', loss_weight=10.0, ignore_index=0, avg_non_ignore=True) # using here correct?\
    \n                                        ]), \
    \n              backbone=dict(in_channels=3))\
    \n        \
    \n# optimizer\
    \noptimizer = dict(\
    \n            _delete_=True,\
    \n            type='AdamW',\
    \n            lr=0.0001, #    lr=0.00006,\
    \n            betas=(0.9, 0.999),\
    \n            weight_decay=0.1,\
    \n            paramwise_cfg=dict(\
    \n                custom_keys={\
    \n                    'pos_block': dict(decay_mult=0.),\
    \n                    'norm': dict(decay_mult=0.),\
    \n                    'head': dict(lr_mult=10.)\
    \n                }))\
    \n        \
    \nworkflow = [('train', 1), ('val', 1)]\
    \nevaluation = dict(interval=10, metric='mDice', save_best='mDice', ignore_index=0, by_epoch=True, \
    \n                        outdir='"+outdir+"/eval_imgs')\
    \nlog_config = dict(\
    \n            interval=10,\
    \n            hooks=[\
    \n                dict(type='TextLoggerHook'),\
    \n                dict(type='TensorboardLoggerHook')\
    \n            ])\
    \n        \
    \n        \
    \nrunner = dict(type='IterBasedRunner', max_iters=1000)\
    \n        \
    \ndata_root = '/home/ws/oc9627/mmseg_fork/data/2d_ms_org_seg_incl_annot_Julian'\
    \ndata = dict(\
    \n        samples_per_gpu=2,\
    \n        workers_per_gpu=2,\
    \n        train=dict(\
    \n            data_root=data_root,\
    \n            img_dir='images/split"+str(split)+"_training',\
    \n            ann_dir='masks/split"+str(split)+"_training'),\
    \n        val=dict(\
    \n            data_root=data_root,\
    \n            img_dir='images/split"+str(split)+"_validation',\
    \n            ann_dir='masks/split"+str(split)+"_validation'))\
    \nwork_dir='"+outdir+"'"
    with open(os.path.join("/home/ws/oc9627/mmsegmentation/configs/segformer/2d_ms_org_seg_incl_Julian_annotations", f"{conf_id}.py"), "w") as text_file:
        text_file.write(conf)
