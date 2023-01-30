import os

outdir_root = "/home/ws/oc9627/mmsegmentation/work_dirs/segformer_mit-b0_512x512_160k_IF/different_model_arch"

lr = 0.0001
wd  = 0.1
for dlw in [10.0]:
    for split in range(5):
        conf_id = f"dlw-{dlw}_split-{split}_lr-{lr}_wd-{wd}"
        outdir = os.path.join(outdir_root, conf_id)
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(os.path.join(outdir, "eval_imgs"), exist_ok=True)
        conf = "_base_ = [\
        \n'../../_base_/datasets/if.py','../../_base_/models/segformer_mit-b0.py',\
        \n'../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py'\
        \n]\
        \n\
        \ncheckpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa\
        \n\
        \n# model settings\
        \nmodel = dict(\
        \n    pretrained=checkpoint,\
        \n    backbone=dict(\
        \n        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 6, 3], in_channels=1),\
        \n          decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=2, \
        \n                              loss_decode=[\
        \n                              dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),\
        \n                              dict(type='DiceLoss', loss_name='loss_dice', loss_weight="+str(dlw)+", \
        \n                              ignore_index=0, avg_non_ignore=True) # using here correct?\
        \n                                        ]))\
        \n        \
        \n# optimizer\
        \noptimizer = dict(\
        \n            _delete_=True,\
        \n            type='AdamW',\
        \n            lr="+str(lr)+", #    lr=0.00006,\
        \n            betas=(0.9, 0.999),\
        \n            weight_decay="+str(wd)+",\
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
        \nrunner = dict(type='IterBasedRunner', max_iters=1500)\
        \n        \
        \ndata_root = '/home/ws/oc9627/mmsegmentation/data/if'\
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
        with open(os.path.join("/home/ws/oc9627/mmsegmentation/configs/segformer/ventricle_seg_configs_larger_arch", f"{conf_id}.py"), "w") as text_file:
            text_file.write(conf)
