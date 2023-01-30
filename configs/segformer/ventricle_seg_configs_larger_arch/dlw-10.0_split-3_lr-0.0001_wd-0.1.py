_base_ = [        
'../../_base_/datasets/if.py','../../_base_/models/segformer_mit-b0.py',        
'../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py'        
]        
        
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa        
        
# model settings        
model = dict(        
    pretrained=checkpoint,        
    backbone=dict(        
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 6, 3], in_channels=1),        
          decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=2,         
                              loss_decode=[        
                              dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),        
                              dict(type='DiceLoss', loss_name='loss_dice', loss_weight=10.0,         
                              ignore_index=0, avg_non_ignore=True) # using here correct?        
                                        ]))        
                
# optimizer        
optimizer = dict(        
            _delete_=True,        
            type='AdamW',        
            lr=0.0001, #    lr=0.00006,        
            betas=(0.9, 0.999),        
            weight_decay=0.1,        
            paramwise_cfg=dict(        
                custom_keys={        
                    'pos_block': dict(decay_mult=0.),        
                    'norm': dict(decay_mult=0.),        
                    'head': dict(lr_mult=10.)        
                }))        
                
workflow = [('train', 1), ('val', 1)]        
evaluation = dict(interval=10, metric='mDice', save_best='mDice', ignore_index=0, by_epoch=True,         
                        outdir='/home/ws/oc9627/mmsegmentation/work_dirs/segformer_mit-b0_512x512_160k_IF/different_model_arch/dlw-10.0_split-3_lr-0.0001_wd-0.1/eval_imgs')        
log_config = dict(        
            interval=10,        
            hooks=[        
                dict(type='TextLoggerHook'),        
                dict(type='TensorboardLoggerHook')        
            ])        
                
                
runner = dict(type='IterBasedRunner', max_iters=2000)        
                
data_root = '/home/ws/oc9627/mmsegmentation/data/if'        
data = dict(        
        samples_per_gpu=2,        
        workers_per_gpu=2,        
        train=dict(        
            data_root=data_root,        
            img_dir='images/split3_training',        
            ann_dir='masks/split3_training'),        
        val=dict(        
            data_root=data_root,        
            img_dir='images/split3_validation',        
            ann_dir='masks/split3_validation'))        
work_dir='/home/ws/oc9627/mmsegmentation/work_dirs/segformer_mit-b0_512x512_160k_IF/different_model_arch/dlw-10.0_split-3_lr-0.0001_wd-0.1'