_base_ = [    
'../../_base_/models/segformer_mit-b0.py', '../../_base_/datasets/2dMS.py',    
'../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py'    
]    
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa    

model = dict(pretrained=checkpoint,     
             decode_head=dict(num_classes=2,     
                              loss_decode=[    
                              dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),    
                              dict(type='DiceLoss', loss_name='loss_dice', loss_weight=10.0, ignore_index=0, avg_non_ignore=True) # using here correct?    
                                        ]),     
              backbone=dict(in_channels=3))    
            
# optimizer    
optimizer = dict(    
            _delete_=True,    
            type='AdamW',    
            lr=0.000000000001, #    lr=0.00006,    
            betas=(0.9, 0.999),
            weight_decay=0.0000000000001,    
            paramwise_cfg=dict(    
                custom_keys={    
                    'pos_block': dict(decay_mult=0.),    
                    'norm': dict(decay_mult=0.),    
                    'head': dict(lr_mult=10.)    
                }))
            
workflow = [('train', 1), ('val', 1)]    
evaluation = dict(interval=1, metric='mDice', save_best='mDice', ignore_index=0, by_epoch=True,     
                        outdir='/home/ws/oc9627/mmseg_fork/work_dirs/2d_ms_org_seg_incl_Julian_annotations/inference_run5/split1/eval_imgs')    
log_config = dict(    
            interval=1,    
            hooks=[    
                dict(type='TextLoggerHook'),    
                dict(type='TensorboardLoggerHook')    
            ])    
            
            
runner = dict(type='IterBasedRunner', max_iters=2)    
            
data_root = '/home/ws/oc9627/mmseg_fork/data/2d_ms_org_seg'    
# data = dict(    
#         samples_per_gpu=2,    
#         workers_per_gpu=2,    
#         train=dict(    
#             data_root=data_root,    
#             img_dir='images/split0_training',    
#             ann_dir='masks/split0_training'),    
#         val=dict(    
#             data_root=data_root,    
#             img_dir='images/split0_training',    
#             ann_dir='masks/split0_training'))

# data = dict(    
#         samples_per_gpu=2,    
#         workers_per_gpu=2,    
#         train=dict(    
#             data_root=data_root,    
#             img_dir='images/split0_training',    
#             ann_dir='masks/split0_training'),    
#         val=dict(    
#             data_root=data_root,    
#             img_dir='images/split1_validation',    
#             ann_dir='masks/split1_validation'))

data = dict(    
        samples_per_gpu=2,    
        workers_per_gpu=2,    
        train=dict(    
            data_root=data_root,    
            img_dir='images/split0_training',    
            ann_dir='masks/split0_training'),    
        val=dict(    
            data_root=data_root,    
            img_dir='images/2d_ms_org_seg_run5_for_inference',    
            ann_dir='masks/for_inference_2d_ms_org_seg_run5_dummy_label'))   
work_dir='/home/ws/oc9627/mmseg_fork/work_dirs/2d_ms_org_seg_incl_Julian_annotations/inference_run5/split1'
