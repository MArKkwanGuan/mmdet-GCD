'''
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.115
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.312
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.057
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.036
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.097
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.160
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.285
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.208
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.235
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.238
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.062
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.242
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.284
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.397
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = -1.000
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = -1.000
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = -1.000
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = -1.000
# Class-specific LRP-Optimal Thresholds #
 [-1. -1. -1. -1. -1. -1. -1. -1.]
2024-07-18 11:56:16,877 - mmdet - INFO -
+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.041 | bridge        | 0.109 | storage-tank | 0.196 |
| ship     | 0.212 | swimming-pool | 0.181 | vehicle      | 0.145 |
| person   | 0.033 | wind-mill     | 0.001 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
2024-07-18 11:56:16,877 - mmdet - INFO -
+----------+------+---------------+------+--------------+------+
| category | oLRP | category      | oLRP | category     | oLRP |
+----------+------+---------------+------+--------------+------+
| airplane | nan  | bridge        | nan  | storage-tank | nan  |
| ship     | nan  | swimming-pool | nan  | vehicle      | nan  |
| person   | nan  | wind-mill     | nan  | None         | None |
+----------+------+---------------+------+--------------+------+
'''
_base_ = [
    '../_base_/datasets/aitodv2_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=8,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss', 
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(type='Gassuian_Combination_Loss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            gpu_assign_thr=512),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=1000))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001) # single gpu
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.001,
    step=[8, 11])
checkpoint_config = dict(interval=4)
evaluation = dict(interval=12, metric='bbox')