'''
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.087
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.154
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.086
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.002
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.006
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.020
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.168
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.177
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.179
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.179
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.005
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.039
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.103
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.308
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.922
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.238
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.690
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.805
# Class-specific LRP-Optimal Thresholds #
 [0.32  0.216 0.405 0.329 0.373 0.224 0.356 0.304]
2024-08-01 08:25:58,401 - mmdet - INFO -
+------------+-------+----------+-------+----------+-------+
| category   | AP    | category | AP    | category | AP    |
+------------+-------+----------+-------+----------+-------+
| pedestrian | 0.032 | bicycle  | 0.006 | car      | 0.284 |
| van        | 0.105 | truck    | 0.094 | tricycle | 0.022 |
| bus        | 0.124 | motor    | 0.030 | None     | None  |
+------------+-------+----------+-------+----------+-------+
'''
_base_ = [
    '../_base_/datasets/visdrone2019_detection.py',
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
        num_classes=10,
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
        loss_bbox=dict(type='Gassuian_Combination_Loss', loss_weight=4.0)),
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
checkpoint_config = dict(interval=4)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.001,
    step=[8, 11])
evaluation = dict(interval=12, metric='bbox')