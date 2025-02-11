"""
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.366
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.571
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.397
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.207
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.403
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.510
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.510
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.510
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.552
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.641
"""
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]
model = dict(
    rpn_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='Gassuian_Combination_Loss', loss_weight=10.0)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                gpu_assign_thr=512)),
        rpn_proposal=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7)),
        rpn=dict(
            assigner=dict(
                gpu_assign_thr=512,
                pos_iou_thr=0.8,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='gcd'))),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=1000)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001) # single gpu
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.001,
    step=[8, 11])
# learning policy
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(interval=4)
evaluation = dict(interval=12, metric='bbox')
