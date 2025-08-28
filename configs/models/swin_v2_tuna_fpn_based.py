# configs/models/swin_v2_tuna_fpn_based.py

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        # --- THIS IS THE ONLY CHANGE ---
        # We specify our new V2 TUNA model as the default backbone
        type='SwinTransformerV2Tuna', 
        # These are default parameters for a 'Tiny' model, 
        # but our final experiment file will override them.
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        # ... other default params ...
        norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        type='FPNHead',
        # These channel numbers are for a 'Large' model, which is what we'll use.
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=512,
        dropout_ratio=0.1,
        # This will be overridden to 104 in the final config
        num_classes=19, 
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
