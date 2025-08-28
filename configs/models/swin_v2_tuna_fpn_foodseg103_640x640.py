# configs/models/swin_v2_tuna_fpn_foodseg103_640x640.py

# We still need this to tell the framework where to find our custom model class
custom_imports = dict(imports=['swin_v2_tuna.mmseg.swin_v2_tuna'], allow_failed_imports=False)

# Inherit the data, schedule, and runtime settings.
# We no longer inherit from a base model config, as our main model file defines everything.
_base_ = [
    '../datasets/foodseg103_640x640.py',
    '../default_runtime.py',
    '../schedule_100k.py'
]

checkpoint_file = 'pretrain/swinv2_large_22k_500k.pth'

# Define the FPN neck, which was previously in the base config
neck=dict(
    type='FPN',
    in_channels=[192, 384, 768, 1536],
    out_channels=256,
    num_outs=4)

# --- KEY CHANGE ---
# The structure now matches the __init__ of our new SwinTransformerV2Tuna class
model = dict(
    type='SwinTransformerV2Tuna', # This points to our new Python class
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=0),
    backbone=dict(
        # This is the config for the SwinV2 model, which our wrapper will build
        type='mmpretrain.SwinTransformerV2',
        arch='large',
        img_size=384,
        drop_path_rate=0.2,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
    ),
    neck=neck, # Pass the FPN neck config
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=512,
        dropout_ratio=0.1,
        num_classes=104, # Critical override for our dataset
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    
    # model training and testing settings
    train_cfg=dict(max_iters=100),
    test_cfg=dict(mode='whole'),

    # Override the data_root in the dataloaders
    train_dataloader=dict(
        dataset=dict(
            data_root='datasets/FoodSeg103_processed'
        )
    ),
    val_dataloader=dict(
        dataset=dict(
            data_root='datasets/FoodSeg103_processed'
        )
    ),
    test_dataloader=dict(
        dataset=dict(
            data_root='datasets/FoodSeg103_processed'
        )
    )
)
