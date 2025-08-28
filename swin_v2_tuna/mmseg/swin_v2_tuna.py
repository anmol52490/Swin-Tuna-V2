# swin_tuna/mmseg/swin_v2_tuna.py

from mmseg.registry import MODELS
from mmengine.model import BaseModule
from mmpretrain.models.backbones import SwinTransformerV2
from ..modeling.tuna_v2_injector import TunaV2Injector

@MODELS.register_module()
class SwinTransformerV2Tuna(BaseModule):
    def __init__(self, backbone, decode_head, neck=None, train_cfg=None, test_cfg=None, data_preprocessor=None, **kwargs):
        super().__init__( **kwargs)

        self.backbone = MODELS.build(backbone)
        
        TunaV2Injector.inject_tuna(self.backbone)
        
        if neck is not None:
            self.neck = MODELS.build(neck)
        
        # --- THIS IS THE FIX ---
        # We no longer pass train_cfg or test_cfg into the decode_head's config.
        # The head is built with its own parameters.
        self.decode_head = MODELS.build(decode_head)
        
        # The main model stores the train_cfg and test_cfg for the runner to use.
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.freeze_parameters()

    def freeze_parameters(self):
        """
        Iterates through all backbone parameters and freezes everything
        that does not have 'tuna_' in its name. The neck and head remain trainable.
        """
        for name, param in self.backbone.named_parameters():
            if 'tuna_' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        if hasattr(self, 'neck'):
            for param in self.neck.parameters():
                param.requires_grad = True
        for param in self.decode_head.parameters():
            param.requires_grad = True
                
    def train(self, mode=True):
        """Overrides train() to ensure parameters remain frozen."""
        super().train(mode=mode)
        self.freeze_parameters()
            
    def extract_feat(self, inputs):
        x = self.backbone(inputs)
        if hasattr(self, 'neck'):
            x = self.neck(x)
        return x

    def loss(self, inputs, data_samples):
        x = self.extract_feat(inputs)
        losses = dict()
        loss_decode = self.decode_head.loss(x, data_samples, self.train_cfg)
        losses.update(loss_decode)
        return losses

    def predict(self, inputs, data_samples=None):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=(0, 0, 0, 0))
            ] * inputs.shape[0]
            
        x = self.extract_feat(inputs)
        return self.decode_head.predict(x, batch_img_metas, self.test_cfg)

    def _forward(self, inputs, data_samples=None):
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else: # mode == 'tensor'
            return self._forward(inputs, data_samples)

