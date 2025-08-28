# swin_tuna/modeling/tuna_v2_injector.py

import torch
import torch.nn as nn
from mmpretrain.models import SwinTransformerV2
from .tuna import Tuna

def inject_attributes_v2(target: nn.Module, conv_size, dim, hidden_dim, device=None, clazz=Tuna):
    """Adds the TUNA modules and scaling parameters to a SwinV2Block."""
    setattr(target, 'tuna_1', clazz(dim, hidden_dim=hidden_dim, conv_size=conv_size).to(device=device))
    setattr(target, 'tuna_2', clazz(dim, hidden_dim=hidden_dim, conv_size=conv_size).to(device=device))
    setattr(target, 'tuna_scale1', nn.Parameter(torch.ones(dim) * 1e-6, requires_grad=True).to(device=device))
    setattr(target, 'tuna_scale2', nn.Parameter(torch.ones(dim) * 1e-6, requires_grad=True).to(device=device))
    setattr(target, 'tuna_x_scale1', nn.Parameter(torch.ones(dim), requires_grad=True).to(device=device))
    setattr(target, 'tuna_x_scale2', nn.Parameter(torch.ones(dim), requires_grad=True).to(device=device))

class TunaV2Injector:
    @staticmethod
    def inject_tuna(target_model: nn.Module):
        assert isinstance(target_model, SwinTransformerV2), "This injector is only for SwinTransformerV2"
        
        device = next(target_model.parameters()).device
        
        conv_size_list = [7, 5, 5, 3] 
        hidden_dim_list = [64, 64, 96, 192]

        # This is the new forward method that will replace the original one.
        # It correctly matches the SwinV2 "post-normalization" data flow.
        def new_forward(self, x, hw_shape):
            identity = x
            
            # --- Attention Branch ---
            x_main_attn = self.attn(self.norm1(x), hw_shape=hw_shape)
            x = identity + self.drop_path(x_main_attn)
            
            # Parallel TUNA branch
            x_tuna_attn = self.tuna_1(identity, hw_shape)
            
            # Combine
            x = self.tuna_x_scale1 * x + self.tuna_scale1 * x_tuna_attn
            
            identity = x

            # --- MLP (FFN) Branch ---
            x_main_ffn = self.ffn(self.norm2(x))
            x = identity + self.drop_path(x_main_ffn)
            
            # Parallel TUNA branch
            x_tuna_ffn = self.tuna_2(identity, hw_shape)
            
            # Combine
            x = self.tuna_x_scale2 * x + self.tuna_scale2 * x_tuna_ffn
            
            return x

        # --- KEY CHANGE: DYNAMIC CLASS FINDING ---
        # 1. We'll find the class to modify during our loop.
        SwinBlockV2_class_to_modify = None

        for i, stage in enumerate(target_model.stages):
            for block in stage.blocks:
                # If this is the first block we've seen, get its class.
                if SwinBlockV2_class_to_modify is None:
                    SwinBlockV2_class_to_modify = type(block)
                
                dim = block.ffn.embed_dims
                inject_attributes_v2(
                    block, 
                    conv_size=conv_size_list[i], 
                    dim=dim, 
                    hidden_dim=hidden_dim_list[i], 
                    device=device
                )
    
        # 2. Now, we monkey-patch the dynamically found class.
        # This is much more robust than using a hardcoded name.
        if SwinBlockV2_class_to_modify is not None:
            SwinBlockV2_class_to_modify.forward = new_forward
        else:
            raise RuntimeError("Could not find any SwinV2 blocks to inject TUNA into.")

