import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, Optional, Tuple, Union
from models.attention import BasicTransformerBlock

def torch_dfs(model: torch.nn.Module):
    r"""
    Performs a depth-first search on the given PyTorch model and returns a list of all its child modules.

    Args:
        model (torch.nn.Module): The PyTorch model to perform the depth-first search on.

    Returns:
        list: A list of all child modules of the given model.
    """
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class LoRAAttnMaskProcessor:
    def __init__(self, device, dtype, lora_rank=128, lora_alpha=1.0, latent_dim=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        # Initialize LoRA layers
        self.lora_alpha = lora_alpha
        self.lora_rank = lora_rank
        self.latent_dim = latent_dim

        # Helper function to create LoRA layers
        def create_lora_layer(in_dim, mid_dim, out_dim, device=device, dtype=dtype):
            # Define the LoRA layers
            lora_a = nn.Linear(in_dim, mid_dim, bias=False, device=device, dtype=dtype)
            lora_b = nn.Linear(mid_dim, out_dim, bias=False, device=device, dtype=dtype)
            
            # Initialize lora_a with random parameters (default initialization)
            nn.init.kaiming_uniform_(lora_a.weight, a=math.sqrt(5))  # or another suitable initialization
            
            # Initialize lora_b with zero values
            nn.init.zeros_(lora_b.weight)

            lora_a.weight.requires_grad = True
            lora_b.weight.requires_grad = True
            
            # Combine the layers into a sequential module
            return nn.Sequential(lora_a, lora_b)

        self.to_q_lora = create_lora_layer(latent_dim, lora_rank, latent_dim)
        self.to_k_lora = create_lora_layer(latent_dim, lora_rank, latent_dim)
        self.to_v_lora = create_lora_layer(latent_dim, lora_rank, latent_dim)
        self.to_out_lora = create_lora_layer(latent_dim, lora_rank, latent_dim)

    def _apply_lora(self, hidden_states, seq_len, query, key, value, scaling):
        """Applies LoRA updates to query, key, and value tensors."""
        query_delta = self.to_q_lora(hidden_states).to(query.device)
        query += query_delta * scaling

        key_delta = self.to_k_lora(hidden_states).to(key.device)
        key += key_delta * scaling

        value_delta = self.to_v_lora(hidden_states).to(value.device)
        value += value_delta * scaling

        return query, key, value

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        if self.i_num == 100:
            for lora_layer in [self.to_q_lora, self.to_k_lora, self.to_v_lora, self.to_out_lora]:
                for name, param in lora_layer.named_parameters():
                    param.requires_grad = False
        elif self.i_num % 2 == 0:
            for lora_layer in [self.to_q_lora, self.to_k_lora, self.to_v_lora, self.to_out_lora]:
                for name, param in lora_layer.named_parameters():
                    param.requires_grad = True
        elif self.i_num % 2 == 1:
            for lora_layer in [self.to_q_lora, self.to_k_lora, self.to_v_lora, self.to_out_lora]:
                for name, param in lora_layer.named_parameters():
                    param.requires_grad = False

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        scaling = self.lora_alpha / self.lora_rank

        ##############global attention##############
        first_query = attn.head_to_batch_dim(query)
        first_key = attn.head_to_batch_dim(key)
        first_value = attn.head_to_batch_dim(value)

        dtype = first_query.dtype
        if attn.upcast_attention:
            first_query = first_query.float()
            first_key = first_key.float()
        
        if attention_mask is None:
            baddbmm_input = torch.empty(
                first_query.shape[0], first_query.shape[1], first_key.shape[1], dtype=first_query.dtype, device=first_query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            first_query,
            first_key.transpose(-1, -2),
            beta=beta,
            alpha=attn.scale,
        )
        del baddbmm_input

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        ori_hidden_states = torch.bmm(attention_probs, first_value)
        ori_hidden_states = attn.batch_to_head_dim(ori_hidden_states)

        sequence_length = query.size(1)
        fro_query, fro_key, fro_value = self._apply_lora(hidden_states, sequence_length, query, key, value, scaling)

        all_msk_mod = self.all_msk_mod
        
        fro_query = attn.head_to_batch_dim(fro_query)
        fro_key = attn.head_to_batch_dim(fro_key)
        fro_value = attn.head_to_batch_dim(fro_value)

        dtype = fro_query.dtype
        if attn.upcast_attention:
            fro_query = fro_query.float()
            fro_key = fro_key.float()
        
        if attention_mask is None:
            baddbmm_input = torch.empty(
                fro_query.shape[0], fro_query.shape[1], fro_key.shape[1], dtype=fro_query.dtype, device=fro_query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            fro_query,
            fro_key.transpose(-1, -2),
            beta=beta,
            alpha=attn.scale,
        )
        del baddbmm_input

        if attn.upcast_softmax:
            attention_scores = attention_scores.float()
            
        mask = all_msk_mod[int(attention_scores.size(1))].repeat(attn.heads,1,1)
        attention_scores = attention_scores + mask
        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        ori_hidden_states1 = torch.bmm(attention_probs, fro_value)
        ori_hidden_states1 = attn.batch_to_head_dim(ori_hidden_states1)

        # linear proj
        original_hidden_states = attn.to_out[0](ori_hidden_states)
        hidden_states_delta = self.to_out_lora(ori_hidden_states1).to(ori_hidden_states.device)

        original_hidden_states = hidden_states_delta * scaling + original_hidden_states

        # dropout
        hidden_states = attn.to_out[1](original_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor


        return hidden_states


def prepare_for_inference(
    model, device: torch.device, dtype: torch.dtype,
    lora_rank: int = 64, lora_alpha: float = 1.0
):
    attn_procs = {}
    attn_modules = [module for module in torch_dfs(model.up_blocks) if isinstance(module, BasicTransformerBlock)]
    i = 0
    for name in model.attn_processors.keys():
        if "up_blocks" in name and name.endswith("attn1.processor"):
            attn_processor = LoRAAttnMaskProcessor(
                device=device, 
                dtype=dtype,
                lora_rank=lora_rank, 
                lora_alpha=lora_alpha,
                latent_dim=attn_modules[i].attn1.to_q.base_layer.in_features
            )
            i = i + 1
            # block.attn1.set_processor(attn_processor)
            attn_procs[name] = attn_processor
        else:
            attn_procs[name] = model.attn_processors[name]
    model.set_attn_processor(attn_procs)

def get_processor_state_dict(model):
    """Save trainable parameters of processors to a checkpoint."""
    processor_state_dict = {}

    for name in model.attn_processors.keys():
        if name.endswith("attn1.processor"):
            processor = model.attn_processors[name]
            for attr_name in ["to_q_lora", "to_k_lora", "to_v_lora", "to_out_lora"]:
                if hasattr(processor, attr_name):
                    lora_layer = getattr(processor, attr_name)
                    for param_name, param in lora_layer.named_parameters():
                        key = f"{name}.{attr_name}.{param_name}"
                        processor_state_dict[key] = param.data.clone()

    return processor_state_dict

def load_processor_state_dict(model, processor_state_dict):
    """Load trainable parameters of processors from a checkpoint."""
    for name in model.attn_processors.keys():
        if name.endswith("attn1.processor"):
            processor = model.attn_processors[name]
            for attr_name in ["to_q_lora", "to_k_lora", "to_v_lora", "to_out_lora"]:
                if hasattr(processor, attr_name):
                    lora_layer = getattr(processor, attr_name)
                    for param_name, param in lora_layer.named_parameters():
                        key = f"{name}.{attr_name}.{param_name}"
                        if key in processor_state_dict:

                            param.data.copy_(processor_state_dict[key])
                        else:
                            raise KeyError(f"Missing key {key} in checkpoint.")