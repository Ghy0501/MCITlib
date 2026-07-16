#!/usr/bin/env python

import argparse
import torch
import os
import sys
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from videollava.model.builder import load_pretrained_model

# from videollava.mm_utils import get_model_name_from_path
def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
# from videollava.utils import disable_torch_init
def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def extract_weights(model_config_path, output_dir, energy_threshold, target_modules):
    """
    Extract principal directions from model weights using SVD.
    This approximates the gradient directions of the pretrained model (Task 0).
    
    Args:
        model_config_path: Path to model config JSON file
        output_dir: Directory to save the extracted weight subspace
        energy_threshold: Energy threshold for selecting principal components
        target_modules: Comma-separated list of target module names
    """
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = model_config.get('model_name')
    model_base = None
    
    print(f"Loading model from {model_path}...")
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    model = model.to(device)
    
    target_module_names = target_modules.split(',')
    weights_dict = {}
    
    for name, param in model.named_parameters():
        if name.startswith('model.layers.') and any(target in name for target in target_module_names) and 'weight' in name:
            weights_dict[name] = param.data.cpu().clone()
    
    print(f"Performing SVD decomposition with energy threshold {energy_threshold}...")
    
    svd_init_dict = {}
    rank_info = {}
    
    device = torch.device('cuda:0')
    
    for weight_name, weight_tensor in weights_dict.items():
        # weight_tensor shape: [out_features, in_features] for linear layers
        # Convert to float32 if needed (SVD doesn't support float16)
        original_dtype = weight_tensor.dtype
        if weight_tensor.dtype in [torch.float16, torch.bfloat16]:
            weight_tensor = weight_tensor.float()
        
        weight_tensor = weight_tensor.to(device)
        
        weight_tensor_t = weight_tensor.t()  # [in_features, out_features]
        
        U, S, Vh = torch.linalg.svd(weight_tensor_t, full_matrices=False)
        
        sval_total = (S**2).sum()
        sval_ratio = (S**2) / sval_total
        
        r = torch.sum(torch.cumsum(sval_ratio, dim=0) < energy_threshold).item()
        actual_rank = max(r, 1)
        
        U_truncated = U[:, :actual_rank]  # [in_features, actual_rank]
        svd_result = U_truncated.t()  # [actual_rank, in_features]
        
        svd_result = svd_result.cpu()
        if original_dtype == torch.float16:
            svd_result = svd_result.half()
        elif original_dtype == torch.bfloat16:
            svd_result = svd_result.to(torch.bfloat16)
        
        svd_init_dict[weight_name] = svd_result
        rank_info[weight_name] = actual_rank
        
        print(f"  Weight {weight_name}: original shape {weight_tensor.shape} -> principal subspace shape {svd_result.shape}, rank={actual_rank}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "gradients_principle_subspace.pt"
    
    save_dict = {
        'gradients': svd_init_dict,
        'rank_info': rank_info,
        'svd_mode': 'energy',
        'energy_threshold': energy_threshold,
        'source': 'weights'
    }
    
    torch.save(save_dict, output_file)
    print(f"Weight principal subspace saved to {output_file}")
    
    avg_rank = sum(rank_info.values()) / len(rank_info)
    print(f"Average rank used: {avg_rank:.2f}")
    print(f"Rank range: [{min(rank_info.values())}, {max(rank_info.values())}]")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract principal directions from model weights")
    parser.add_argument("--model-config", type=str, required=True,
                        help="Path to model config JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the extracted weight subspace")
    parser.add_argument("--energy-threshold", type=float, default=0.99,
                        help="Energy threshold for selecting principal components")
    parser.add_argument("--target-modules", type=str, 
                        default="down_proj,q_proj,v_proj,o_proj,gate_proj,up_proj,k_proj",
                        help="Comma-separated list of target module names")
    
    args = parser.parse_args()
    
    extract_weights(
        model_config_path=args.model_config,
        output_dir=args.output_dir,
        energy_threshold=args.energy_threshold,
        target_modules=args.target_modules
    )