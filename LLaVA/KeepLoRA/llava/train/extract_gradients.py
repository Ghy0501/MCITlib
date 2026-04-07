#!/usr/bin/env python

import argparse
import torch
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
from llava.train.train import make_supervised_data_module, DataArguments
from torch.utils.data import DataLoader, Subset


def extract_gradients(args):
    with open(args.data_config, 'r') as f:
        data_config = json.load(f)
    with open(args.model_config, 'r') as f:
        model_config = json.load(f)
    with open(args.train_config, 'r') as f:
        train_config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    disable_torch_init()
    model_path = train_config.get('model_path')
    model_base = model_config.get('model_name')
    model_name = get_model_name_from_path(model_path)
    if "merged" in model_path or model_path == model_base:
        model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    model = model.to(device)
    model.train()
    
    data_args = DataArguments(
        data_path=data_config['train_path'],
        image_folder=data_config['train_folder'],
        lazy_preprocess=True,
        is_multimodal=True,
        image_aspect_ratio='pad'
    )
    data_args.image_processor = image_processor
    data_args.mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    data_args.mm_use_im_patch_token = getattr(model.config, 'mm_use_im_patch_token', True)
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    dataset = data_module['train_dataset']
    
    total_samples = len(dataset)
    num_samples = int(total_samples * args.data_ratio)
    indices = torch.linspace(0, total_samples - 1, num_samples).long().tolist()

    chunk_size = len(indices) // args.num_chunks
    start_idx = args.chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if args.chunk_idx < args.num_chunks - 1 else len(indices)
    chunk_indices = indices[start_idx:end_idx]
    
    chunk_subset = Subset(dataset, chunk_indices)
    data_loader = DataLoader(
        chunk_subset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=data_module['data_collator']
    )
    
    target_module_names = args.target_modules.split(',')
    
    for param in model.parameters():
        param.requires_grad = False

    accumulated_grads = {}
    for name, param in model.named_parameters():
        if name.startswith('model.layers.') and any(target in name for target in target_module_names) and 'weight' in name:
            accumulated_grads[name] = torch.zeros_like(param.data)
            param.requires_grad = True
    
    num_batches = 0
    for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Rank {local_rank}")):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        
        if 'images' in batch:
            images = batch['images']
            if isinstance(images, list):
                images = [img.to(device=device, dtype=torch.float16, non_blocking=True) for img in images]
            else:
                images = images.to(device=device, dtype=torch.float16, non_blocking=True)
        else:
            images = None
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, images=images)
        loss = outputs.loss
        loss.backward()
        
        num_batches += 1
        for name, param in model.named_parameters():
            if name in accumulated_grads and param.grad is not None:
                accumulated_grads[name] = accumulated_grads[name] * ((num_batches - 1) / num_batches) + param.grad.data * (1.0 / num_batches)
        
        model.zero_grad()
    
    accumulated_grads_cpu = {name: grad.cpu() for name, grad in accumulated_grads.items()}
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gradients_chunk_{args.chunk_idx}_of_{args.num_chunks}.pt"
    
    torch.save({
        'gradients': accumulated_grads_cpu,
        'num_batches': num_batches,
        'num_samples': len(chunk_subset),
    }, output_file)
    
    return output_file


def merge_gradient_chunks(output_dir, num_chunks, train_config_path, svd_mode):
    """
    Merge gradient chunks and perform SVD decomposition.
    
    Args:
        output_dir: Directory containing gradient chunks
        num_chunks: Number of chunks to merge
        model_config_path: Path to model config JSON file to read rank and energy_threshold
        svd_mode: 'fixed_rank' or 'energy' - determines which mode to use
    """
    output_dir = Path(output_dir)
    all_grads = []
    chunk_files = []
    
    with open(train_config_path, 'r') as f:
        train_config = json.load(f)
    
    fixed_rank = train_config.get('rank')
    energy_threshold = train_config.get('energy_threshold')
    space_path = train_config.get('space_path')
    
    if svd_mode == 'fixed_rank':
        print(f"Using fixed LoRA rank: {fixed_rank} from config file")
        if space_path:
            principle_subspace_file = Path(space_path) / "gradients_principle_subspace.pt"
            print(f"Loading previous task's principle subspace from {principle_subspace_file}...")
            prev_data = torch.load(principle_subspace_file, map_location='cpu')
            prev_gradients = prev_data['gradients']
            print(f"Loaded {len(prev_gradients)} previous principle subspace tensors!")
        else:
            print("No space_path specified, using standard SVD without subspace projection")
            prev_gradients = None
    elif svd_mode == 'energy':
        print(f"Using energy-based rank selection with threshold: {energy_threshold}!")
        if space_path:
            principle_subspace_file = Path(space_path) / "gradients_principle_subspace.pt"
            print(f"Loading previous task's principle subspace from {principle_subspace_file}...")
            prev_data = torch.load(principle_subspace_file, map_location='cpu')
            prev_gradients = prev_data['gradients']
            print(f"Loaded {len(prev_gradients)} previous principle subspace tensors!")
        else:
            print("No space_path specified, this is the first task")
            prev_gradients = None
    else:
        raise ValueError(f"Unknown svd_mode: {svd_mode}. Must be 'fixed_rank' or 'energy'")
    
    for chunk_idx in range(num_chunks):
        chunk_file = output_dir / f"gradients_chunk_{chunk_idx}_of_{num_chunks}.pt"
        if not chunk_file.exists():
            print(f"Warning: {chunk_file} not found, skipping...")
            continue
        data = torch.load(chunk_file, map_location='cpu')
        all_grads.append(data['gradients'])
        chunk_files.append(chunk_file)
    
    if len(all_grads) == 0:
        raise ValueError(f"No gradient chunks found in {output_dir}")
    
    merged_grads = {}
    for name in all_grads[0].keys():
        merged_grads[name] = torch.zeros_like(all_grads[0][name])
        for grads in all_grads:
            if name in grads:
                merged_grads[name] += grads[name]
        merged_grads[name] /= len(all_grads)
    
    print(f"Merged {len(merged_grads)} gradient tensors")
    
    if svd_mode == 'fixed_rank':
        print(f"Performing SVD decomposition with fixed rank {fixed_rank}...")
    elif svd_mode == 'energy':
        print(f"Performing SVD decomposition with energy threshold {energy_threshold}...")
    
    svd_init_dict = {}
    rank_info = {}
    
    device = torch.device('cuda:0')
    
    for grad_name, grad_tensor in merged_grads.items():
        if 'weight' in grad_name:
            # grad_tensor shape: [out_features, in_features] for linear layers
            # Convert to float32 if needed (SVD doesn't support float16)
            original_dtype = grad_tensor.dtype
            if grad_tensor.dtype == torch.float16:
                grad_tensor = grad_tensor.float()
            
            grad_tensor = grad_tensor.to(device)
            
            grad_tensor_t = grad_tensor.t()  # [in_features, out_features]
            
            if svd_mode == 'fixed_rank':
                if prev_gradients is not None and grad_name in prev_gradients:
                    U_p = prev_gradients[grad_name].to(device=device, dtype=grad_tensor_t.dtype)  # [prev_rank, in_features]
                    # grad_tensor_t - U_p.t * U_p * grad_tensor_t
                    # grad_tensor_t: [in_features, out_features]
                    # U_p: [prev_rank, in_features]
                    grad_tensor_t_projected = grad_tensor_t - torch.mm(U_p.t(), torch.mm(U_p, grad_tensor_t))
                    print(f"  Applied subspace projection for {grad_name}: U_p shape {U_p.shape}")
                    U, S, V = torch.svd_lowrank(grad_tensor_t_projected, q=fixed_rank)
                else:
                    U, S, V = torch.svd_lowrank(grad_tensor_t, q=fixed_rank)
                
                # U shape: [in_features, fixed_rank]
                svd_result = U.t()  # [fixed_rank, in_features]
                actual_rank = fixed_rank
                
            elif svd_mode == 'energy':
                activation = grad_tensor_t.clone()  # [in_features, out_features]
                
                U1, S1, Vh1 = torch.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                
                if prev_gradients is not None and grad_name in prev_gradients:
                    feature_tensor = prev_gradients[grad_name].to(device=device, dtype=activation.dtype)  # [prev_rank, in_features]
                    
                    # act_hat = activation - feature_tensor.t * feature_tensor * activation
                    act_hat = activation - torch.mm(feature_tensor.t(), torch.mm(feature_tensor, activation))
                    
                    U, S, Vh = torch.linalg.svd(act_hat, full_matrices=False)
                    
                    sval_hat = (S**2).sum()
                    sval_ratio = (S**2) / sval_total
                    accumulated_sval = (sval_total - sval_hat) / sval_total
                    
                    r = 0
                    for ii in range(sval_ratio.shape[0]):
                        if accumulated_sval < energy_threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    
                    if r == 0:
                        print(f'  Skip updating {grad_name}, previous subspace is sufficient')
                        svd_result = feature_tensor  # [prev_rank, in_features]
                        actual_rank = feature_tensor.shape[0]
                    else:
                        Ui = torch.hstack((feature_tensor.t(), U[:, 0:r]))  # [in_features, prev_rank+r]
                        
                        if Ui.shape[1] > Ui.shape[0]:
                            Ui = Ui[:, 0:Ui.shape[0]]
                        
                        svd_result = Ui.t()  # [new_rank, in_features]
                        actual_rank = svd_result.shape[0]
                        print(f'  Updated {grad_name}: prev_rank={feature_tensor.shape[0]}, added_rank={r}, new_rank={actual_rank}')
                else:
                    energy = S1 ** 2
                    total_energy = energy.sum()
                    cumulative_energy = torch.cumsum(energy, dim=0)
                    energy_ratio = cumulative_energy / total_energy
                    
                    actual_rank = (energy_ratio >= energy_threshold).nonzero(as_tuple=True)[0][0].item() + 1
                    
                    U_truncated = U1[:, :actual_rank]  # [in_features, actual_rank]
                    svd_result = U_truncated.t()  # [actual_rank, in_features]
                    print(f'  First task {grad_name}: rank={actual_rank}')
            
            svd_result = svd_result.cpu()
            if original_dtype == torch.float16:
                svd_result = svd_result.half()
            
            svd_init_dict[grad_name] = svd_result
            rank_info[grad_name] = actual_rank
            
            print(f"  SVD for {grad_name}: grad shape {grad_tensor.shape} -> A shape {svd_init_dict[grad_name].shape}")
    
    if svd_mode == 'fixed_rank':
        merged_file = output_dir / "gradients_merged.pt"
    elif svd_mode == 'energy':
        merged_file = output_dir / "gradients_principle_subspace.pt"
    
    save_dict = {
        'gradients': svd_init_dict,
        'rank_info': rank_info,
        'svd_mode': svd_mode
    }
    
    if svd_mode == 'fixed_rank':
        save_dict['fixed_rank'] = fixed_rank
    elif svd_mode == 'energy':
        save_dict['energy_threshold'] = energy_threshold
    
    torch.save(save_dict, merged_file)
    print(f"SVD-decomposed gradients saved to {merged_file}")
    
    if svd_mode == 'energy':
        avg_rank = sum(rank_info.values()) / len(rank_info)
        print(f"Average rank used: {avg_rank:.2f}")
        print(f"Rank range: [{min(rank_info.values())}, {max(rank_info.values())}]")

    for chunk_file in chunk_files:
        try:
            chunk_file.unlink()
            print(f"Deleted {chunk_file}")
        except Exception as e:
            print(f"Warning: Failed to delete {chunk_file}: {e}")
    
    return merged_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--train-config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./checkpoints/gradients")
    parser.add_argument("--data-ratio", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--target-modules", type=str, default="down_proj,q_proj,v_proj,o_proj,gate_proj,up_proj,k_proj")
    parser.add_argument("--merge-only", action="store_true", help="Only merge and perform SVD, don't extract gradients")
    parser.add_argument("--svd-mode", type=str, default="fixed_rank", choices=["fixed_rank", "energy"],
                        help="SVD mode: 'fixed_rank' uses fixed rank, 'energy' uses energy threshold")
    
    args = parser.parse_args()
    
    if args.merge_only:
        merge_gradient_chunks(args.output_dir, args.num_chunks, args.train_config, svd_mode=args.svd_mode)
    else:
        extract_gradients(args)
