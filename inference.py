import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import time
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from magicbokeh import model_test
from utils.wavelet_color_fix import adain_color_fix

tensor_transforms = transforms.Compose([
                    transforms.ToTensor(),
                ])

def adjust_depth_with_mask(depth_map_pil, mask_pil):

    depth_map_np = np.array(depth_map_pil, dtype=np.float32)
    mask_np = np.array(mask_pil, dtype=np.uint8)
    depth_map_tensor = torch.tensor(depth_map_np)
    mask_tensor = torch.tensor(mask_np)

    assert depth_map_tensor.shape == mask_tensor.shape, \
        "Mask and Depth map must have the same shape."
    
    mask = mask_tensor > 0.5
    masked_depth_values = depth_map_tensor[mask]
    if masked_depth_values.numel() == 0:
        return 0
    
    focus_value = torch.median(masked_depth_values).item() 

    return focus_value

def extract_patches(images, patch_size=(512, 512), stride=256):

    batch_size, channels, height, width = images.shape
    patches = []
    positions = []

    for i in range(0, height - patch_size[0] + 1, stride):
        for j in range(0, width - patch_size[1] + 1, stride):
            patch = images[:, :, i:i+patch_size[0], j:j+patch_size[1]]
            patches.append(patch)
            positions.append((i, j))

    # edge patches
    if (height % stride != 0) or (width % stride != 0):
        for j in range(0, width - patch_size[1] + 1, stride):
            if height % stride != 0:
                patch = images[:, :, height-patch_size[0]:height, j:j+patch_size[1]]
                patches.append(patch)
                positions.append((height-patch_size[0], j))
        
        for i in range(0, height - patch_size[0] + 1, stride):
            if width % stride != 0:
                patch = images[:, :, i:i+patch_size[0], width-patch_size[1]:width]
                patches.append(patch)
                positions.append((i, width-patch_size[1]))

        if (height % stride != 0) and (width % stride != 0):
            patch = images[:, :, height-patch_size[0]:height, width-patch_size[1]:width]
            patches.append(patch)
            positions.append((height-patch_size[0], width-patch_size[1]))
    return torch.stack(patches), positions

def generate_weight_matrix(patch_size, overlap, position, image_shape):
    
    batch_size, channels, height, width = image_shape
    h_start, w_start = position
    weight_h = torch.ones(patch_size[0])
    weight_v = torch.ones(patch_size[1])

    # horizontal
    if h_start < height - patch_size[0]:
        weight_h[-overlap:] = torch.linspace(1, 0, overlap)

    if h_start > 0:
        weight_h[:overlap] = torch.linspace(0, 1, overlap)

    # vertical
    if w_start < width - patch_size[1]:
        weight_v[-overlap:] = torch.linspace(1, 0, overlap)

    if w_start > 0:
        weight_v[:overlap] = torch.linspace(0, 1, overlap)

    weight_matrix = torch.matmul(weight_h.unsqueeze(1), weight_v.unsqueeze(0))
    
    return weight_matrix


def reconstruct_from_patches(patches, original_shape, positions, patch_size=(512, 512), overlap=256):
    
    batch_size, channels, height, width = original_shape
    output = torch.zeros(batch_size, channels, height, width).to(patches.device)
    weights_sum = torch.zeros(batch_size, channels, height, width).to(patches.device)

    for i, pos in enumerate(positions):
        h_start, w_start = pos
        current_patch = patches[i]
        weight_matrix = generate_weight_matrix(patch_size, overlap, pos, original_shape).to(patches.device)

        h_end = h_start + patch_size[0]
        w_end = w_start + patch_size[1]

        output[:, :, h_start:h_end, w_start:w_end] += current_patch * weight_matrix
        weights_sum[:, :, h_start:h_end, w_start:w_end] += weight_matrix

    mask = weights_sum > 0

    output[mask] /= weights_sum[mask]

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./test_data/inputs')
    parser.add_argument('--depth_dir', type=str, default='./test_data/depth')
    parser.add_argument('--mask_dir', type=str, default='./test_data/mask')
    parser.add_argument('--output_dir', type=str, default='./test_data/output')
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--mixed_precision', type=str, choices=['fp16', 'fp32'], default='fp16')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = model_test(args)

    image_names = sorted(glob.glob(f'{args.input_dir}/*.jpg') + glob.glob(f'{args.input_dir}/*.png') + glob.glob(f'{args.input_dir}/*.JPG'))
    print(f"There are {len(image_names)} images.")

    for image_name in image_names:
        print(f"Processing {image_name}")
        input_image = Image.open(image_name).convert('RGB')
        bname = os.path.basename(image_name)
        name, ext = os.path.splitext(bname)

        depth_path = os.path.join(args.depth_dir, f"{name}.npy")
        depth_npy = np.load(depth_path)
        depth_npy = (depth_npy * 255).astype(np.uint8)
        input_depth = Image.fromarray(depth_npy, mode='L')

        mask_path = os.path.join(args.mask_dir, f"{name}.png")
        input_mask = Image.open(mask_path).convert('L')

        focus_values = adjust_depth_with_mask(input_depth, input_mask)
        focus_values = focus_values / 255.
        print(f"focus value: {focus_values}")

        lq = tensor_transforms(input_image).unsqueeze(0).to('cuda')
        depth = tensor_transforms(input_depth).unsqueeze(0).to('cuda')
        validation_prompt = ""

        with torch.no_grad():
            disp = torch.abs(depth - focus_values)
            K_focus_value = args.K
            disp_k = disp * (K_focus_value / 32)
            image_patches, image_positions = extract_patches(lq)
            mask_patches, mask_positions = extract_patches(disp_k)
            output_patches = []

            for img_patch, mask_patch in zip(image_patches, mask_patches):
                mask_copy = 1 - (mask_patch > 0.1).float()
                img_patch = img_patch * 2 - 1
                mask_patch = mask_patch * 2 - 1
                mask_copy = mask_copy * 2 - 1
                output_image = model(img_patch, mask_patch, mask_copy, validation_prompt)
                output_patches.append(output_image)

            output_image = reconstruct_from_patches(torch.stack(output_patches), lq.shape, positions=image_positions)
            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            output_pil = adain_color_fix(target=output_pil, source=input_image)
            save_path = os.path.join(args.output_dir, f"{name}.png")
            output_pil.save(save_path)