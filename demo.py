import os
import sys
import glob
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

sys.path.append(os.getcwd())

from magicbokeh import model_test
from utils.wavelet_color_fix import adain_color_fix
from depth_anything_v2.dpt import DepthAnythingV2


tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

LDF_transforms = transforms.Compose([
    transforms.Resize((352, 352)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@dataclass
class AppState:
    image_pil: Optional[Image.Image] = None
    clicked_point: Optional[Tuple[int, int]] = None


class MagicBokehApp:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tensor_transforms = tensor_transforms
        self.depth_model = self._load_depth_model()
        self.magicbokeh = model_test(args)

    def _load_depth_model(self):
        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }

        encoder = self.args.depth_encoder
        model_depth = DepthAnythingV2(**model_configs[encoder])
        dav2 = torch.load(self.args.depth_model_path, map_location='cpu')
        state_dict = dav2["model"]
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_depth.load_state_dict(state_dict)
        model_depth = model_depth.to("cuda").eval()
        return model_depth

    def get_validation_prompt(self, image, mask):
        lq = self.tensor_transforms(image).unsqueeze(0).to(self.device)
        msk = self.tensor_transforms(mask).unsqueeze(0).to(self.device)
        validation_prompt = "high quality,"
        return validation_prompt, lq, msk

    def adjust_depth_with_value(self, depth_map_pil, x_value, y_value):
        x = int(x_value)
        y = int(y_value)
        width, height = depth_map_pil.size
        if x < 0 or x >= width or y < 0 or y >= height:
            raise ValueError(f"Coordinates ({x}, {y}) are outside the image bounds ({width}x{height})")
        return depth_map_pil.getpixel((x, y))

    def extract_patches(self, images, patch_size=(512, 512), stride=256):
        batch_size, channels, height, width = images.shape
        patches = []
        positions = []

        for i in range(0, height - patch_size[0] + 1, stride):
            for j in range(0, width - patch_size[1] + 1, stride):
                patch = images[:, :, i:i + patch_size[0], j:j + patch_size[1]]
                patches.append(patch)
                positions.append((i, j))

        if (height % stride != 0) or (width % stride != 0):
            for j in range(0, width - patch_size[1] + 1, stride):
                if height % stride != 0:
                    patch = images[:, :, height - patch_size[0]:height, j:j + patch_size[1]]
                    patches.append(patch)
                    positions.append((height - patch_size[0], j))

            for i in range(0, height - patch_size[0] + 1, stride):
                if width % stride != 0:
                    patch = images[:, :, i:i + patch_size[0], width - patch_size[1]:width]
                    patches.append(patch)
                    positions.append((i, width - patch_size[1]))

            if (height % stride != 0) and (width % stride != 0):
                patch = images[:, :, height - patch_size[0]:height, width - patch_size[1]:width]
                patches.append(patch)
                positions.append((height - patch_size[0], width - patch_size[1]))

        return torch.stack(patches), positions

    def generate_weight_matrix(self, patch_size, overlap, position, image_shape):
        batch_size, channels, height, width = image_shape
        h_start, w_start = position
        weight_h = torch.ones(patch_size[0])
        weight_v = torch.ones(patch_size[1])

        if h_start < height - patch_size[0]:
            weight_h[-overlap:] = torch.linspace(1, 0, overlap)
        if h_start > 0:
            weight_h[:overlap] = torch.linspace(0, 1, overlap)

        if w_start < width - patch_size[1]:
            weight_v[-overlap:] = torch.linspace(1, 0, overlap)
        if w_start > 0:
            weight_v[:overlap] = torch.linspace(0, 1, overlap)

        return torch.matmul(weight_h.unsqueeze(1), weight_v.unsqueeze(0))

    def reconstruct_from_patches(self, patches, original_shape, positions, patch_size=(512, 512), overlap=256):
        batch_size, channels, height, width = original_shape
        output = torch.zeros(batch_size, channels, height, width).to(patches.device)
        weights_sum = torch.zeros(batch_size, channels, height, width).to(patches.device)

        for i, pos in enumerate(positions):
            h_start, w_start = pos
            current_patch = patches[i]
            weight_matrix = self.generate_weight_matrix(patch_size, overlap, pos, original_shape).to(patches.device)
            h_end, w_end = h_start + patch_size[0], w_start + patch_size[1]
            output[:, :, h_start:h_end, w_start:w_end] += current_patch * weight_matrix
            weights_sum[:, :, h_start:h_end, w_start:w_end] += weight_matrix

        mask = weights_sum > 0
        output[mask] /= weights_sum[mask]
        return output

    def infer_depth(self, image_pil):
        w, h = image_pil.size
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image_bgr = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
        depth = self.depth_model.infer_image(image_bgr)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0
        return Image.fromarray(depth.astype("uint8"), mode="L")

    def draw_focus_point(self, image_pil, point, radius=8):
        canvas = image_pil.copy()
        draw = ImageDraw.Draw(canvas)
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="red", width=4)
        draw.line((x - radius * 2, y, x + radius * 2, y), fill="red", width=3)
        draw.line((x, y - radius * 2, x, y + radius * 2), fill="red", width=3)
        return canvas

    def prepare_image(self, image):
        if image is None:
            return None, None, "Please upload an image first."
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        state = AppState(image_pil=image, clicked_point=None)
        return state, image, "Image loaded. Please click on the image to select a focus point."

    def select_focus(self, evt: gr.SelectData, state: AppState):
        if state is None or state.image_pil is None:
            return state, None, "Please upload an image first."

        x, y = evt.index
        state.clicked_point = (int(x), int(y))
        preview = self.draw_focus_point(state.image_pil, state.clicked_point)
        return state, preview, f"Selected focus point: ({int(x)}, {int(y)}), try to get bokeh effects."

    def run_inference(self, state: AppState, k):
        if state is None or state.image_pil is None:
            return None, None, "Please upload an image first."
        if state.clicked_point is None:
            return None, None, "Please select a focus point by clicking on the image."

        input_image = state.image_pil.copy()
        ori_width, ori_height = input_image.size
        x_value, y_value = state.clicked_point

        input_depth = self.infer_depth(input_image)
        print("depth finish!")
        focus_value = self.adjust_depth_with_value(input_depth, x_value, y_value) / 255.0

        print(f"focus value: {focus_value}")

        if math.isnan(focus_value):
            focus_value = 1.0

        validation_prompt, lq, depth = self.get_validation_prompt(input_image, input_depth)

        with torch.no_grad():
            disp = torch.abs(depth - focus_value)
            k_focus_value = k
            disp = disp * (k_focus_value / 32)

            image_patches, image_positions = self.extract_patches(lq)
            mask_patches, _ = self.extract_patches(disp)
            output_patches = []

            for img_patch, mask_patch in zip(image_patches, mask_patches):
                mask_copy = 1 - (mask_patch > 0.1).float()
                img_patch = img_patch * 2 - 1
                mask_patch = mask_patch * 2 - 1
                mask_copy = mask_copy * 2 - 1
                output_image = self.magicbokeh(img_patch, mask_patch, mask_copy, prompt=validation_prompt)
                output_patches.append(output_image)

            output_image = self.reconstruct_from_patches(torch.stack(output_patches), lq.shape, positions=image_positions)
            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

            output_pil = adain_color_fix(target=output_pil, source=input_image)

        depth_vis = input_depth.convert("RGB")
        return output_pil, depth_vis, "View the result and depth map on the right."

    def build_ui(self):
        with gr.Blocks(title="MagicBokeh Interactive Demo") as demo:
            state = gr.State(AppState())

            gr.Markdown("""
                # 🎬 Towards Photorealistic and Efficient Bokeh Rendering via Diffusion Framework

                ## 🚀 Getting Started

                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">

                <div style="padding:16px; border:1px solid #e5e7eb; border-radius:10px;">
                <b>① Upload Image</b><br>
                Upload an LR image on the left.
                </div>

                <div style="padding:16px; border:1px solid #e5e7eb; border-radius:10px;">
                <b>② Select Focus</b><br>
                Click on the region you want to keep sharp. A red marker will appear.
                </div>

                <div style="padding:16px; border:1px solid #e5e7eb; border-radius:10px;">
                <b>③ Render Bokeh</b><br>
                Click the render button. The system applies bokeh rendering.
                </div>

                <div style="padding:16px; border:1px solid #e5e7eb; border-radius:10px;">
                <b>④ Preview Result</b><br>
                View the result and depth map on the right. Adjust parameters for variations.
                </div>

                </div>

                <br>

                <div style="background:#eef6ff; padding:14px; border-radius:8px; border:1px solid #c7ddff;">
                💡 <b>Tips:</b> Large images may take longer to process.
                </div>
            """)

            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Upload Image")
                    click_preview = gr.Image(type="pil", label="Select Focus", interactive=False)
                    slider = gr.Slider(1,32,value=32,step=1,label="Aperture Parameters")
                    status_text = gr.Textbox(label="Demo States", interactive=False)
                    run_button = gr.Button("Generate Bokeh Effects", variant="primary")
                with gr.Column():
                    output_image = gr.Image(type="pil", label="Output Image")
                    depth_image = gr.Image(type="pil", label="Depth Map")

            input_image.upload(
                fn=self.prepare_image,
                inputs=[input_image],
                outputs=[state, click_preview, status_text],
            )

            click_preview.select(
                fn=self.select_focus,
                inputs=[state],
                outputs=[state, click_preview, status_text],
            )

            run_button.click(
                fn=self.run_inference,
                inputs=[state, slider],
                outputs=[output_image, depth_image, status_text],
            )

        return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--depth_encoder", type=str, choices=["vits", "vitb", "vitl", "vitg"], default="vitl")
    parser.add_argument("--depth_model_path", type=str, required=True)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = MagicBokehApp(args)
    demo = app.build_ui()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=False)
