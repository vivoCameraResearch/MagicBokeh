import os
import sys
sys.path.append(os.getcwd())
import glob
import argparse
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from models.LDF_net import LDF
import cv2
from depth_anything_v2.dpt import DepthAnythingV2

LDF_transforms = transforms.Compose([
            transforms.Resize((352, 352)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class LDFInference:
    def __init__(self, resnet_path=None, model_path=None, device="cuda"):
        self.device = device
        self.model = LDF(resnet_path=resnet_path, model_path=model_path)
        self.model.train(False)
        self.model.to(self.device)


    def infer(self, image_pil):
        width, height = image_pil.size
        image = LDF_transforms(image_pil).unsqueeze(0).to(self.device)
        outb1, outd1, out1, outb2, outd2, out2 = self.model(image, (torch.tensor([height]), torch.tensor([width])))
        out = out2
        mask = torch.sigmoid(out[0,0]).cpu().detach().numpy()
        mask = np.round(mask * 255).astype('uint8')
        return mask

class DADepthInference:
    def __init__(self, encoder='vitl', ckpt_path=None, device="cuda"):
        self.device = device
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.model = DepthAnythingV2(**model_configs[encoder])
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    def infer(self, image_bgr):
        depth = self.model.infer_image(image_bgr)
        depth = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8))
        return depth.astype(np.float32)

def main(args):
    os.makedirs(args.depth_dir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)

    if os.path.isdir(args.input_dir):
        image_names = sorted(glob.glob(f'{args.input_dir}/*.jpg') + glob.glob(f'{args.input_dir}/*.png'))
    else:
        image_names = [args.input_dir]
    print(f"Found {len(image_names)} images.")

    #Init LDF models
    print("Loading LDF model...")
    ldf_model = LDFInference(resnet_path=args.resnet_path, model_path=args.model_path, device=args.device)

    # Init depth models
    print("Loading DepthAnythingV2 model...")
    depth_model = DADepthInference(encoder=args.encoder, ckpt_path=args.depth_ckpt, device=args.device)

    # Process images
    for image_name in image_names:
        basename = os.path.basename(image_name)
        name = os.path.splitext(basename)[0]
        image_pil = Image.open(image_name).convert("RGB")
        image_bgr = cv2.imread(image_name)

        # LDF infer
        ori_width, ori_height = image_pil.size
        input_mask = ldf_model.infer(image_pil)
        mask_save_path = os.path.join(args.mask_dir, f"{name}.png")
        Image.fromarray(input_mask).save(mask_save_path)

        # depth infer
        depth = depth_model.infer(image_bgr)
        depth_save_path = os.path.join(args.depth_dir,f"{name}.npy")
        np.save(depth_save_path, depth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./test_data/inputs")
    parser.add_argument("--depth_dir", type=str, default="./test_data/depth")
    parser.add_argument("--mask_dir", type=str, default="./test_data/mask")
    parser.add_argument("--resnet_path", type=str, default="./resnet50-19c8e357.pth")
    parser.add_argument("--model_path", type=str, default="./model-40")
    parser.add_argument("--depth_ckpt", type=str, default="./depth_model/DAdepth.pth")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)