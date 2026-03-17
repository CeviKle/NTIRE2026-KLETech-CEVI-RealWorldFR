import os
import sys
import cv2
import glob
import torch

try:
    torch._C._jit_set_nvfuser_enabled(False)
except Exception:
    pass

from torchvision.transforms.functional import normalize
from omegaconf import OmegaConf
from torchvision import transforms
from accelerate.utils import set_seed

from models.team01_AllForFace.util import imwrite, img2tensor, tensor2img, load_model_from_url
from models.team01_AllForFace.FidelityGenerationModel import FidelityModel
from models.team01_AllForFace.ldm.cldm import ControlLDM
from models.team01_AllForFace.ldm.gaussian_diffusion import Diffusion
from models.team01_AllForFace.ldm.pipeline import DiffusionPipe
from models.team01_AllForFace.ram.caption import RAMCaptioner
from models.team06_KLETechCEVI.semantic_wavelet_refiner import SemanticWaveletRefiner


def main(model_dir, input_path, output_path, device):

    pos_prompt = ''
    neg_prompt = 'low quality, blurry, low-resolution, noisy, unsharp, weird textures, artifacts'
    steps = 50
    seed = 231

    fidelity_path      = 'fidelity_model.pth'
    natural_path       = 'naturalness_model.pt'
    sd_path            = 'v2-1_512-ema-pruned.ckpt'
    control_path       = 'v2.pth'
    caption_path       = 'ram_plus_swin_large_14m.pth'
    text_encoder_type  = 'bert-base-uncased'
    bisenet_path       = os.path.join(model_dir, '79999_iter.pth')
    wavelet_checkpoint = os.path.join(model_dir, 'semantic_best.pth')

    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(output_path, exist_ok=True)

    if input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
        input_img_list = [input_path]
    else:
        if input_path.endswith('/'):
            input_path = input_path[:-1]
        input_img_list = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))

    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError('No input image/folder found.\n')

    fidelity_model = FidelityModel(
        out_size=512, num_style_feat=512, channel_multiplier=1,
        decoder_load_path=None, fix_decoder=True, num_mlp=8,
        input_is_latent=True, different_w=True, narrow=1, sft_half=False)
    fidelity_model.load_state_dict(torch.load(os.path.join(model_dir, fidelity_path)))
    fidelity_model.eval().to(device)
    print("Loaded Stage 1: Fidelity model")

    enhance_model = torch.load(os.path.join(model_dir, natural_path))
    enhance_model.eval().to(device)
    print("Loaded Stage 3: Naturalness model")

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'team01_AllForFace', 'ldm', 'cldm.yaml'
    )
    ldm_config = OmegaConf.load(config_path).get("params", dict())

    control_net = ControlLDM(
        ldm_config["unet_cfg"], ldm_config["vae_cfg"], ldm_config["clip_cfg"],
        ldm_config["controlnet_cfg"], ldm_config["latent_scale_factor"])
    control_net.load_pretrained_sd(load_model_from_url(os.path.join(model_dir, sd_path)))
    control_net.load_controlnet_from_ckpt(load_model_from_url(os.path.join(model_dir, control_path)))
    control_net.eval().to(device)
    control_net.cast_dtype(torch.float32)

    diffusion = Diffusion().to(device)

    captioner = RAMCaptioner(
        pretrained_path=os.path.join(model_dir, caption_path),
        text_encoder_type=os.path.join(model_dir, text_encoder_type),
        device=device)
    print("Loaded Stage 2: Diffusion model")

    wavelet_refiner = SemanticWaveletRefiner(bisenet_path=bisenet_path).to(device)
    if os.path.exists(wavelet_checkpoint):
        wavelet_refiner.load_state_dict(torch.load(wavelet_checkpoint, map_location=device))
        print(f"Loaded Wavelet Refiner from {wavelet_checkpoint}")
    else:
        print(f"Wavelet checkpoint not found at {wavelet_checkpoint}")
    wavelet_refiner.eval()

    for i, img_path in enumerate(input_img_list):
        img_name = os.path.basename(img_path)
        basename, _ = os.path.splitext(img_name)
        print(f'[{i+1}/{test_img_num}] Processing: {img_name}')

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = img2tensor(img / 255., bgr2rgb=True, float32=True)
        normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            s1, _ = fidelity_model(img)
            s1 = torch.clamp(s1, min=-1, max=1)

            prompt = captioner(transforms.ToPILImage()(s1.squeeze(0)))
            pos_prompt_full = ", ".join([t for t in [prompt, pos_prompt] if t])
            s2 = DiffusionPipe(control_net, diffusion, (s1 + 1.0) / 2.0,
                               steps, pos_prompt_full, neg_prompt, device)
            s2 = torch.clamp(s2, min=0, max=1)

            s2 = wavelet_refiner(s2)

            s3 = enhance_model(s2)
            s3 = torch.clamp(s3, min=0, max=1)

            restored_face = tensor2img(s3, rgb2bgr=True, min_max=(0, 1))
            imwrite(restored_face, os.path.join(output_path, f'{basename}.png'))

    print(f'\nAll results saved to {output_path}')
