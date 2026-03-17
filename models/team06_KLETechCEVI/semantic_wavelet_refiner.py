#!/usr/bin/env python
# semantic_wavelet_refiner.py
# Semantic-Aware Wavelet Frequency Refiner
# Replaces uniform FGRC with region-specific correction per facial area

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import lpips

# ============================================
# BiSeNet Face Parser
# ============================================
class BiSeNetParser(nn.Module):
    """
    Wraps BiSeNet to produce per-region masks.
    BiSeNet classes:
      0: background  1: skin       2: l_brow    3: r_brow
      4: l_eye       5: r_eye      6: eye_g     7: l_ear
      8: r_ear       9: ear_r     10: nose     11: mouth
     12: u_lip      13: l_lip     14: neck     15: neck_l
     16: cloth      17: hair      18: hat
    """

    SKIN_CLASSES  = [1, 2, 3, 10]           # skin, brows, nose
    EYE_CLASSES   = [4, 5, 6]               # eyes + glasses
    MOUTH_CLASSES = [11, 12, 13]            # mouth, lips
    HAIR_CLASSES  = [17, 18]               # hair, hat
    BG_CLASSES    = [0, 7, 8, 9, 14, 15, 16]  # background, ears, neck, cloth

    def __init__(self, bisenet_path, device='cuda'):
        super().__init__()
        # Add face-parsing repo to path
        repo_dir = os.path.dirname(bisenet_path).replace('/res/cp', '')
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        from model import BiSeNet as _BiSeNet
        net = _BiSeNet(n_classes=19)
        net.load_state_dict(torch.load(bisenet_path, map_location='cpu'))
        net.eval()
        self.net = net

        # Normalization for BiSeNet input
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        x: [B, 3, H, W] in [0, 1]
        returns: dict of masks each [B, 1, H, W] in [0, 1]
        """
        B, C, H, W = x.shape

        # Normalize for BiSeNet
        mean = self.mean.to(x.device)
        std  = self.std.to(x.device)

        # BiSeNet expects 512x512
        x_resized = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        x_norm = (x_resized - mean) / std

        with torch.no_grad():
            out = self.net(x_norm)[0]  # [B, 19, 512, 512]
            parsed = out.argmax(dim=1)  # [B, 512, 512]

        # Build masks
        def make_mask(classes):
            m = torch.zeros(B, 1, 512, 512, device=x.device)
            for c in classes:
                m += (parsed == c).unsqueeze(1).float()
            m = m.clamp(0, 1)
            # Resize back to original resolution
            return F.interpolate(m, size=(H, W), mode='bilinear', align_corners=False)

        return {
            'skin':  make_mask(self.SKIN_CLASSES),
            'eye':   make_mask(self.EYE_CLASSES),
            'mouth': make_mask(self.MOUTH_CLASSES),
            'hair':  make_mask(self.HAIR_CLASSES),
            'bg':    make_mask(self.BG_CLASSES),
        }


# ============================================
# Haar Wavelet Transform
# ============================================
class HaarWavelet2D(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[[[1., 1.], [1., 1.]]]]) * 0.5
        lh = torch.tensor([[[[1., 1.], [-1., -1.]]]]) * 0.5
        hl = torch.tensor([[[[1., -1.], [1., -1.]]]]) * 0.5
        hh = torch.tensor([[[[1., -1.], [-1., 1.]]]]) * 0.5
        self.register_buffer('filters', torch.cat([ll, lh, hl, hh], dim=0))

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0: x = F.pad(x, (0, 0, 0, 1))
        if W % 2 != 0: x = F.pad(x, (0, 1, 0, 0))
        x_flat = x.reshape(B * C, 1, x.shape[2], x.shape[3])
        out = F.conv2d(x_flat, self.filters, stride=2)
        out = out.view(B, C, 4, x.shape[2]//2, x.shape[3]//2)
        return out[:,:,0], out[:,:,1], out[:,:,2], out[:,:,3]


class InverseHaarWavelet2D(nn.Module):
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[[[1., 1.], [1., 1.]]]]) * 0.5
        lh = torch.tensor([[[[1., 1.], [-1., -1.]]]]) * 0.5
        hl = torch.tensor([[[[1., -1.], [1., -1.]]]]) * 0.5
        hh = torch.tensor([[[[1., -1.], [-1., 1.]]]]) * 0.5
        self.register_buffer('filters', torch.cat([ll, lh, hl, hh], dim=0))

    def forward(self, LL, LH, HL, HH):
        B, C, H, W = LL.shape
        x = torch.stack([LL, LH, HL, HH], dim=2).view(B * C, 4, H, W)
        out = F.conv_transpose2d(x, self.filters, stride=2)
        return out.view(B, C, H * 2, W * 2)


# ============================================
# Region-Specific Tiny CNN
# ============================================
class RegionCNN(nn.Module):
    """Tiny CNN for one facial region — ~23K params each"""

    def __init__(self, in_channels=9, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, in_channels, 3, padding=1)
        )
        # Learnable scale per region
        self.scale = nn.Parameter(torch.tensor(0.1))

        # Near-zero init — starts as identity
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x) * self.scale


# ============================================
# Semantic-Aware Wavelet Frequency Refiner
# ============================================
class SemanticWaveletRefiner(nn.Module):
    """
    Semantic-Aware Frequency-Guided Residual Correction (SA-FGRC)

    Applies region-specific high-frequency correction:
    - Skin:  stronger suppression (DiffBIR over-generates skin pores)
    - Hair:  medium correction (fix strand artifacts)
    - Eyes:  light correction (preserve identity-critical details)
    - Mouth: light correction (preserve lip texture)
    - Base:  default for remaining regions

    LL band is NEVER touched — identity/structure fully protected.
    """

    def __init__(self, bisenet_path):
        super().__init__()

        self.wavelet     = HaarWavelet2D()
        self.inv_wavelet = InverseHaarWavelet2D()

        # One CNN per region
        self.cnn_skin  = RegionCNN(in_channels=9, hidden=32)  # strongest
        self.cnn_hair  = RegionCNN(in_channels=9, hidden=32)  # medium
        self.cnn_eye   = RegionCNN(in_channels=9, hidden=32)  # light
        self.cnn_mouth = RegionCNN(in_channels=9, hidden=32)  # light
        self.cnn_base  = RegionCNN(in_channels=9, hidden=32)  # default

        # Face parser (frozen)
        self.parser = BiSeNetParser(bisenet_path)
        for p in self.parser.parameters():
            p.requires_grad = False

        # Loss functions
        self.patch_stats = PatchStatisticsLoss()

        from facenet_pytorch import InceptionResnetV1
        self.arcface = InceptionResnetV1(pretrained='vggface2').eval()
        self.arcface.requires_grad_(False)

        self.lpips_fn = lpips.LPIPS(net='vgg').eval()
        self.lpips_fn.requires_grad_(False)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ SemanticWaveletRefiner initialized")
        print(f"   Trainable params: {trainable:,}")
        print(f"   Region CNNs: skin, hair, eye, mouth, base")

    def forward(self, stage2_out):
        """
        stage2_out: [B, 3, H, W] in [0, 1]
        returns:    [B, 3, H, W] — semantically corrected
        """
        B, C, H, W = stage2_out.shape

        # 1. Get face region masks at full resolution
        masks = self.parser(stage2_out)

        # 2. Wavelet decompose
        LL, LH, HL, HH = self.wavelet(stage2_out)
        Hh, Wh = LH.shape[2], LH.shape[3]

        # 3. Downsample masks to HF band resolution
        def ds(m): return F.interpolate(m, size=(Hh, Wh),
                                        mode='bilinear', align_corners=False)
        m_skin  = ds(masks['skin'])   # [B, 1, Hh, Wh]
        m_hair  = ds(masks['hair'])
        m_eye   = ds(masks['eye'])
        m_mouth = ds(masks['mouth'])

        # Background = everything not covered by above regions
        m_base  = (1 - (m_skin + m_hair + m_eye + m_mouth).clamp(0, 1))

        # 4. Concatenate HF bands
        hf = torch.cat([LH, HL, HH], dim=1)  # [B, 9, Hh, Wh]

        # 5. Region-specific residual corrections
        r_skin  = self.cnn_skin(hf)  * m_skin
        r_hair  = self.cnn_hair(hf)  * m_hair
        r_eye   = self.cnn_eye(hf)   * m_eye
        r_mouth = self.cnn_mouth(hf) * m_mouth
        r_base  = self.cnn_base(hf)  * m_base

        # 6. Merge all region corrections
        hf_corrected = hf + r_skin + r_hair + r_eye + r_mouth + r_base

        # 7. Split back to LH, HL, HH
        LH_c, HL_c, HH_c = torch.split(hf_corrected, C, dim=1)

        # 8. Inverse wavelet — LL completely untouched
        corrected = self.inv_wavelet(LL, LH_c, HL_c, HH_c)
        return torch.clamp(corrected, 0.0, 1.0)

    def compute_loss(self, corrected, hq, stage2_out):
        # Wavelet L1 on HF bands
        _, LH_c, HL_c, HH_c   = self.wavelet(corrected)
        _, LH_hq, HL_hq, HH_hq = self.wavelet(hq)
        hf_loss = (F.l1_loss(LH_c, LH_hq) +
                   F.l1_loss(HL_c, HL_hq) +
                   F.l1_loss(HH_c, HH_hq))

        # Patch statistics (FID proxy)
        fid_loss = self.patch_stats(corrected, hq)

        # Identity loss
        pred_r = F.interpolate(corrected * 2 - 1, (160, 160),
                               mode='bilinear', align_corners=False)
        hq_r   = F.interpolate(hq * 2 - 1,        (160, 160),
                               mode='bilinear', align_corners=False)
        pred_emb = self.arcface(pred_r)
        hq_emb   = self.arcface(hq_r).detach()
        id_loss  = 1 - F.cosine_similarity(pred_emb, hq_emb).mean()

        # LPIPS
        lpips_loss = self.lpips_fn(corrected * 2 - 1, hq * 2 - 1).mean()

        loss = hf_loss + 0.1 * fid_loss + 0.1 * id_loss + 0.05 * lpips_loss

        return loss, {
            'hf_loss':    hf_loss.item(),
            'fid_loss':   fid_loss.item(),
            'id_loss':    id_loss.item(),
            'lpips_loss': lpips_loss.item(),
        }


# ============================================
# Patch Statistics Loss
# ============================================
class PatchStatisticsLoss(nn.Module):
    def __init__(self, patch_size=16, stride=8):
        super().__init__()
        self.patch_size = patch_size
        self.stride     = stride

    def forward(self, pred, target):
        def stats(x):
            p = F.unfold(x, self.patch_size, stride=self.stride).permute(0,2,1)
            m = p.mean(1)
            v = ((p - m.unsqueeze(1))**2).mean(1)
            return m, v
        pm, pv = stats(pred)
        tm, tv = stats(target)
        return F.mse_loss(pm, tm) + F.mse_loss(pv, tv)


# ============================================
# Dataset
# ============================================
class Stage2Dataset(Dataset):
    def __init__(self, stage2_dir, hq_dir, num_images=399):
        self.transform = T.ToTensor()
        self.pairs = []
        for i in range(num_images):
            s2 = os.path.join(stage2_dir, f"{i:05d}.png")
            folder = (i // 1000) * 1000
            hq = os.path.join(hq_dir, f"{folder:05d}", f"{i%1000:05d}.png")
            if os.path.exists(s2) and os.path.exists(hq):
                self.pairs.append((s2, hq))
        print(f"📸 Loaded {len(self.pairs)} training pairs")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        s2_path, hq_path = self.pairs[idx]
        s2 = self.transform(Image.open(s2_path).convert('RGB'))
        hq = self.transform(Image.open(hq_path).convert('RGB')
                            .resize((512, 512), Image.BICUBIC))
        return s2, hq


# ============================================
# Trainer
# ============================================
class Trainer:
    def __init__(self, model, device, epochs=30):
        self.model  = model
        self.device = device

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

    def train_epoch(self, loader, epoch):
        self.model.train()
        self.model.parser.eval()
        self.model.arcface.eval()
        self.model.lpips_fn.eval()

        total = 0
        comp_sum = {}
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")

        for s2, hq in pbar:
            s2, hq = s2.to(self.device), hq.to(self.device)
            self.optimizer.zero_grad()

            corrected = self.model(s2)
            loss, comp = self.model.compute_loss(corrected, hq, s2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0)
            self.optimizer.step()

            total += loss.item()
            for k, v in comp.items():
                comp_sum[k] = comp_sum.get(k, 0) + v
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        self.scheduler.step()
        n = len(loader)
        return total / n, {k: v/n for k, v in comp_sum.items()}


# ============================================
# Wavelet Roundtrip Test
# ============================================
def test_roundtrip():
    print("\n🔍 Testing wavelet roundtrip...")
    fwd = HaarWavelet2D()
    inv = InverseHaarWavelet2D()
    x   = torch.rand(1, 3, 512, 512)
    LL, LH, HL, HH = fwd(x)
    recon = inv(LL, LH, HL, HH)
    err = (recon - x).abs().max().item()
    print(f"   Roundtrip error: {err:.8f}")
    ok = err < 1e-4
    print("   ✅ PASSED" if ok else "   ❌ FAILED")
    return ok


# ============================================
# Main
# ============================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'inference', 'test'], default='train')
    parser.add_argument('--bisenet_path',
                        default='/NTIRE2026/runs/C15_RWFaceRestoration/AllforFace/face-parsing.PyTorch/res/cp/79999_iter.pth')
    parser.add_argument('--stage2_dir',  default='./stage2_outputs')
    parser.add_argument('--hq_dir',
                        default='/NTIRE2026/C15_RWFaceRestoration/ffhq-dataset/images1024x1024')
    parser.add_argument('--num_images',  type=int,   default=399)
    parser.add_argument('--batch_size',  type=int,   default=4)
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--save_dir',    default='./checkpoints_semantic')
    parser.add_argument('--checkpoint',  default='./checkpoints_semantic/semantic_best.pth')
    parser.add_argument('--input_dir',   default=None)
    parser.add_argument('--output_dir',  default='./results_semantic')
    args   = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.mode == 'test':
        test_roundtrip()
        return

    if args.mode == 'train':
        print("\n🎯 Training Semantic-Aware Wavelet Refiner...")
        os.makedirs(args.save_dir, exist_ok=True)

        if not test_roundtrip():
            print("❌ Wavelet test failed"); return

        dataset = Stage2Dataset(args.stage2_dir, args.hq_dir, args.num_images)
        loader  = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=2, pin_memory=True)

        model   = SemanticWaveletRefiner(args.bisenet_path).to(device)
        trainer = Trainer(model, device, args.epochs)

        best_loss = float('inf')
        for epoch in range(args.epochs):
            loss, comp = trainer.train_epoch(loader, epoch)
            print(f"\n📊 Epoch {epoch+1}/{args.epochs}  loss={loss:.4f}")
            print(f"   hf={comp['hf_loss']:.4f}  fid={comp['fid_loss']:.4f}"
                  f"  id={comp['id_loss']:.4f}  lpips={comp['lpips_loss']:.4f}")
            print(f"   LR={trainer.scheduler.get_last_lr()[0]:.2e}")

            # Print learned scale factors (shows what each region learned)
            print(f"   Scales → skin:{model.cnn_skin.scale.item():.3f}"
                  f"  hair:{model.cnn_hair.scale.item():.3f}"
                  f"  eye:{model.cnn_eye.scale.item():.3f}"
                  f"  mouth:{model.cnn_mouth.scale.item():.3f}"
                  f"  base:{model.cnn_base.scale.item():.3f}")

            ckpt = os.path.join(args.save_dir, f"semantic_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)

            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), args.checkpoint)
                print("   🏆 New best!")

        print(f"\n✅ Done. Best loss={best_loss:.4f}")
        print(f"   Saved to {args.checkpoint}")
        return

    # ---- INFERENCE ----
    if not args.input_dir:
        print("❌ Provide --input_dir"); return

    print(f"\n🎯 Inference: {args.input_dir}")
    model = SemanticWaveletRefiner(args.bisenet_path).to(device)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"✅ Loaded {args.checkpoint}")
    else:
        print(f"⚠️  No checkpoint at {args.checkpoint}")

    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    transform = T.ToTensor()

    images = sorted(f for f in os.listdir(args.input_dir)
                    if f.lower().endswith('.png'))
    for name in tqdm(images, desc="Processing"):
        img = Image.open(os.path.join(args.input_dir, name)).convert('RGB')
        t   = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t)
        T.ToPILImage()(out.squeeze(0).cpu()).save(
            os.path.join(args.output_dir, name))

    print(f"✅ Saved to {args.output_dir}")


if __name__ == '__main__':
    main()