# Runtime -> Change runtime type -> GPU (T4/A100 preferred)
import torch, os, glob, numpy as np, matplotlib.pyplot as plt
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config (feel free to tweak)
IMG_SIZE   = 128          # start with 128 for speed; later try 256
BATCH_SIZE = 8
N_EPOCHS   = 20           # you can stop earlier if samples look good
LR_G       = 1e-4
LR_D       = 1e-4
N_CRITIC   = 5            # WGAN-GP: #critic steps per generator step
LAMBDA_GP  = 10.0
Z_DIM      = 128          # latent dim
SAVE_EVERY = 500          # steps between image dumps
OUT_DIR    = "/content/gan_oasis_out"
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(42)


from PIL import Image
from torch.utils.data import Dataset, DataLoader

class OasisSliceDataset(Dataset):
    def __init__(self, img_dirs, img_size=IMG_SIZE):
        self.paths = []
        for d in img_dirs:
            self.paths += glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.nii.png"))
        self.paths = sorted(self.paths)
        assert len(self.paths) > 0, "No images found."
        self.size = img_size

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        im = np.array(Image.open(p))    # HxW (grayscale)
        if im.ndim == 3:                # if any RGB sneaks in, convert to gray
            im = np.mean(im, axis=2)
        # resize to square
        if im.shape[0] != self.size:
            im = np.array(Image.fromarray(im).resize((self.size, self.size), Image.BILINEAR))
        im = im.astype(np.float32)
        if im.max() > 1: im /= 255.0
        im = im*2.0 - 1.0               # [0,1] -> [-1,1] for Tanh
        im = torch.from_numpy(im).unsqueeze(0)  # (1,H,W)
        return im

dataset = OasisSliceDataset(IMG_DIRS, IMG_SIZE)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
len(dataset), next(iter(loader)).shape




import torch.nn as nn
import torch.nn.utils.spectral_norm as SN

def gen_block(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(True),
    )

def disc_block(in_ch, out_ch):
    return nn.Sequential(
        SN(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)),
        nn.LeakyReLU(0.2, inplace=True),
    )

class Generator(nn.Module):
    # input z: (N, Z_DIM)
    def __init__(self, z_dim=Z_DIM, img_ch=1, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, base*8, 4, 1, 0, bias=False),  # 1x1 -> 4x4
            nn.BatchNorm2d(base*8), nn.ReLU(True),
            gen_block(base*8, base*4),  # 8x8
            gen_block(base*4, base*2),  # 16x16
            gen_block(base*2, base),    # 32x32
            gen_block(base, base//2),   # 64x64
            nn.ConvTranspose2d(base//2, img_ch, 4, 2, 1, bias=False),# 128x128
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))

class Critic(nn.Module):  # WGAN critic (no sigmoid)
    def __init__(self, img_ch=1, base=64):
        super().__init__()
        self.net = nn.Sequential(
            disc_block(img_ch, base),       # 64x64
            disc_block(base, base*2),       # 32x32
            disc_block(base*2, base*4),     # 16x16
            disc_block(base*4, base*8),     # 8x8
            nn.Conv2d(base*8, 1, 4, 1, 0, bias=False)  # 5x5 -> 1x1 (with 128x128 in)
        )
    def forward(self, x):
        out = self.net(x)
        return out.view(-1)

G, D = Generator().to(device), Critic().to(device)
sum(p.numel() for p in G.parameters()), sum(p.numel() for p in D.parameters())


import torch.optim as optim

opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=(0.0, 0.9))
opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(0.0, 0.9))

def gradient_penalty(D, real, fake):
    bs = real.size(0)
    eps = torch.rand(bs, 1, 1, 1, device=real.device)
    inter = eps*real + (1-eps)*fake
    inter.requires_grad_(True)
    d_inter = D(inter)
    grad = torch.autograd.grad(outputs=d_inter, inputs=inter,
                               grad_outputs=torch.ones_like(d_inter),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad = grad.view(bs, -1)
    gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp

# fixed noise to monitor progress (same grid each time)
FIXED = torch.randn(64, Z_DIM, device=device)


from torchvision.utils import make_grid, save_image
from math import inf

step = 0
G.train(); D.train()
g_losses, d_losses = [], []

for epoch in range(1, N_EPOCHS+1):
    for real in loader:
        real = real.to(device)
        bs = real.size(0)

        # --- Train Critic N_CRITIC times ---
        for _ in range(N_CRITIC):
            z = torch.randn(bs, Z_DIM, device=device)
            fake = G(z).detach()
            d_real = D(real)
            d_fake = D(fake)
            gp = gradient_penalty(D, real, fake)
            loss_D = -(d_real.mean() - d_fake.mean()) + LAMBDA_GP * gp

            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

        # --- Train Generator once ---
        z = torch.randn(bs, Z_DIM, device=device)
        gen = G(z)
        d_gen = D(gen)
        loss_G = -d_gen.mean()

        opt_G.zero_grad(set_to_none=True)
        loss_G.backward()
        opt_G.step()

        # logging
        g_losses.append(loss_G.item())
        d_losses.append(loss_D.item())
        step += 1

        # save grid
        if step % SAVE_EVERY == 0:
            with torch.no_grad():
                G.eval()
                samples = G(FIXED).clamp(-1,1)
                grid = make_grid(samples, nrow=8, normalize=True, value_range=(-1,1))
                save_path = os.path.join(OUT_DIR, f"samples_step_{step:07d}.png")
                save_image(grid, save_path)
                G.train()
            print(f"[ep {epoch:03d} | step {step}]  D: {loss_D.item():.3f}  G: {loss_G.item():.3f}  -> {save_path}")

    # save a checkpoint each epoch
    torch.save({
        "G": G.state_dict(),
        "D": D.state_dict(),
        "step": step,
        "config": dict(IMG_SIZE=IMG_SIZE, Z_DIM=Z_DIM)
    }, os.path.join(OUT_DIR, f"ckpt_ep_{epoch:03d}.pt"))

print("Training done. Samples/checkpoints in:", OUT_DIR)


