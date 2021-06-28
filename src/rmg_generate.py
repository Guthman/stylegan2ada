# Generate images using pretrained network pickle.

import os
import dnnlib
import numpy as np
import PIL.Image
from pathlib import Path
import torch
from tqdm.auto import tqdm
import legacy
import functools

no_images = 2
pkl_list = [
    r'F:\temp\thesisdata\stylegan2ada_OUTPUT\000-saatchi_portrait_cond128-128-cond-auto4-gamma5-batch48-gf_bnc\saatchi_portrait_cond128-128-2177.pkl',
    r'F:\temp\thesisdata\stylegan2ada_OUTPUT\000-saatchi_portrait_cond128-128-cond-auto4-gamma10-batch48-gf_bnc\saatchi_portrait_cond128-128-3145.pkl']
class_idx_list = [0, 4, 1, 2, 3]
seed_list = list(range(0, no_images))
psi_list = [0.9]
noise_mode = 'const'
output_directory = r'F:\temp\thesisdata\stylegan2ada_OUTPUT\new_gen_script_test'

# Enable image generation using cpu
# device = torch.device('cuda')
device = torch.device('cpu')


def generate_images(
    G_,
    seeds: list,
    truncation_psi: float,
    noise_mode_: str,
    outdir: str,
    class_idx_: int,
    projected_w: str
):
    G = G_

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device)
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(tqdm(ws)):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode_)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        print('No seed list passed!')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx_ is None:
            print('No class label/idx passed!')
        label[:, class_idx_] = 1
    else:
        if class_idx_ is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(tqdm(seeds)):
        # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode_)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')


for pkl in pkl_list:
    with dnnlib.util.open_url(pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    G.forward = functools.partial(G.forward, force_fp32=True)

    folder_name = pkl.split('\\')[-1].split('.')[0]
    output_path = output_directory + '/' + str(folder_name) + '_psi' + str(psi_list[0])
    Path(output_path).mkdir(parents=True, exist_ok=True)
    for class_idx in class_idx_list:
        class_output_path = output_path + '/' + str(class_idx)
        Path(class_output_path).mkdir(parents=True, exist_ok=True)
        generate_images(G_=G,
                        seeds=seed_list,
                        truncation_psi=psi_list[0],
                        noise_mode_=noise_mode,
                        outdir=class_output_path,
                        class_idx_=class_idx,
                        projected_w=None)
