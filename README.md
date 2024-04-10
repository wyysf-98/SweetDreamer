# SweetDreamer: Aligning Geometric Priors in 2D Diffusion for Consistent Text-to-3D (ICLR 2024)

#####  <p align="center"> [Weiyu Li](https://wyysf-98.github.io/), [Rui Chen](https://aruichen.github.io/), [Xuelin Chen](https://xuelin-chen.github.io/), [Ping Tan](https://ece.hkust.edu.hk/pingtan)</p>

<p align="center">
  <img src="https://sweetdreamer3d.github.io/assets/images/overview_pipelines.png"/>
</p>

#### <p align="center">[Project Page](https://sweetdreamer3d.github.io/) | [ArXiv](https://arxiv.org/abs/2310.02596) | [Paper]() | [Video]()</p>
<p align="center"> All Code and Ckpt will be released in the next few days, sorry for the delay due to some to some permission issues :( 🏗️ 🚧 🔨</p>

- [x] Release the reorganized code
- [ ] Release the pretrained model (tiny-version)
- [ ] Release the full model

## Prerequisite

### Setup environment (Install threestudio)

**This part is the same as original threestudio. Skip it if you already have installed the environment.**

See [installation.md](https://github.com/threestudio-project/threestudio/blob/main/docs/installation.md) for additional information, including installation via Docker.

- You must have an NVIDIA graphics card with at least 20GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
- Install `Python >= 3.8`.
- (Optional, Recommended) Create a virtual environment:

```sh
python3 -m virtualenv venv
. venv/bin/activate

# Newer pip versions, e.g. pip-23.x, can be much faster than old versions, e.g. pip-20.x.
# For instance, it caches the wheels of git packages to avoid unnecessarily rebuilding them later.
python3 -m pip install --upgrade pip
```

- Install `PyTorch >= 1.12`. We have tested on `torch1.12.1+cu113` and `torch2.0.0+cu118`, but other versions should also work fine.

```sh
# torch1.12.1+cu113
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# or torch2.0.0+cu118
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

- (Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions:

```sh
pip install ninja
```

- Install dependencies:

```sh
pip install -r requirements.txt
```


### Download the pretrained CCM model(TBD)

```sh
sh download.sh
```



## Quick demo

```sh
python launch.py --config configs/sweetdreamer-stage1.yaml --train --gpu 0 \
                 system.prompt_processor.prompt="Albert Einstein with grey suit is riding a bicycle" \
                 system.cmm_prompt_processor.prompt="Albert Einstein with grey suit is riding a bicycle" \
                 tag=einstein

python launch.py --config configs/sweetdreamer-stage2.yaml --train --gpu 0 \
                 system.prompt_processor.prompt="Albert Einstein with grey suit is riding a bicycle" \
                 system.cmm_prompt_processor.prompt="Albert Einstein with grey suit is riding a bicycle" \
                 tag=einstein
```


## Acknowledgement

This code is built on the amazing open-source projects:
 - [threestudio-project](https://github.com/threestudio-project/threestudio?tab=readme-ov-file)
 - [diffusers](https://github.com/huggingface/diffusers)
 - [stable-diffusion](https://stability.ai/news/stable-diffusion-public-release)
 - [deep-floyed](https://github.com/deep-floyd/IF?tab=readme-ov-file)

We also thank Jianxiong Pan and Feipeng Tian for the help of the data and GPU server.

## Citation

If you find our work useful for your research, please consider citing using the following BibTeX entry.

```BibTeX
@article{sweetdreamer,
  author    = {Weiyu Li and Rui Chen and Xuelin Chen and Ping Tan},
  title     = {SweetDreamer: Aligning Geometric Priors in 2D Diffusion for Consistent Text-to-3D},
  journal   = {arxiv:2310.02596},
  year      = {2023},
}
```