# WG-DSANet

Official PyTorch implementation of the paper *"Detail-Aware Change Detection with Coordinated Frequency and Spatial Cues"* (Applied Intelligence, under review).

WG-DSANet is a lightweight framework for change detection in high-resolution bi-temporal remote sensing imagery. It integrates wavelet-guided feature fusion, dual-sieve attention, and edge-aware enhancement to preserve fine textures, boundaries, and small targets.

## Requirements

- Python ≥ 3.10, PyTorch ≥ 2.7.0, CUDA ≥ 12.6

```bash
pip install torch torchvision pytorch-wavelets einops thop
```

## Usage

```python
from WG_DSANet import CDNet

model = CDNet(backbone='mobilenet', output_stride=16, num_classes=2)

# input1, input2: [B, 3, 256, 256]
output = model(input1, input2) # [B, 2, 256, 256]
```

The model uses ~8M parameters and ~10.25 GFLOPs. Training configuration and dataset preprocessing details are provided in the paper (Section 4).

## Pretrained Weights

`mobilenet_v2-6a65762b.pth` contains pretrained MobileNetV2 ImageNet weights, loaded automatically. Update the path in `mobilenet.py` if your directory structure differs.

## Citation

```bibtex
@article{yuan2025detail,
title = {Detail-Aware Change Detection with Coordinated Frequency and Spatial Cues},
author = {Yuan, Jianjun and Zhou, Jianjun and Chen, Siyu and Zhao, Luoming and Wang, Zhongshu},
journal = {Applied Intelligence},
year = {2025},
note = {Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
