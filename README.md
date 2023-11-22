# Vision Transformers for High-Resolution Canopy Height Mapping

tbd

## Available Models

| Model                                     | Architecture                 | Params     |
| ----------------------------------------- | ---------------------------- | ---------- |
| [vit-tiny](models/vit/__init__.py)        | Vision Transformer + ConvNet | 6.576.777  |
| [vit-small](models/vit/__init__.py)       | Vision Transformer + ConvNet | 14.518.713 |
| [vit-base](models/vit/__init__.py)        | Vision Transformer + ConvNet | 25.563.369 |
| [unet](models/unet/__init__.py)           | U-Net                        | 10.103.933 |
| [unet++](models/unetplusplus/__init__.py) | U-Net++                      | 10.479.187 |

## Training Data
- 2693 multispectral PlanetScope satellite images (4096x4096x4, ~650 GB)
- 4 channels per image (R, G, B, NIR)
- 505.724 patches (256x256) after preprocessing
- 16.933.743 GEDI labels after preprocessing

## Install
To install the required packages to a virtual environment and activate it, run:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env
```
For merging the output files into one, you need to install [GDAL](https://gdal.org/index.html):
```bash
brew install gdal
```

## Usage

```bash
python [preprocessing|train|evaluate|run].py
```

## Inference

```bash
python run.py

cd output

ls *.tif > merge_list.txt

gdal_merge.py -o output.tif --optfile merge_list.txt -co COMPRESS=LZW -co BIGTIFF=YES
```

## References
1. [Vision Transformers for Dense Prediction](http://arxiv.org/abs/2103.13413), [DPT](https://github.com/isl-org/DPT)

2. [The overlooked contribution of trees outside forests to tree cover and woody biomass across Europe](https://www.science.org/doi/full/10.1126/sciadv.adh4097), 
[planet_canopy_height_v0](https://zenodo.org/records/8156190)