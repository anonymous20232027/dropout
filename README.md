# Dropout Reduces Underfitting

Official PyTorch implementation of **Dropout Reduces Underfitting**.

--- 

<p align="center">
<img src="https://tva1.sinaimg.cn/large/008vxvgGgy1h894eu0vbdj30uv0romz3.jpg" width=50% height=50% 
class="center">
</p>

We investigate **early dropout**, where dropout is used only during the initial phase of training, and switched off afterwards, and **late dropout**, where dropout is used only during the later phase of training.


<!-- ✅ ⬜️  -->

## Results

### Basic recipe results
| model | baseline | standard dropout | standard s.d. | early dropout  | early s.d.|
|:---:|:---:|:---:|:---:| :---:|:---:|
| ViT-T | 73.9 | 67.9 | 72.6 | **74.3** | **74.4** |
| Mixer-S/32 | 71.0 | 67.1 | 70.5 | **71.3** | **71.7** |
| Swin-F | 74.3 | 71.6 | 73.7 | **74.7** | **75.2** |
| ConvNeXt-F | 76.1 | - | 75.5 | - | **76.3** |

### Improved recipe results

| model | baseline | standard dropout | standard s.d. | early dropout  | early s.d.|
|:---:|:---:|:---:|:---:| :---:|:---:|
| ViT-T | 76.3 | 71.5 | 75.6 | **76.7** | **76.7** |
| Swin-F | 76.1 | 73.5 | 75.6 | **76.6** | **76.6** |
| ConvNeXt-F | 77.5 | - | 77.4 | - | **77.7** |

## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Training

### Basic Recipe
We give an example traing command for early dropout/early stochastic depth on `ViT-T`.
- For training on other models, change `--model` accordingly. The main results contain the model `vit_tiny`, `mixer_s32`, `convnext_femto`, `swin_femto`.

**Early Dropout**
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_tiny --warmup_epochs 50 --epochs 300 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--dropout 0 --early_dropout_epochs 50 --early_dropout_value 0.5 --early_dropout_mode regular \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```

**Early Stochastic Depth**
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_tiny  --warmup_epochs 50 --epochs 300 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--drop_path 0 --early_drop_path_epochs 50 --early_drop_path_value 0.5 --early_drop_path_mode regular \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```

**Late Stochastic Depth**
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_base  --warmup_epochs 50 --epochs 300 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--drop_path 0.4 --early_drop_path_epochs 50 --early_drop_path_value 0 --early_drop_path_mode regular \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```



### Improved Recipe

**Early Dropout**
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_tiny --warmup_epochs 50 --epochs 600 --mixup 0.3 --cutmix 0.3 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--dropout 0 --early_dropout_epochs 50 --early_dropout_value 0.5 --early_dropout_mode regular \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```

**Early Stochastic Depth**
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_tiny  --warmup_epochs 50 --epochs 600 --mixup 0.3 --cutmix 0.3 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--drop_path 0 --early_drop_path_epochs 50 --early_drop_path_value 0.5 --early_drop_path_mode regular \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```

- For training on other models, change `--model` accordingly. The main results contain the model `vit_tiny`, `mixer_s32`, `convnext_femto`, `swin_femto`.


## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
