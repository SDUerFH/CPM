Convolutional_Pose_Machines 加注释版本


## Download datas
I found that original link of the Leeds Sports Pose Dataset at University of Leeds has been removed. You can download the dataset [here](http://sam.johnson.io/research/lsp.html) and the extended dataset [here](http://sam.johnson.io/research/lspet.html).

Please download the dataset and unzip it in `data` folder with a directory tree like this:

```bash
data
└── LSP
    ├── lsp_dataset
    │   ├── images
    │   └── visualized
    └── lspet_dataset
        └── images
```

## Usage
### Training
#### With weighted loss
```bash
python -W ignore::UserWarning cpm_train.py --lsp-root ./data/LSP --ckpt-dir ./model  --summary-dir ./summary --cuda
```
#### Without weighted loss
```bash
python -W ignore::UserWarning cpm_train.py --lsp-root ./data/LSP --ckpt-dir ./model  --summary-dir ./summary --cuda --wl
```
More argument for training please refer to `cpm_train.py`.

## References

[1] [Wei, Shih-En, et al. "Convolutional pose machines." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.](https://arxiv.org/abs/1602.00134)
