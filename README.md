# DVMUNet

PyTorch implementation of **DVMUNet** for skin lesion segmentation.

> The complete DVMUNet model implementation will be released upon acceptance of the paper.

The repository currently provides data processing, dataset splits, experiment configuration, and training and evaluation utilities. Model source code and pretrained weights are not included in the current release.

## Data

Set `DATASET_HOME` to the dataset root. Each `.npz` sample should contain an image of shape `[3, H, W]` and a label of shape `[H, W]`.

Dataset lists and five-fold splits for ISIC2016, ISIC2017, and ISIC2018 are provided in `lists/`. PH2 is included as an external test set.

## Acknowledgements

This project builds on [VMamba](https://github.com/MzeroMiko/VMamba) and [Mamba](https://github.com/state-spaces/mamba).
