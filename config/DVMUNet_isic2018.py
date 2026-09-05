

max_epochs = 100
img_size = 224

CONFIG = {
    "model": ("DVMUNet", {}),
    "max_epochs": max_epochs,
    "check_val_every_n_epoch": 1,
    "deep_supervision": False,
    "in_channels": 3,
    "freeze_encoder_epochs": 0,
    "img_size": (img_size, img_size),
    "train_transform": (
        "ours",
        {
            "output_size": (img_size, img_size),
            "num_classes": 2,
        },
    ),
    "test_transform": ("noops", {}),
    "train_dataloader": (
        "default",
        {
            "batch_size": 32,
            "num_workers": 16,
            "shuffle": True,
            "pin_memory": True,
            "persistent_workers": True,
        },
    ),
    "val_dataloader": (
        "default",
        {
            "batch_size": 1,
            "shuffle": False,
            "pin_memory": True,
            "num_workers": 4,
            "persistent_workers": True,
        },
    ),
    "loss": (
        "DiceCELoss",
        {
            "ce_weight": 0.5,
            "dc_weight": 0.5,
        },
    ),
    "optimizer": (
        "AdamW",
        {
            "lr": 3e-4,
            "weight_decay": 1e-2,
            "eps": 1e-8,
            "amsgrad": False,
            "betas": (0.9, 0.999),
        },
    ),
    "lr_scheduler": ("CosineAnnealingLR", {"T_max": max_epochs, "eta_min": 1e-6}),
}
