import argparse
from collections import defaultdict
import os
import os.path as osp
from pathlib import Path
import random
import sys
from typing import Any, Dict

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger
import monai
from monai.metrics import CumulativeAverage
import numpy as np
import torch
from torch import Tensor

from config import get_config, parse_cfg, update_config
from data import DATALOADERS, DATASETS, TRANSFORMS, BaseDataset
from model import build_model
from utils import pretty_object_str
from utils.eval import eval_single_volume
from utils.loss import LOSSES
from utils.optimization import LR_SCHEDULERS, OPTIMIZERS
from utils.visualization import plot_image_mask_groups


def is_rank_zero() -> bool:
    """
    判断当前进程是否为主进程。
    单卡训练时 LOCAL_RANK 通常不存在，默认视为 rank 0。
    多卡 DDP 时，只有 LOCAL_RANK=0 的进程负责打印主要日志和保存 args/config 文本。
    """
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def setup_loguru(log_dir: str) -> None:
    """
    配置 loguru 日志：
    1. 移除默认 handler，避免多卡时重复输出。
    2. 只让 rank 0 打印控制台日志和写 training.log。
    3. 屏蔽 vmamba 逐层加载预训练权重的刷屏信息。
    """
    logger.remove()

    if is_rank_zero():
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        logger.add(sys.stderr, level="INFO", format=log_format)
        logger.add(
            osp.join(log_dir, "training.log"),
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            encoding="utf-8",
        )


class Trainer(L.LightningModule):
    def __init__(self, log_dir: str, cfg: dict) -> None:
        super(Trainer, self).__init__()
        self.log_dir = log_dir
        self.cfg = cfg

        assert args.dataset in DATASETS, f"Dataset {args.dataset} not supported"
        self.dataset_cfg = DATASETS[args.dataset]
        self.train_dataset = None
        self.val_dataset = None

        model_name, model_cfg = parse_cfg(self.cfg, "model")
        self._model = build_model(
            name=model_name,
            in_channels=self.cfg["in_channels"],
            num_classes=self.dataset_cfg["num_classes"],
            **model_cfg,
        )

        loss_name, loss_cfg = parse_cfg(self.cfg, "loss")
        assert loss_name in LOSSES, f"Loss {loss_name} not supported"
        self.criterion = LOSSES[loss_name](**loss_cfg)

        self.tl_metric = CumulativeAverage()
        self.vs_metric = defaultdict(lambda: defaultdict(list))

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    def prepare_data(self) -> None:
        root = osp.expandvars(osp.join("$DATASET_HOME", self.dataset_cfg["root_suffix"]))

        if "$DATASET_HOME" in root or not osp.exists(root):
            raise FileNotFoundError(
                f"数据根目录不存在: {root}\n"
                f"请先执行:\n"
                f"export DATASET_HOME=/data/pytorch/wan/datasets"
            )

        tt_name, tt_cfg = parse_cfg(self.cfg, "train_transform")
        train_transform = TRANSFORMS[tt_name](**tt_cfg) if tt_name else None

        train_split = f"fold_{args.fold}_train" if args.fold is not None else "train"

        self.train_dataset = BaseDataset(
            base_dir=root,
            split=train_split,
            list_dir=self.dataset_cfg["list_dir"],
            transform=train_transform,
        )

        vt_name, vt_cfg = parse_cfg(self.cfg, "test_transform")
        test_transform = TRANSFORMS[vt_name](**vt_cfg) if vt_name else None

        val_split = f"fold_{args.fold}_valid" if args.fold is not None else "valid"

        self.val_dataset = BaseDataset(
            base_dir=root,
            split=val_split,
            list_dir=self.dataset_cfg["list_dir"],
            transform=test_transform,
        )

        logger.info(f"数据根目录 root: {root}")
        logger.info(f"列表目录 list_dir: {self.dataset_cfg['list_dir']}")
        logger.info(f"训练 split: {train_split}, 样本数: {len(self.train_dataset)}")
        logger.info(f"验证 split: {val_split}, 样本数: {len(self.val_dataset)}")

    def train_dataloader(self) -> Any:
        def worker_init_fn(worker_id: int) -> None:
            random.seed(cfg["seed"] + worker_id)

        loader_name, loader_cfg = parse_cfg(self.cfg, "train_dataloader")
        if "worker_init_fn" not in loader_cfg:
            loader_cfg["worker_init_fn"] = worker_init_fn
        assert loader_name is not None, "train dataloader is not configured"
        return DATALOADERS[loader_name](self.train_dataset, **loader_cfg)

    def val_dataloader(self) -> Any:
        loader_name, loader_cfg = parse_cfg(self.cfg, "val_dataloader")
        assert loader_name is not None, "valid dataloader is not configured"
        return DATALOADERS[loader_name](self.val_dataset, **loader_cfg)

    def configure_optimizers(self) -> dict:
        optim_name, optim_cfg = parse_cfg(self.cfg, "optimizer")
        optimizer = OPTIMIZERS[optim_name](self._model.parameters(), **optim_cfg)

        lrsch_name, lrsch_cfg = parse_cfg(self.cfg, "lr_scheduler")
        if lrsch_name is not None:
            interval = lrsch_cfg.pop("lightning_interval", "epoch")
            lrsch_cfg = update_config(lrsch_cfg, trainer=self)
            scheduler = LR_SCHEDULERS[lrsch_name](optimizer, **lrsch_cfg)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": interval},
            }
        return {"optimizer": optimizer}

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == 0:
            self.log_and_logger("mean_train_loss", 0.0)

        freeze_encoder_epochs = self.cfg.get("freeze_encoder_epochs", 0)
        if freeze_encoder_epochs > 0:
            assert hasattr(
                self._model, "freeze_encoder"
            ), f"Model {self.cfg['model'][0]} does not support freezing encoder"
            if self.current_epoch < freeze_encoder_epochs:
                self._model.freeze_encoder()
            else:
                self._model.unfreeze_encoder()
        super().on_train_epoch_start()

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        image = batch["image"].to(self.device)
        label = batch["label"].to(self.device)

        logits = self.forward(image)
        loss = self.criterion(logits, label)

        self.log("loss", loss.item(), prog_bar=True)
        self.tl_metric.append(loss.item())
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        tl = self.tl_metric.aggregate().item()
        self.log_and_logger("mean_train_loss", tl)
        self.tl_metric.reset()

    def validation_step(self, batch: Dict[str, Tensor], *args: Any) -> None:
        volume, label = batch["image"], batch["label"]
        metric = eval_single_volume(
            model=self._model,
            volume=volume,
            label=label,
            num_classes=self.dataset_cfg["num_classes"],
            output=osp.join(self.log_dir, str(self.current_epoch)),
            patch_size=self.cfg["img_size"],
            device=self.device,
            norm_x_transform=getattr(self.train_dataset.transform, "norm_x_transform", None),
        )

        for metric_name, class_metric in metric.items():
            for class_name, value in class_metric.items():
                self.vs_metric[metric_name][class_name].append(np.mean(value))

    def on_validation_epoch_end(self) -> None:
        for metric_name, class_metric in self.vs_metric.items():
            avg_metric = []
            for class_name, value in class_metric.items():
                t = np.mean(value)
                self.log(f"val_{metric_name}_{class_name}", t)
                avg_metric.append(t)
            self.log_and_logger(f"val_mean_{metric_name}", np.mean(avg_metric))
        self.vs_metric = defaultdict(lambda: defaultdict(list))

    def log_and_logger(self, name: str, value: Any, **kwargs: Any) -> None:
        self.log(name, value, **kwargs)
        logger.info(f"{name}: {value}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset name")
    parser.add_argument("-c", "--config", type=str, required=True, help="config name")
    parser.add_argument("-r", "--round", type=int, default=0, help="round to run the experiment")
    parser.add_argument("--seed", type=int, default=42, help="seed to experiment")

    parser.add_argument("-t", "--tag", type=str, default="", help="自定义实验标签 (如: lr5e4, no_aug)")

    parser.add_argument("--fold", type=int, default=None, help="5-fold cross validation fold index, 0-4")

    parser.add_argument("--devices", type=int, default=1, help="number of visible GPUs to use")

    parser.add_argument("--progress_bar", action="store_true", help="enable Lightning progress bar")
    parser.add_argument("--log_root",type=str,default="./log_train",help="root directory for saving training logs and checkpoints")

    args = parser.parse_args()

    if args.fold is not None:
        assert 0 <= args.fold <= 4, "--fold must be in [0, 1, 2, 3, 4]"

    if device == "cuda":
        visible_gpu_count = torch.cuda.device_count()
        assert args.devices >= 1, "--devices must be >= 1"
        assert args.devices <= visible_gpu_count, (
            f"--devices={args.devices} 超过当前可见 GPU 数量 {visible_gpu_count}。\n"
            f"请检查 CUDA_VISIBLE_DEVICES 设置。"
        )
    else:
        args.devices = 1

    cfg = get_config(args.config)

    cfg["seed"] = args.seed + int(args.round)

    if args.fold is not None:
        run_name = f"fold_{args.fold}"
    else:
        run_name = f"r{args.round}"

    if args.tag:
        run_name += f"-{args.tag}"

    log_dir = osp.join(args.log_root, args.dataset, args.config, run_name)
    os.makedirs(log_dir, exist_ok=True)

    setup_loguru(log_dir)

    args_txt_path = osp.join(log_dir, "args.txt")
    config_txt_path = osp.join(log_dir, "config.txt")

    if is_rank_zero():
        with open(args_txt_path, "w", encoding="utf-8") as f:
            for k, v in vars(args).items():
                f.write(f"{k}: {v}\n")
            f.write(f"effective_seed: {cfg['seed']}\n")
            f.write(f"log_dir: {log_dir}\n")
            f.write(f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}\n")

        with open(config_txt_path, "w", encoding="utf-8") as f:
            f.write(pretty_object_str(cfg))

    logger.info("=" * 60)
    logger.info(f"🚀 正在训练数据集 : {args.dataset.upper()}")
    logger.info(f"🏷️  实验标签 (Tag) : {args.tag if args.tag else '无'}")
    logger.info(f"🔄 实验轮次 (Round): {args.round}")
    logger.info(f"🧩 五折编号 (Fold): {args.fold if args.fold is not None else '未启用'}")
    logger.info(f"🌱 随机种子 (Seed): {cfg['seed']}")
    logger.info(f"🖥️  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
    logger.info(f"🖥️  Lightning devices: {args.devices}")
    logger.info(f"📂 日志保存目录: {log_dir}")
    logger.info("=" * 60)
    logger.info(f"Config: {pretty_object_str(cfg)}")

    L.seed_everything(cfg["seed"])
    monai.utils.set_determinism(cfg["seed"])

    callbacks = [
        ModelCheckpoint(
            dirpath=osp.join(log_dir, "checkpoints"),
            monitor="val_mean_dice",
            mode="max",
            filename="best",
            save_top_k=1,
            save_last=True,
        )
    ]
    model = Trainer(log_dir, cfg)

    if is_rank_zero():
        total_params = sum(p.numel() for p in model._model.parameters())
        trainable_params = sum(p.numel() for p in model._model.parameters() if p.requires_grad)

        logger.info("=" * 60)
        logger.info(f"📊 模型规模统计:")
        logger.info(f" - 总参数量: {total_params / 1e6:.2f} M")
        logger.info(f" - 可训练参数量: {trainable_params / 1e6:.2f} M")

        img_size = cfg.get("img_size", [224, 224])
        if isinstance(img_size, int):
            img_size = [img_size, img_size]

        flops_input = torch.randn(1, cfg.get("in_channels", 3), *img_size).to(device)
        model._model = model._model.to(device)
        try:
            from fvcore.nn import FlopCountAnalysis

            was_training = model._model.training
            model._model.eval()

            with torch.no_grad():
                flops = FlopCountAnalysis(model._model, flops_input)
                flops.unsupported_ops_warnings(False)
                logger.info(f" - 计算量 (FLOPs): {flops.total() / 1e9:.2f} G")

            if was_training:
                model._model.train()

        except ImportError:
            logger.warning("未安装 fvcore，跳过 FLOPs 统计。")

        except Exception as e:
            logger.warning(f" - 计算量 (FLOPs): 统计失败，已跳过。原因: {repr(e)}")
            if "was_training" in locals() and was_training:
                model._model.train()

        logger.info("=" * 60)
    trainer = L.Trainer(
        precision="16-mixed",
        accelerator=device,
        devices=args.devices,
        strategy="ddp" if (device == "cuda" and args.devices > 1) else "auto",
        max_epochs=cfg["max_epochs"],
        check_val_every_n_epoch=cfg.get("check_val_every_n_epoch", 1),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        default_root_dir=log_dir,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=args.progress_bar,
        logger=False,
    )
    trainer.fit(model, ckpt_path=None)
