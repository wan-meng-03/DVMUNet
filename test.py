import argparse
from collections import OrderedDict
from glob import glob
import json
import os
import os.path as osp
import re
from typing import Any, Dict, List, Optional

from loguru import logger
from medpy import metric
import numpy as np

np.bool = np.bool_

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

from sklearn.metrics import matthews_corrcoef

import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm

from config import parse_cfg
from data import CLS2COLOR_MAPPING, DATALOADERS, DATASETS, TRANSFORMS, BaseDataset
from model import build_model


METRICS = ("dc", "hd95", "jc", "asd", "se", "sp", "pr", "acc")


def try_gpu() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pretty_number(number: float, metric_name: str) -> float:
    assert metric_name in METRICS
    if any([e in metric_name for e in ("hd", "asd")]):
        return round(number, 2)
    return round(float(number * 100), 2)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, ensure_ascii=False, indent=4)


def calc_metric_per_class(pred: np.ndarray, gt: np.ndarray) -> List[float]:
    """
    input ndarray shape:
        pred: [height, width]
        gt  : [height, width]

    return:
        dice, hd95, jaccard, asd, sensitivity, specificity, precision, accuracy
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    tp = np.logical_and(pred == 1, gt == 1).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()

    eps = 1e-7

    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    jaccard = tp / (tp + fp + fn + eps)
    se = tp / (tp + fn + eps)
    sp = tn / (tn + fp + eps)
    pr = tp / (tp + fp + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    # 距离类指标需要 pred 和 gt 都有前景，否则 medpy 会报错
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.assd(pred, gt)
    elif pred.sum() == 0 and gt.sum() == 0:
        # 两者都没有前景，认为完全正确
        dice = 1.0
        jaccard = 1.0
        se = 1.0
        sp = 1.0
        pr = 1.0
        acc = 1.0
        hd95 = 0.0
        asd = 0.0
    else:
        # 一个有前景，一个没有前景，区域指标已经由 TP/FP/FN 算出
        # 距离类指标无法稳定计算，这里设为 0，避免程序报错
        hd95 = 0.0
        asd = 0.0

    return dice, hd95, jaccard, asd, se, sp, pr, acc


def test_single_image(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    use_tta: bool = False,
    threshold: float = 0.5,
    **kwargs: Any,
) -> List[Any]:
    """
    input tensor shape:
        image: [1, 3, height, width]
        label: [1, height, width]
    """
    assert image.shape[0] == 1, f"batch size must be 1, not {image.shape[0]}"
    assert num_classes > 1, "only support multi-classes evaluation"

    label_np = label.squeeze(0).cpu().detach().numpy()

    device = try_gpu()
    input_tensor = image.float().to(device)

    model.eval()
    with torch.no_grad():
        if use_tta:
            raw_0 = model(input_tensor)
            assert isinstance(raw_0, torch.Tensor), "Multiple outputs detected"
            out_0 = torch.softmax(raw_0, dim=1)

            raw_h = model(torch.flip(input_tensor, dims=[3]))
            assert isinstance(raw_h, torch.Tensor), "Multiple outputs detected"
            out_h = torch.flip(torch.softmax(raw_h, dim=1), dims=[3])

            raw_v = model(torch.flip(input_tensor, dims=[2]))
            assert isinstance(raw_v, torch.Tensor), "Multiple outputs detected"
            out_v = torch.flip(torch.softmax(raw_v, dim=1), dims=[2])

            raw_hv = model(torch.flip(input_tensor, dims=[2, 3]))
            assert isinstance(raw_hv, torch.Tensor), "Multiple outputs detected"
            out_hv = torch.flip(torch.softmax(raw_hv, dim=1), dims=[2, 3])

            prob_mean = (out_0 + out_h + out_v + out_hv) / 4.0

            if num_classes == 2:
                out = (prob_mean[:, 1] > threshold).long().squeeze(0)
            else:
                out = torch.argmax(prob_mean, dim=1).squeeze(0)

        else:
            outputs = model(input_tensor)
            assert isinstance(outputs, torch.Tensor), "Multiple outputs detected"

            prob = torch.softmax(outputs, dim=1)

            if num_classes == 2:
                out = (prob[:, 1] > threshold).long().squeeze(0)
            else:
                out = torch.argmax(prob, dim=1).squeeze(0)

        prediction = out.cpu().detach().numpy()

    metrics = []
    for c in range(1, num_classes):
        metrics.append(calc_metric_per_class(prediction == c, label_np == c))

    if label_np.sum() > 0:
        mcc = matthews_corrcoef(
            y_true=label_np.reshape(-1),
            y_pred=prediction.reshape(-1),
        )
    else:
        mcc = 1.0 if prediction.sum() == 0 else 0.0

    return metrics, mcc


def inference(
    model: nn.Module,
    dataloader: data.DataLoader,
    num_classes: int,
    use_tta: bool = False,
    threshold: float = 0.5,
    **kwargs: Any,
) -> Dict:
    eval_metrics = {"per_case": {}, "mean_case": None, "mean_metric": {}}
    metric_list = 0.0
    mcc_list = 0.0

    for sample in tqdm(dataloader, desc="Testing", ncols=100):
        image, label = sample["image"], sample["label"]
        case_name = sample["case_name"][0]

        metric_overall, mcc = test_single_image(
            model=model,
            image=image,
            label=label,
            num_classes=num_classes,
            use_tta=use_tta,
            threshold=threshold,
            **kwargs,
        )

        metric_list += np.array(metric_overall)
        mcc_list += mcc

        metric_avg_c = np.mean(metric_overall, axis=0)
        eval_metrics["per_case"][case_name] = {
            "overall": metric_overall,
            "mcc": mcc,
            "avg_c": metric_avg_c.tolist(),
        }

    metric_list = metric_list / len(dataloader)
    eval_metrics["mean_case"] = metric_list.tolist()

    for class_name, (i, _) in CLS2COLOR_MAPPING[num_classes].items():
        t = f"#class_name: {class_name}\n"
        for j, name in enumerate(METRICS):
            t += f"{name}: {pretty_number(metric_list[i - 1][j], name)}\n"
        logger.info(t)

    mean_metric = np.mean(metric_list, axis=0)
    mean_mcc = mcc_list / len(dataloader)

    mean_metric_dict = {"mcc": mean_mcc}
    t = f"Performance: \nmcc: {pretty_number(mean_mcc, 'dc')}\n"

    for i, name in enumerate(METRICS):
        t += f"{name}: {pretty_number(mean_metric[i], name)}\n"
        mean_metric_dict[name] = mean_metric[i]

    logger.info(t)
    eval_metrics["mean_metric"] = mean_metric_dict

    return eval_metrics


def load_training_cfg(model: str, dataset: str, config_name: str = None) -> Optional[Dict]:
    from importlib import import_module

    if config_name:
        mod = import_module(f"config.{config_name}")
        logger.info(f"Loaded training cfg from {mod.__name__}")
        return getattr(mod, "CONFIG", None)

    for cfg in glob(osp.join(".", "config", f"{model}*.py")):
        if dataset in cfg:
            mod = import_module(f"config.{osp.splitext(osp.basename(cfg))[0]}")
            logger.info(f"Loaded training cfg from {mod.__name__}")
            return getattr(mod, "CONFIG", None)

    return None


def load_checkpoint(model: nn.Module, log_dir: str, ckpt_name: str = "best.ckpt") -> nn.Module:
    """
    优先加载新训练脚本保存的 checkpoints/best.ckpt。
    如果找不到 best.ckpt，则兼容旧版本的 epoch*.ckpt。
    """
    ckpt_dir = osp.join(log_dir, "checkpoints")
    ckpt_path = osp.join(ckpt_dir, ckpt_name)

    if osp.exists(ckpt_path):
        selected_ckpt = ckpt_path
    else:
        ckpt_names = glob(osp.join(ckpt_dir, "epoch*.ckpt"))
        assert len(ckpt_names) == 1, (
            f"找不到 {ckpt_path}，且旧版 epoch*.ckpt 数量不是 1。\n"
            f"ckpt_dir={ckpt_dir}\n"
            f"found={ckpt_names}"
        )
        selected_ckpt = ckpt_names[0]

    ckpt = torch.load(selected_ckpt, map_location="cpu")

    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("_model.", "", 1)] = v

    model = model.to("cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(try_gpu())
    logger.info(f"Loaded checkpoint from {selected_ckpt}")

    return model


def get_log_dirs(
    log_root_dir: str,
    dataset: str,
    config_name: str,
    tag_str: str,
    fold_test: bool,
) -> List[str]:
    """
    根据训练保存目录查找待测试模型。
    fold_test=True:
        搜索 fold_0, fold_1, ..., fold_4
    fold_test=False:
        搜索 r0, r1, ...
    """
    if fold_test:
        if tag_str:
            search_pattern = osp.join(log_root_dir, dataset, config_name, f"fold_*{tag_str}")
        else:
            search_pattern = osp.join(log_root_dir, dataset, config_name, "fold_*")
    else:
        search_pattern = osp.join(log_root_dir, dataset, config_name, f"r*{tag_str}")

    log_names = glob(search_pattern)

    if fold_test:
        filtered = []
        for p in log_names:
            name = osp.basename(p)
            if tag_str:
                pattern = r"^fold_[0-4]" + re.escape(tag_str) + r"$"
            else:
                pattern = r"^fold_[0-4]$"
            if re.match(pattern, name):
                filtered.append(p)
        log_names = filtered

        log_names.sort(key=lambda x: int(re.findall(r"fold_(\d+)", osp.basename(x))[0]))
    else:
        log_names.sort()

    assert len(log_names) >= 1, f"找不到任何符合条件的训练日志文件夹: {search_pattern}"

    return log_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str)
    parser.add_argument("-d", "--dataset", required=True, type=str)
    parser.add_argument("-o", "--output", default="./test_results", type=str)

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="指定配置文件的名字，例如 skin_unet_isic2018",
    )

    parser.add_argument("-t", "--tag", type=str, default="", help="自定义实验标签，例如 exp1")
    parser.add_argument("--tta", action="store_true", help="是否启用 Test-Time Augmentation")
    parser.add_argument("--fold_test", action="store_true", help="测试五折模型 fold_0 到 fold_4")
    parser.add_argument("--ckpt_name", type=str, default="best.ckpt", help="要加载的 checkpoint 名字")
    parser.add_argument(
        "--log_root",
        type=str,
        default="./log_train",
        help="root directory of training logs and checkpoints",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="lesion probability threshold for binary segmentation",
    )

    args = parser.parse_args()

    tag_str = f"-{args.tag}" if args.tag else ""
    tta_str = "-tta" if args.tta else ""

    threshold_str = f"-th{args.threshold:.2f}".replace(".", "p")
    result_name = f"{args.model}-{args.dataset}{tag_str}{tta_str}{threshold_str}"

    assert args.dataset in DATASETS, f"dataset {args.dataset} not found"
    dataset_cfg = DATASETS[args.dataset]

    train_cfg = load_training_cfg(args.model, args.dataset, args.config)
    assert train_cfg is not None, "train_cfg not found"

    config_name = args.config if args.config else f"{args.model}_{args.dataset}"

    root = osp.join(args.output, args.dataset, config_name, args.model)
    os.makedirs(root, exist_ok=True)

    logger.add(osp.join(root, f"test-{result_name}.log"))

    logger.info("=" * 60)
    logger.info(f"🚀 正在测试数据集: {args.dataset.upper()}")
    logger.info(f"模型名称: {args.model}")
    logger.info(f"配置名称: {config_name}")
    logger.info(f"是否五折测试: {args.fold_test}")
    logger.info(f"是否使用 TTA: {args.tta}")
    logger.info(f"Checkpoint: {args.ckpt_name}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"结果保存目录: {root}")
    logger.info("=" * 60)

    base_dir = osp.expandvars(osp.join("$DATASET_HOME", dataset_cfg["root_suffix"]))
    if "$DATASET_HOME" in base_dir or not osp.exists(base_dir):
        raise FileNotFoundError(
            f"数据根目录不存在: {base_dir}\n"
            f"请先执行:\n"
            f"export DATASET_HOME=/data/pytorch/wan/datasets"
        )

    tf_name, tf_cfg = parse_cfg(train_cfg, "test_transform")
    test_transform = TRANSFORMS[tf_name](**tf_cfg) if tf_name else None

    test_dataset = BaseDataset(
        base_dir=base_dir,
        split="test",
        list_dir=dataset_cfg["list_dir"],
        transform=test_transform,
    )

    loader_name, loader_cfg = parse_cfg(train_cfg, "val_dataloader")
    test_dataloader = DATALOADERS[loader_name](test_dataset, **loader_cfg)

    logger.info(f"测试集路径 base_dir: {base_dir}")
    logger.info(f"测试集样本数: {len(test_dataset)}")
    logger.info(f"测试 list_dir: {dataset_cfg['list_dir']}")

    log_root_dir = args.log_root
    log_names = get_log_dirs(
        log_root_dir=log_root_dir,
        dataset=args.dataset,
        config_name=config_name,
        tag_str=tag_str,
        fold_test=args.fold_test,
    )

    logger.info(f"找到 {len(log_names)} 个匹配的实验文件夹准备测试:")
    for p in log_names:
        logger.info(f"  - {p}")

    results = {}

    for i, log_dir in enumerate(log_names):
        logger.info("=" * 60)
        logger.info(f"正在测试第 {i + 1}/{len(log_names)} 个模型: {log_dir}")
        logger.info("=" * 60)

        model_name, model_cfg = parse_cfg(train_cfg, "model")
        assert model_name == args.model

        model = build_model(
            name=args.model,
            in_channels=train_cfg.get("in_channels", 3),
            num_classes=dataset_cfg["num_classes"],
            **model_cfg,
        )

        model = load_checkpoint(model, log_dir, ckpt_name=args.ckpt_name)

        results[i] = inference(
            model=model,
            dataloader=test_dataloader,
            num_classes=dataset_cfg["num_classes"],
            use_tta=args.tta,
            threshold=args.threshold,
        )

    final_summary = {}
    print_str = "\n" + "=" * 50 + "\n"

    if args.fold_test:
        print_str += f"🏅 最终汇总成绩，基于 {len(log_names)} 个 fold\n"
    else:
        print_str += f"🏅 最终汇总成绩，基于 {len(log_names)} 次独立实验\n"

    print_str += "=" * 50 + "\n"

    all_metric_names = ["mcc"] + list(METRICS)

    for metric_name in all_metric_names:
        raw_values = [results[i]["mean_metric"][metric_name] for i in range(len(log_names))]

        raw_mean = np.mean(raw_values)
        raw_std = np.std(raw_values, ddof=1) if len(raw_values) > 1 else 0.0

        format_name = "dc" if metric_name == "mcc" else metric_name

        pretty_mean = pretty_number(raw_mean, format_name)
        pretty_std = pretty_number(raw_std, format_name)

        formatted_str = f"{pretty_mean} ± {pretty_std}"

        final_summary[metric_name] = {
            "mean": pretty_mean,
            "std": pretty_std,
            "raw_mean": float(raw_mean),
            "raw_std": float(raw_std),
            "formatted": formatted_str,
            "per_run_raw_values": [float(v) for v in raw_values],
        }

        print_str += f"{metric_name.upper():<6}: {formatted_str}\n"

    print_str += "=" * 50
    logger.info(print_str)

    results["final_summary"] = final_summary
    results["tested_log_dirs"] = log_names
    results["test_config"] = {
        "dataset": args.dataset,
        "model": args.model,
        "config": config_name,
        "fold_test": args.fold_test,
        "tta": args.tta,
        "tag": args.tag,
        "ckpt_name": args.ckpt_name,
        "threshold": args.threshold,
        "result_dir": root,
    }

    json_path = osp.join(root, f"{result_name}.json")
    save_json(results, json_path)

    logger.info(f"所有测试完成！汇总结果已保存至: {json_path}")
