from typing import Any, Tuple, Union

import numpy as np
from scipy.ndimage import zoom
import torch
from torch.nn.functional import interpolate

from .metric import SegMeter


def eval_single_volume(
    model: torch.nn.Module,
    volume: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    patch_size: Tuple[int, int] = (224, 224),
    device: Union[str, torch.device] = None,
    **kwargs: Any,
) -> dict:
    volume_np = volume.squeeze(0).cpu().detach().numpy()
    label_np = label.squeeze(0).cpu().detach().numpy()

    model.eval()
    prediction = np.zeros_like(label_np)

    if len(label_np.shape) == 2:
        if len(volume_np.shape) == 3:
            if volume_np.shape[-1] == 3:  # 如果是 [224, 224, 3]
                volume_np = volume_np.transpose(2, 0, 1)
            elif volume_np.shape[1] == 3: # 如果被扭曲成了 [224, 3, 224]
                volume_np = volume_np.transpose(1, 0, 2)
            
            inp = torch.from_numpy(volume_np)
            
        elif len(volume_np.shape) == 2:
            if kwargs.get("norm_x_transform", None) is not None:
                inp = kwargs.get("norm_x_transform")(volume_np)
            else:
                inp = torch.from_numpy(volume_np).unsqueeze(0).repeat(3, 1, 1)
        else:
            inp = torch.from_numpy(volume_np)

        inp = inp.unsqueeze(0).float().to(device)

        with torch.no_grad():
            out = model(inp)
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    else:
        for depth in range(volume_np.shape[0]):
            vol_slice = volume_np[depth, :, :]
            h, w = vol_slice.shape[0], vol_slice.shape[1]

            if h != patch_size[0] or w != patch_size[1]:
                vol_slice = zoom(vol_slice, (patch_size[0] / h, patch_size[1] / w), order=3)

            if kwargs.get("norm_x_transform", None) is not None:
                input = kwargs.get("norm_x_transform")(vol_slice)
            else:
                input = torch.from_numpy(vol_slice).unsqueeze(0)
            input = input.unsqueeze(0).float().to(device)

            with torch.no_grad():
                out = model(input)
                out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if h != patch_size[0] or w != patch_size[1]:
                    pred = zoom(out, (h / patch_size[0], w / patch_size[1]), order=0)
                else:
                    pred = out

            prediction[depth] = pred

    meter = SegMeter(num_classes=num_classes)
    meter(
        torch.from_numpy(prediction[None]).to(device),
        torch.from_numpy(label_np[None]).to(device),
    )
    metric = meter.get_metric()
    return metric


def eval_single_image(
    model: torch.nn.Module,
    image: torch.Tensor,  # [1, 3, h, w]
    label: torch.Tensor,  # [1, 1, h, w]
    num_classes: int,
) -> dict:
    assert num_classes == 1, "Only support binary segmentation evaluation"
    assert image.shape[0] == 1, "Only support batch size = 1 for single image evaluation"
    assert label.min() >= 0 and label.max() <= 1, "Ground Truth needs be normalized to the range between 0 and 1"

    model.eval()
    image = image.float().cuda()
    label = label.float().cuda()
    label = label / (label.max() + 1e-8)
    with torch.no_grad():
        out = model(image)
        out = interpolate(out, size=label.shape[-2:], mode="bilinear", align_corners=False)
        out = torch.sigmoid(out)  # [1, 1, h, w]
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)

    def dc_binary(result, reference, smooth=1):
        result = torch.reshape(result, (-1,))
        reference = torch.reshape(reference, (-1,))
        intersection = result * reference
        union = result.sum() + reference.sum()
        return (2 * intersection.sum() + smooth) / (union + smooth)

    return {"dice": {"roi": [dc_binary(out, label).item()]}}
