import os.path as osp
from typing import Callable

import h5py
import numpy as np
from torch.utils.data import Dataset

__all__ = ["BaseDataset"]


class BaseDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        list_dir: str,
        split: str,
        transform: Callable = None,
        image_key: str = "image",
        label_key: str = "label",
    ) -> None:
        self.data_dir = base_dir
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

        assert isinstance(split, str) and len(split) > 0, "split must be a non-empty string"
        self.split = split

        list_path = osp.join(list_dir, self.split + ".txt")
        if not osp.exists(list_path):
            raise FileNotFoundError(f"List file not found: {list_path}")

        with open(list_path, "r", encoding="utf-8") as fp:
            self.sample_list = [line.strip() for line in fp.readlines() if line.strip()]

        if len(self.sample_list) == 0:
            raise RuntimeError(f"Empty sample list: {list_path}")

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> dict:
        """
        output tensor shape:
            {
                "case_name": str,
                "image": [C, H, W] or other format,
                "label": [H, W] or other format
            }
        """
        fname = self.sample_list[idx]

        if self.split == "test":
            data_subdir = "test"
        else:
            data_subdir = "train"

        data_path = osp.join(self.data_dir, data_subdir, fname)

        if not osp.exists(data_path):
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                f"split={self.split}, data_subdir={data_subdir}, fname={fname}"
            )

        sample = self.load_data(data_path)

        if self.transform:
            sample = self.transform(sample)

        sample["case_name"] = fname
        return sample

    def load_data(self, fname: str) -> dict:
        suffix = osp.splitext(fname)[1]

        if suffix == ".h5":
            data = h5py.File(fname, "r")

            actual_img_key = (
                self.image_key
                if self.image_key in data.keys()
                else ("data" if "data" in data.keys() else list(data.keys())[0])
            )

            actual_lab_key = (
                self.label_key
                if self.label_key in data.keys()
                else ("mask" if "mask" in data.keys() else "label")
            )

            return {
                "image": data[actual_img_key][:],
                "label": data[actual_lab_key][:],
            }

        elif suffix in (".npy", ".npz"):
            data = np.load(fname)

            if suffix == ".npz":
                available_keys = data.files

                if self.image_key in available_keys:
                    actual_img_key = self.image_key
                elif "data" in available_keys:
                    actual_img_key = "data"
                else:
                    actual_img_key = [k for k in available_keys if k != self.label_key][0]

                actual_lab_key = self.label_key if self.label_key in available_keys else "label"

                return {
                    "image": data[actual_img_key],
                    "label": data[actual_lab_key],
                }

            else:
                return {
                    "image": data,
                    "label": data,
                }

        else:
            raise ValueError(f"Unsupported file format: {fname}")
