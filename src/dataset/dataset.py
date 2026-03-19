import os
from pathlib import Path

import numpy as np
import timm
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from torchvision.datasets import ImageFolder

from config import Config
from .ImageNetMask import a_to_origin, r_to_origin

COMMON_CORRUPTIONS_15 = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

COMMON_CORRUPTIONS_4 = [
    "speckle_noise",
    "spatter",
    "gaussian_blur",
    "saturate",
]

COMMON_CORRUPTIONS = COMMON_CORRUPTIONS_15 + COMMON_CORRUPTIONS_4


class CIFARCorruptionDataset(Dataset):
    def __init__(
        self,
        root: str,
        corruption: str,
        level: int,
        transform=None,
    ):
        self.transform = transform
        if level < 1 or level > 5:
            raise ValueError(f"CIFAR-C severity level must be in [1, 5], got {level}.")
        data_path = os.path.join(root, f"{corruption}.npy")
        labels_path = os.path.join(root, "labels.npy")
        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(
                "CIFAR-C npy format expects both "
                f"'{data_path}' and '{labels_path}'."
            )

        data = np.load(data_path)
        labels = np.load(labels_path)
        if data.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Mismatched CIFAR-C sample count: {data.shape[0]} vs {labels.shape[0]}"
            )

        chunk_size = data.shape[0] // 5
        if chunk_size * 5 != data.shape[0]:
            raise ValueError(
                f"Unexpected CIFAR-C sample count {data.shape[0]} for {corruption}."
            )
        start = (level - 1) * chunk_size
        end = level * chunk_size
        self.data = data[start:end]
        self.labels = labels[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx])
        if self.transform:
            image = self.transform(image)
        label = int(self.labels[idx])
        return image, label


def _expand_path(path: str | Path) -> str:
    return os.path.expanduser(str(path))


def _build_cifar_c_dataset(root: str, corruption: str, level: int, transform):
    npy_file = os.path.join(root, f"{corruption}.npy")
    labels_file = os.path.join(root, "labels.npy")
    if os.path.exists(npy_file) and os.path.exists(labels_file):
        return CIFARCorruptionDataset(root, corruption, level, transform)

    imagefolder_path = os.path.join(root, corruption, str(level))
    if os.path.isdir(imagefolder_path):
        return ImageFolder(root=imagefolder_path, transform=transform)

    raise FileNotFoundError(
        f"Cannot find CIFAR-C data under {root}. "
        "Expected either '*.npy + labels.npy' format or ImageFolder format "
        f"at '{imagefolder_path}'."
    )


def get_reference_data_name(config: Config) -> str:
    match config.data.corruption:
        case "cifar10" | "cifar10-c":
            return "cifar10"
        case "cifar100" | "cifar100-c":
            return "cifar100"
        case _:
            return "original"


def get_data(corruption, config: Config):
    model = timm.create_model(config.model.model, pretrained=False)
    data_cfg = timm.data.resolve_data_config({}, model=model)
    normalize = transforms.Normalize(
        mean=data_cfg["mean"],
        std=data_cfg["std"],
    )
    del model
    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transforms_imagenet_C = transforms.Compose(
        [transforms.CenterCrop(224), transforms.ToTensor(), normalize]
    )

    match corruption:
        case "original":
            test_set = ImageFolder(
                root=os.path.join(os.path.expanduser(config.env.original_data_path), "val"),
                transform=test_transforms,
            )
        case corruption if corruption in COMMON_CORRUPTIONS:
            test_set = ImageFolder(
                root=os.path.join(
                    os.path.expanduser(config.env.corruption_data_path),
                    corruption,
                    str(config.data.level),
                ),
                transform=test_transforms_imagenet_C,
            )
        case "rendition":
            test_set = ImageFolder(
                root=os.path.expanduser(config.env.rendition_data_path),
                transform=test_transforms,
                target_transform=lambda idx: r_to_origin[idx],
            )
        case "sketch":
            test_set = datasets.ImageFolder(
                root=os.path.expanduser(config.env.sketch_data_path),
                transform=test_transforms,
            )
        case "imagenet_a":
            test_set = datasets.ImageFolder(
                root=os.path.expanduser(config.env.adv_data_path),
                transform=test_transforms,
                target_transform=lambda idx: a_to_origin[idx],
            )
        case "cifar10":
            test_set = datasets.CIFAR10(
                root=_expand_path(config.env.cifar10_data_path),
                train=False,
                download=config.data.download,
                transform=test_transforms,
            )
        case "cifar100":
            test_set = datasets.CIFAR100(
                root=_expand_path(config.env.cifar100_data_path),
                train=False,
                download=config.data.download,
                transform=test_transforms,
            )
        case "cifar10-c":
            test_set = _build_cifar_c_dataset(
                root=_expand_path(config.env.cifar10c_data_path),
                corruption=config.data.cifar_corruption,
                level=config.data.level,
                transform=test_transforms,
            )
        case "cifar100-c":
            test_set = _build_cifar_c_dataset(
                root=_expand_path(config.env.cifar100c_data_path),
                corruption=config.data.cifar_corruption,
                level=config.data.level,
                transform=test_transforms,
            )
        case _:
            raise ValueError(f"Corruption not found: {corruption}")

    return test_set


def prepare_test_data(config: Config):
    match config.data.corruption:
        case "original" | "rendition" | "sketch" | "imagenet_a":
            test_set = get_data(config.data.corruption, config)
        case "cifar10" | "cifar100" | "cifar10-c" | "cifar100-c":
            test_set = get_data(config.data.corruption, config)
        case corruption if corruption in COMMON_CORRUPTIONS:
            test_set = get_data(corruption, config)
        case "imagenet_c_test_mix":
            dataset_list = [
                get_data(corruption, config) for corruption in COMMON_CORRUPTIONS_15
            ]
            test_set = torch.utils.data.ConcatDataset(dataset_list)  # 合并多个dataset
            if config.data.used_data_num != -1:
                logger.info(
                    f"Creating subset of {config.data.used_data_num} samples from imagenet_c_test_mix"
                )
                test_set = Subset(
                    test_set, torch.randperm(len(test_set))[: config.data.used_data_num]
                )
        case "imagenet_c_val_mix":
            dataset_list = [
                get_data(corruption, config) for corruption in COMMON_CORRUPTIONS_4
            ]
            test_set = torch.utils.data.ConcatDataset(dataset_list)  # 合并多个dataset
            if config.data.used_data_num != -1:
                # create subset
                logger.info(
                    f"Creating subset of {config.data.used_data_num} samples from imagenet_c_val_mix"
                )
                test_set = Subset(
                    test_set, torch.randperm(len(test_set))[: config.data.used_data_num]
                )
        case "potpourri":
            dataset_list = [
                get_data(corruption, config) for corruption in COMMON_CORRUPTIONS_15
            ]
            dataset_list.append(get_data("rendition", config))
            dataset_list.append(get_data("sketch", config))
            dataset_list.append(get_data("imagenet_a", config))
            test_set = torch.utils.data.ConcatDataset(dataset_list)  # 合并多个dataset
        case "potpourri+":
            dataset_list = [
                get_data(corruption, config) for corruption in COMMON_CORRUPTIONS_15
            ]
            dataset_list.append(get_data("rendition", config))
            dataset_list.append(get_data("sketch", config))
            dataset_list.append(get_data("imagenet_a", config))
            dataset_list.append(get_data("original", config))
            test_set = torch.utils.data.ConcatDataset(dataset_list)  # 合并多个dataset
        case _:
            raise ValueError("Corruption not found!")

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.train.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.train.workers,
        pin_memory=True,
    )
    return test_set, test_loader
