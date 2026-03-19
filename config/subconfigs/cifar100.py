from dataclasses import replace

from ..config import CONFIG, Config

config_base = Config()
config_cifar100 = replace(
    config_base,
    data=replace(
        config_base.data,
        num_class=100,
        corruption="cifar100-c",
        cifar_corruption="gaussian_noise",
    ),
)
CONFIG["cifar100"] = ("Config for CIFAR-100-C corruption", config_cifar100)
