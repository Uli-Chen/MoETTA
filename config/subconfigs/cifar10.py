from dataclasses import replace

from ..config import CONFIG, Config

config_base = Config()
config_cifar10 = replace(
    config_base,
    data=replace(
        config_base.data,
        num_class=10,
        corruption="cifar10-c",
        cifar_corruption="gaussian_noise",
    ),
)
CONFIG["cifar10"] = ("Config for CIFAR-10-C corruption", config_cifar10)
