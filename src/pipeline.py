import torch
import timm
import math
from tqdm import tqdm
import wandb
import ray.tune as tune
from torch.utils.data import Subset
import torch.nn as nn
from loguru import logger

from config import Config
from src.utils import (
    wandb_log,
    deterministic,
    show_config,
    timer,
    mem_trace,
    CumulativeTimer,
    count_correct,
    get_logger,
)
from src.adaptation import tent
from src.adaptation import eata
from src.adaptation import sar
from src.adaptation import deyo
from src.adaptation import cotta
from src.adaptation import mgtta
from src.adaptation import becotta
from src.adaptation.sam import SAM
from src.dataset.dataset import get_data, get_reference_data_name, prepare_test_data
from src.adaptation.vpt import FOAViT
from src.adaptation.moetta import MoETTA
from src.adaptation.moe_normalization import switch_to_MoE


CIFAR10_TO_IMAGENET_CLASS_GROUPS: tuple[tuple[int, ...], ...] = (
    (404, 405, 895),  # airplane
    (436, 468, 511, 609, 627, 654, 656, 751, 817, 829),  # automobile
    tuple(range(7, 25)) + tuple(range(80, 102)) + tuple(range(127, 147)),  # bird
    (281, 282, 283, 284, 285, 383),  # cat
    (351, 352, 353),  # deer-like ungulates in ImageNet-1k
    tuple(range(151, 269)),  # dog
    (30, 31, 32),  # frog
    (339,),  # horse
    (403, 472, 484, 510, 554, 625, 628, 724, 780, 814, 833, 871, 914),  # ship
    (555, 569, 675, 717, 734, 864, 867),  # truck
)
_cifar10_group_index_cache: dict[str, list[torch.Tensor]] = {}


def adapt_logits_for_dataset(
    logits: torch.Tensor, config: Config, device: torch.device
) -> torch.Tensor:
    """Map model logits to dataset label space when needed."""
    if config.data.corruption not in {"cifar10", "cifar10-c"}:
        return logits
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits [B, C], got shape={tuple(logits.shape)}")
    if logits.shape[1] == 10:
        return logits
    if logits.shape[1] != 1000:
        logger.warning(
            "CIFAR-10 evaluation expects 10 or 1000 logits, "
            f"but got {logits.shape[1]}. Skip CIFAR logit adaptation."
        )
        return logits

    cache_key = str(device)
    if cache_key not in _cifar10_group_index_cache:
        _cifar10_group_index_cache[cache_key] = [
            torch.tensor(indices, device=device, dtype=torch.long)
            for indices in CIFAR10_TO_IMAGENET_CLASS_GROUPS
        ]
    group_indices = _cifar10_group_index_cache[cache_key]
    class_logits = [
        logits.index_select(1, class_index).amax(dim=1) for class_index in group_indices
    ]
    return torch.stack(class_logits, dim=1)


def resolve_device(config: Config) -> torch.device:
    requested = str(config.env.device).strip().lower()
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(config.env.device)
        else:
            logger.warning(
                "env.device is set to CUDA but CUDA is unavailable. Falling back to CPU."
            )
            device = torch.device("cpu")
    else:
        try:
            device = torch.device(config.env.device)
        except (RuntimeError, ValueError, TypeError):
            logger.warning(
                f"Invalid env.device '{config.env.device}'. Falling back to CPU."
            )
            device = torch.device("cpu")
    config.env.device = str(device)
    return device


@CumulativeTimer
def validate(val_loader, model, config: Config, device: torch.device):
    n_top1, n_top5, n_sample = 0, 0, 0
    for i, batch in enumerate(tqdm(val_loader)):
        images = batch[0].to(device, non_blocking=device.type == "cuda")
        target = batch[1].to(device, non_blocking=device.type == "cuda")

        if config.algo.algorithm != "mgtta":
            with torch.no_grad():
                output = model(images)
        else:
            output, loss = model(images)

        output = adapt_logits_for_dataset(output, config, device)
        # measure accuracy and record loss
        topk_max = min(5, output.shape[1])
        acc1, acc5 = count_correct(output, target, topk=(1, topk_max))
        n_top1 += acc1
        n_top5 += acc5
        n_sample += images.shape[0]

        del output
        del target
        # measure elapsed time
        wandb.log(
            dict(
                batch_accuracy_1=acc1 / images.shape[0],
                batch_accuracy_5=acc5 / images.shape[0],
                overall_accuracy_1=n_top1 / n_sample,
                overall_accuracy_5=n_top5 / n_sample,
            ),
            step=i,
        )
        if config.tune.search_space:
            tune.report(
                dict(
                    batch_accuracy_1=acc1 / images.shape[0],
                    batch_accuracy_5=acc5 / images.shape[0],
                    overall_accuracy_1=n_top1 / n_sample,
                    overall_accuracy_5=n_top5 / n_sample,
                    step=i,
                )
            )

    return n_top1 / n_sample


def configure_model(config: Config, device: torch.device):
    net = timm.create_model(config.model.model, pretrained=True)
    net = net.to(device)
    net.eval()
    net.requires_grad_(False)
    reference_data_name = get_reference_data_name(config)

    match config.algo.algorithm:
        case "tent":
            net = tent.configure_model(net)
            if config.algo.switch_to_MoE:
                switch_to_MoE(net, config)
            params, _ = tent.collect_params(net)
            optimizer = torch.optim.SGD(params, config.algo.tent.lr, momentum=0.9)
            adapt_model = tent.Tent(net, optimizer)
        case "eata":
            # compute fisher informatrix
            fisher_dataset = get_data(reference_data_name, config)
            fisher_dataset = Subset(
                fisher_dataset,
                torch.randperm(len(fisher_dataset))[: config.algo.eata.fisher_size],
            )
            fisher_loader = torch.utils.data.DataLoader(
                fisher_dataset,
                batch_size=config.train.batch_size,
                shuffle=config.data.shuffle,
                num_workers=config.train.workers,
                pin_memory=device.type == "cuda",
            )
            fisher_loader = torch.utils.data.DataLoader(
                fisher_dataset,
                batch_size=config.train.batch_size,
                shuffle=config.data.shuffle,
                num_workers=config.train.workers,
                pin_memory=device.type == "cuda",
            )

            net = eata.configure_model(net)
            if config.algo.switch_to_MoE:
                switch_to_MoE(net, config)
            params, param_names = eata.collect_params(net)
            ewc_optimizer = torch.optim.SGD(params, 0.001)
            fishers = {}
            train_loss_fn = nn.CrossEntropyLoss().to(device)
            for iter_, (images, targets) in enumerate(fisher_loader, start=1):
                targets = targets.to(device, non_blocking=device.type == "cuda")
                images = images.to(device, non_blocking=device.type == "cuda")
                outputs = net(images)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in net.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = (
                                param.grad.data.clone().detach() ** 2 + fishers[name][0]
                            )
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == len(fisher_loader):
                            fisher = fisher / iter_
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()
            del ewc_optimizer
            optimizer = torch.optim.SGD(params, config.algo.eata.lr, momentum=0.9)
            adapt_model = eata.EATA(
                net,
                optimizer,
                fishers,
                config.algo.eata.fisher_alpha,
                e_margin=config.algo.eata.e_margin_coeff
                * math.log(config.data.num_class),
                d_margin=config.algo.eata.d_margin,
            )
        case "deyo":
            net = deyo.configure_model(net)
            if config.algo.switch_to_MoE:
                switch_to_MoE(net, config)
            params, _ = deyo.collect_params(net)
            optimizer = torch.optim.SGD(params, config.algo.deyo.lr, momentum=0.9)
            adapt_model = deyo.DeYO(net, config, optimizer)
        case "sar":
            net = sar.configure_model(net)
            if config.algo.switch_to_MoE:
                switch_to_MoE(net, config)
            params, _ = sar.collect_params(net)
            base_optimizer = torch.optim.SGD
            optimizer = SAM(params, base_optimizer, lr=config.algo.sar.lr, momentum=0.9)
            # NOTE: set margin_e0 to 0.4*math.log(200) on ImageNet-R
            adapt_model = sar.SAR(
                net,
                optimizer,
                margin_e0=config.algo.sar.margin_e0_coeff
                * math.log(config.data.num_class),
                reset_constant_em=config.algo.sar.reset_constant_em,
            )
        case "cotta":
            net = cotta.configure_model(net)
            params, _ = cotta.collect_params(net)
            optimizer = torch.optim.SGD(params, lr=config.algo.cotta.lr, momentum=0.9)
            adapt_model = cotta.CoTTA(net, optimizer, steps=1, episodic=False)
        case "mgtta":
            net = FOAViT(net).to(device)
            mgg = mgtta.create_mgg(
                config.algo.mgtta.mgg_path,
                hidden_size=config.algo.mgtta.ttt_hidden_size,
                num_attention_heads=config.algo.mgtta.num_attention_heads,
            ).to(device)
            adapt_model = mgtta.MGTTA(
                net, mgg, config.algo.mgtta.lr, norm_dim=config.algo.mgtta.norm_dim
            )
            train_set = get_data(reference_data_name, config)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=config.train.batch_size,
                shuffle=config.data.shuffle,
                num_workers=config.train.workers,
                pin_memory=device.type == "cuda",
            )
            adapt_model.obtain_origin_stat(
                train_loader, config.algo.mgtta.train_info_path
            )
            adapt_model.configure_model()
        case "becotta":
            net = becotta.configure_model(net, config)
            params, _ = becotta.collect_params(net)
            optimizer = torch.optim.Adam(
                params, config.algo.becotta.lr, betas=(0.9, 0.999)
            )
            adapt_model = becotta.BECoTTA(net, optimizer)
        case "noadapt":
            adapt_model = net
        case "moetta":
            adapt_model = MoETTA(
                net,
                config,
                num_expert=config.algo.moetta.num_expert,
                topk=config.algo.moetta.topk,
                dynamic_threshold=config.algo.moetta.dynamic_threshold,
                dynamic_lb=config.algo.moetta.dynamic_lb,
                lb_coeff=config.algo.moetta.lb_coeff,
                weight_by_prob=config.algo.moetta.weight_by_prob,
                weight_by_entropy=config.algo.moetta.weight_by_entropy,
                e_margin_coeff=config.algo.moetta.e_margin_coeff,
                randomness=config.algo.moetta.randomness,
                activate_shared_expert=config.algo.moetta.activate_shared_expert,
                route_penalty=config.algo.moetta.route_penalty,
                decay=config.algo.moetta.decay,
                self_router=config.algo.moetta.self_router,
                samplewise=config.algo.moetta.samplewise,
                moe_logger=get_logger(config),
                grad_hook=config.algo.moetta.grad_hook,
                disabled_layer=config.algo.moetta.disabled_layer,
                normal_layer=config.algo.moetta.normal_layer,
                pass_through_coeff=config.algo.moetta.pass_through_coeff,
                global_router_idx=config.algo.moetta.global_router_idx,
                device=config.env.device
            )
        case _:
            raise ValueError("Invalid algorithm!")

    return adapt_model


@wandb_log
@show_config
@deterministic
@mem_trace
@timer
def pipeline(config: Config):
    device = resolve_device(config)
    model = configure_model(config, device)
    val_dataset, val_loader = prepare_test_data(config)
    acc = validate(val_loader, model, config, device)
    return acc
