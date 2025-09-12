import argparse
from pathlib import Path

import torch
import yaml

from src.loss.infonce import InfoNCELoss
from src.loss.pcmepp import ClosedFormSampledDistanceLoss

KEYS_ARGUMENT = [
    "model_cfg",
    "loss_cfg",
    "optimizer_cfg",
    "dataset_cfg",
    "sampler_cfg",
]


DICT_LOSS = {
    "info_nce": InfoNCELoss,
    "pcmepp": ClosedFormSampledDistanceLoss,
}


def get_loss_fn(loss_config: dict):  # type: ignore
    """Get the loss function based on the loss configuration."""
    loss_name = loss_config.get("loss_name")
    assert loss_name in list(DICT_LOSS.keys()), f"Unknown loss function: {loss_name}"
    return DICT_LOSS[loss_name](**loss_config)


@torch.no_grad()
def recall_at_k_from_cosine_sim_torch(
    sim: torch.Tensor,
    ks: tuple[int, ...] = (1, 5),
    exclude_self: bool = False,
) -> dict[int, float]:
    """Rough estimate of recall@k from cosine similarity matrix. We do not consider same text."""
    if sim.dim() != 2 or sim.size(0) != sim.size(1):
        raise ValueError("sim must be a square (N x N) tensor")
    N = sim.size(0)
    if N == 0:
        return {int(k): 0.0 for k in ks}

    ks = tuple(sorted({int(k) for k in ks if k >= 1}))
    max_k = max(ks)

    s = sim.clone()
    s = torch.nan_to_num(s, nan=float("-inf"))

    if exclude_self:
        s.fill_diagonal_(float("-inf"))

    _, topk_idx = torch.topk(s, k=max_k, dim=1, largest=True, sorted=True)  # [N, max_k]
    gt = torch.arange(N, device=sim.device)

    match = topk_idx == gt.unsqueeze(1)  # [N, max_k]

    recalls = {}
    for k in ks:
        hits_k = match[:, :k].any(dim=1).float().mean().item()
        recalls[k] = hits_k
    return recalls


def load_cfgs(args: argparse.Namespace) -> dict:
    """Load YAML configs into a single dictionary keyed by arg name."""
    root_path = Path(args.config_dir)
    assert root_path is not None, "make sure path is valid"
    cfgs = {}
    for arg_name in KEYS_ARGUMENT:
        # Get the filename from args (e.g., "model" -> "model.yaml")
        fname = root_path / f"{getattr(args, arg_name)}.yaml"
        with open(fname) as f:
            cfgs[arg_name] = yaml.safe_load(f)
    return cfgs


def device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
