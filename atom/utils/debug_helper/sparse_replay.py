# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in JSONL probe for sparse MLA CUDA graph replay diagnosis."""

from __future__ import annotations

import json
import os
import threading
from typing import Any

import torch

from atom.utils import envs

TAG = "[SPARSE_REPLAY_C]"

_LOCK = threading.Lock()
_COUNT = 0
_SEEN_BY_EVENT: dict[str, int] = {}


def _get_rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


def enabled() -> bool:
    return bool(envs.ATOM_SPARSE_REPLAY_C_PATH) and _get_rank() == 0


def _layer_set() -> set[int] | None:
    raw = envs.ATOM_SPARSE_REPLAY_C_LAYERS
    if raw == "*":
        return None
    return {int(x) for x in raw.split(",") if x}


def should_log_layer(layer_num: int | None) -> bool:
    if not enabled():
        return False
    layers = _layer_set()
    return layers is None or (layer_num is not None and layer_num in layers)


def _limited_tensor(tensor: torch.Tensor, limit: int) -> torch.Tensor:
    if limit <= 0:
        return tensor.detach()
    return tensor.detach().reshape(-1)[:limit]


def tensor_head(tensor: torch.Tensor | None, limit: int) -> list[Any]:
    if tensor is None:
        return []
    return _limited_tensor(tensor, limit).cpu().tolist()


def tensor_tail(tensor: torch.Tensor | None, limit: int) -> list[Any]:
    if tensor is None or limit <= 0:
        return []
    flat = tensor.detach().reshape(-1)
    return flat[-limit:].cpu().tolist()


def tensor_stats(tensor: torch.Tensor | None, limit: int = 0) -> dict[str, Any]:
    if tensor is None:
        return {}
    sample = _limited_tensor(tensor, limit)
    if sample.numel() == 0:
        return {"shape": list(tensor.shape), "numel": int(tensor.numel())}

    sample_i64 = sample.to(torch.int64)
    return {
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "sample_numel": int(sample.numel()),
        "min": int(sample_i64.min().item()),
        "max": int(sample_i64.max().item()),
        "neg": int((sample_i64 < 0).sum().item()),
        "zero": int((sample_i64 == 0).sum().item()),
        "nonneg": int((sample_i64 >= 0).sum().item()),
        "pos": int((sample_i64 > 0).sum().item()),
    }


def checksum(tensor: torch.Tensor | None, limit: int = 4096) -> dict[str, Any]:
    if tensor is None:
        return {}
    sample = _limited_tensor(tensor, limit)
    if sample.numel() == 0:
        return {"shape": list(tensor.shape), "numel": int(tensor.numel())}

    sample_i64 = sample.to(torch.int64)
    return {
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "sample_numel": int(sample.numel()),
        "sum": int(sample_i64.sum().item()),
        "abs_sum": int(sample_i64.abs().sum().item()),
        "head": tensor_head(sample_i64, 8),
        "tail": tensor_tail(sample_i64, 8),
    }


def count_ge_by_row(values: torch.Tensor, limits: torch.Tensor, rows: int) -> int:
    if rows <= 0:
        return 0
    values_i64 = values[:rows].to(torch.int64)
    limits_i64 = limits[:rows].to(torch.int64).view(rows, 1)
    return int(((values_i64 >= 0) & (values_i64 >= limits_i64)).sum().item())


def maybe_log(event: str, **fields: Any) -> None:
    global _COUNT

    path = envs.ATOM_SPARSE_REPLAY_C_PATH
    if not path or _get_rank() != 0:
        return

    with _LOCK:
        event_seen = _SEEN_BY_EVENT.get(event, 0) + 1
        _SEEN_BY_EVENT[event] = event_seen
        interval = envs.ATOM_SPARSE_REPLAY_C_INTERVAL
        if interval > 1 and (event_seen - 1) % interval != 0:
            return
        max_records = envs.ATOM_SPARSE_REPLAY_C_MAX_RECORDS
        if max_records > 0 and _COUNT >= max_records:
            return
        _COUNT += 1
        seq = _COUNT

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {
        "tag": TAG,
        "seq": seq,
        "event": event,
        "event_seen": event_seen,
        "rank": 0,
    }
    record.update(fields)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, sort_keys=True) + "\n")
