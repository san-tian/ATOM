# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py
# Copyright 2023 The vLLM team.
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import Any
import logging
from math import prod
from aiter.jit.utils.chip_info import get_gfx
from aiter.utility.fp4_utils import mxfp4_to_f32, f32_to_mxfp4
from atom.model_ops.moe_utils import (
    check_and_swizzle_scales,
    quantize,
)
from atom.utils import envs


logger = logging.getLogger("atom")


if envs.ATOM_USE_TRITON_GEMM:
    # try:
    #     from triton_kernels.matmul_ogs import matmul_ogs
    #     #from triton_kernels.routing import routing
    #     from triton_kernels.matmul_ogs import PrecisionConfig
    # except (AttributeError, ImportError) as e:
    #     logger.error(
    #         "Failed to import Triton kernels. Please make sure your triton "
    #         "version is compatible. Error: %s",
    #         e,
    #     )
    from aiter.ops.triton.moe.moe_routing.routing import routing
    from aiter.ops.triton.moe.moe_op_gemm_a4w4 import (
        mxfp4_quant,
        moe_gemm_a4w4,
    )
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (
        moe_gemm_a8w4,
    )
    from aiter.ops.triton.moe.quant_moe import downcast_to_static_fp8


def _swizzle_mxfp4(w1, w1_scale, w2, w2_scale, w_dtype, N_1, K_1, N_2, K_2, TP=1):
    """weight swizzle for mxfp4 moe, used for aiter triton mxfp4 moe kernel"""
    # is there any need for layouts or do i just swizzle scales and call it a day
    # weight and scale passed in
    # scrap all the layouts and whatever and just implement swizzle basically
    assert envs.ATOM_USE_TRITON_GEMM
    #from triton_kernels.numerics import InFlexData

    # this should do two things -- quantize and swizzle. 
    # 

    # x: ()
    # W1: (experts, N, K)
    # W2: (experts, N, K)
    # expected W1 for aiter triton matmul: (experts, N, K)
    # expected W2 for aiter triton matmul: (experts, K, N)
    w2_triton_layout = w2.transpose(-2, -1)
    w2_scale_triton_layout = w2_scale.transpose(-2, -1)

    # logger.warning("shape and dtype")
    # logger.warning(w1.shape)
    # logger.warning(w1.dtype)
    # logger.warning(w2.shape)
    # logger.warning(w2.dtype)
    # logger.warning(w2_triton_layout.shape)
    # logger.warning(N_1)
    # logger.warning(K_1)
    # logger.warning(N_2)
    # logger.warning(K_2)

    # i think this is the problem area. quantizing is taking the atom padding into account
    # maybe????????????
    # i think the swizzle perhaps is messing things up

    w1, w1_scale = quantize(w1, w_dtype)
    w2_triton_layout, w2_scale_triton_layout = quantize(w2_triton_layout, w_dtype)
    # removed for now
    w1_scale, _ = check_and_swizzle_scales(w1_scale, N_1, K_1 // TP)
    w2_scale_triton_layout, _ = check_and_swizzle_scales(
        w2_scale_triton_layout, K_2 // TP // 2, N_2
    )
    # logger.warning("post quant shape")
    # logger.warning(w1.shape)
    # logger.warning(w1)
    # logger.warning(w2.shape)
    # logger.warning(w2)
    # logger.warning(w2_triton_layout.shape)

    # from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    # from triton_kernels.tensor_details.layout import StridedLayout

    # value_layout_opts: dict[str, Any] = {}
    # scale_layout_opts: dict[str, Any] = {}
    # value_layout = StridedLayout
    # if get_gfx() == "gfx950":
    #     from triton_kernels.tensor_details.layout import GFX950MXScaleLayout

    #     scale_layout = GFX950MXScaleLayout
    # else:
    #     scale_layout = StridedLayout

    # quant_tensor = quant_tensor.transpose(-2, -1)
    # scale = scale.transpose(-2, -1)
    # quant_tensor = convert_layout(
    #     wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout, **value_layout_opts
    # )
    # scale = convert_layout(wrap_torch_tensor(scale), scale_layout, **scale_layout_opts)
    return w1, w1_scale, w2_triton_layout, w2_scale_triton_layout# transferring triton layout for now


def routing_from_topk(topk_weights, topk_ids, n_expts_tot):
    """Convert FusedMoE.select_experts output to triton routing data structures.

    This bridges the gap between ATOM's grouped topk / sigmoid routing
    (which triton_kernels routing() does not support) and the triton
    matmul_ogs compute kernels.

    Args:
        topk_weights: (n_tokens, n_expts_act) routing weights from select_experts
        topk_ids: (n_tokens, n_expts_act) expert indices from select_experts
        n_expts_tot: total number of experts (global, before EP)

    Returns:
        (RoutingData, GatherIndx, ScatterIndx) compatible with triton_kernel_fused_experts
    """
    # from triton_kernels.routing import (
    #     RoutingData,
    #     GatherIndx,
    #     ScatterIndx,
    #     compute_expt_data,
    # )
    from aiter.ops.triton.moe.moe_routing.routing import (
        routing,
        RoutingData,
        compute_expt_data_torch,
    )

    n_tokens, n_expts_act = topk_weights.shape
    n_gates_pad = n_tokens * n_expts_act

    # Sort each token's selected experts by expert_id (required by triton kernels)
    expt_indx_sorted, sort_indices = torch.sort(topk_ids.int(), dim=1)
    expt_scal_sorted = torch.gather(topk_weights, 1, sort_indices.long())

    # Flatten to 1D
    expt_scal = expt_scal_sorted.reshape(-1).to(topk_weights.dtype)
    expt_indx = expt_indx_sorted.reshape(-1).to(torch.int32)

    # Sort by expert_id globally so experts are contiguous for the matmul
    topk_indx = torch.argsort(expt_indx, stable=True).int()
    gate_indx = torch.argsort(topk_indx, stable=True).int()
    gate_scal = expt_scal[topk_indx.long()]

    # Histogram of tokens over experts
    hist = torch.histc(expt_indx.float(), bins=n_expts_tot, max=n_expts_tot - 1).int()

    # Build routing data structures using triton-accelerated compute_expt_data
    # gather_indx = GatherIndx(src_indx=topk_indx, dst_indx=gate_indx)
    # scatter_indx = ScatterIndx(src_indx=gate_indx, dst_indx=topk_indx)
    m = n_tokens * n_expts_act
    tokens_per_expt = max(1, m // n_expts_tot)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    expt_data = compute_expt_data_torch(hist, n_expts_tot, n_gates_pad, block_m)

    routing_data = RoutingData(gate_scal, hist, n_expts_tot, n_expts_act, expt_data)
    return routing_data, topk_indx, gate_indx#gather_indx, scatter_indx


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert (
        prod(v) <= x.numel()
    ), f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"  # CUDAGRAPH unfriendly?
    return x.flatten()[: prod(v)].view(*v)


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: str = "silu",
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a13_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w13_swizzle_layout: torch.Tensor | None = None,
    w2_swizzle_layout: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    x_q_dtype: str | None = None,
    static_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    routing_data, gather_idx, scatter_idx = routing(
        gating_output, topk, sm_first=not renormalize
    )

    output = torch.empty_like(hidden_states)


    return triton_kernel_fused_experts(
        output,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        topk=topk,
        activation=activation,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        a13_scale=a13_scale,
        a2_scale=a2_scale,
        w13_swizzle_layout=w13_swizzle_layout,
        w2_swizzle_layout=w2_swizzle_layout,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        x_q_dtype=x_q_dtype,
        static_scale=static_scale
    )


# This is a triton implementation of the fused_experts function
def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    routing_data,  # RoutingData
    gather_indx,  # GatherIndx -> tensor
    scatter_indx,  # ScatterIndx -> tensor
    topk: int,
    activation: str = "silu",
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w13_swizzle_layout: torch.Tensor | None = None,
    w2_swizzle_layout: torch.Tensor | None = None,
    a13_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    intermediate_cache: torch.Tensor | None = None,
    a1q_scale: torch.Tensor | None = None,
    x_q_dtype: str | None = None,
    static_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert w1_bias is None or w1_bias.dtype == torch.float32
    assert w2_bias is None or w2_bias.dtype == torch.float32

    # Shape check, only check non-mxfp4
    assert hidden_states.ndim == 2
    # assert hidden_states.shape[-1] == w1.shape[-2] # flipped to mimic benchmark weights -- #w1.shape[-2]
    # assert w2.shape[-2] == w1.shape[-1] # flipped to mimic benchmark weights -- both column major

    batch_dim = 1
    M, K = hidden_states.shape[-2:] # 
    E, _, N = w1.shape # N = hidden_size // 2?

    if global_num_experts == -1:
        global_num_experts = E

    half_N = N // 2

    if intermediate_cache is None:
        intermediate_cache = torch.empty(
            (batch_dim, M * topk, half_N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    # Add batch_dim to output buffer because matmul_ogs expects 3D output
    intermediate_cache = _resize_cache(
        intermediate_cache, (batch_dim, M * topk, half_N)
    )
    output_tensor = _resize_cache(output_tensor, (batch_dim, M, K))

    #gammas = routing_data.gate_scal if routing_data else None

    # On account of manual swiglu being required, HIP stream errors, and bench for triton having a 
    # static scale incorrect for given input states of about -3 - 3, we adjust static scale at 
    # each point of data loss to avoid overly incorrect results.
    moving_scale = static_scale

    # NOTE: We intentionally do NOT use the triton fused SwiGLU activation
    # because it expects interleaved [gate0, up0, gate1, up1, ...] layout
    # while our w13 weights produce concatenated [gate | up] output.
    # It also uses a non-standard formula: s*sigmoid(alpha*s)*(linear+1)
    # with alpha=1.702, which differs from the standard SiLU activation
    # (x*sigmoid(x)*up) used by most MoE models.
    # Instead, we compute the matmul without fused activation and apply
    # standard silu(gate) * up manually.
    raw_intermediate = torch.empty(
        (batch_dim, M * topk, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    # matmul_ogs(
    #     hidden_states,
    #     w1,
    #     w1_bias,
    #     routing_data,
    #     gather_indx=gather_indx,
    #     precision_config=w13_precision_config,
    #     gammas=gammas if apply_router_weight_on_input else None,
    #     y=raw_intermediate,
    # )
    x_q_dtype_base = x_q_dtype.split("_")[0]
    # logger.warning("split")
    # logger.warning(x_q_dtype_base)
    logger.warning("hidden states og")
    logger.warning(hidden_states)
    hidden_states_og = hidden_states
    if (x_q_dtype_base == "fp8"):
        from aiter.ops.triton.utils._triton.arch_info import get_arch

        # If input type is fp8, input scales must be available
        assert a13_scale is not None
        assert a2_scale is not None
        logger.warning(a13_scale)
        logger.warning(a2_scale)
        a13_scale = a13_scale.max().to(torch.float32)/448
        a2_scale = a2_scale.max().to(torch.float32)/448
        logger.warning(a13_scale)
        logger.warning(a2_scale)

        x_dtype = torch.float8_e4m3fn # model is fp8_e4m3 type

        if x_dtype == torch.float8_e4m3fn and get_arch() == "gfx942":
            x_dtype = torch.float8_e4m3fnuz

        hidden_states = downcast_to_static_fp8(hidden_states, a13_scale)
        quick_test = hidden_states.to(torch.bfloat16) * a13_scale
        logger.warning("hidden states")
        logger.warning(hidden_states)
        logger.warning(quick_test)
        raw_intermediate = moe_gemm_a8w4(
            hidden_states,
            w1,
            None,
            w13_scale,
            a13_scale,
            None,
            w1_bias,
            routing_data,
            gather_indx=gather_indx,
            swizzle_mx_scale="CDNA4_SCALE", # TODO assuming it's not None for simplicity's sake for now but should be checked
            out_dtype=raw_intermediate.dtype,
            apply_swiglu=False,
        ) 
        #quick_test = downcast_to_static_fp8(raw_intermediate, static_scale)
        logger.warning("quick test")
        logger.warning(raw_intermediate)
        #raw_intermediate = raw_intermediate * moving_scale * moving_scale
        # logger.warning("raw intermed")
        logger.warning(raw_intermediate)
    else:
        hidden_states, x_scale = mxfp4_quant(hidden_states)
        raw_intermediate = moe_gemm_a4w4(
            hidden_states,
            w1, # w1
            x_scale, # x scale
            w13_scale, # w1 scale
            None, # x static scale
            None, # quant static scale
            w1_bias,
            routing_data,
            gather_indx=gather_indx,
            swizzle_mx_scale="CDNA4_SCALE", # ?
            out_dtype=raw_intermediate.dtype,
            apply_swiglu=False,
        )

    # Standard SiLU/SwiGLU activation: silu(gate) * up
    # needed for all quant versions
    raw_2d = raw_intermediate.view(M * topk, N)
    logger.warning("raw 2d")
    logger.warning(raw_2d)
    logger.warning(intermediate_cache.shape)
    gate = raw_2d[:, :half_N]
    up = raw_2d[:, half_N:]
    logger.warning("issue with swiglu erasing intermediate cache")
    logger.warning(gate)
    logger.warning(up)
    logger.warning(torch.nn.functional.silu(gate))
    intermediate_cache = torch.nn.functional.silu(gate) * up
    logger.warning(intermediate_cache.shape)

    # matmul_ogs(
    #     intermediate_cache.view(M * topk, half_N),
    #     w2,
    #     w2_bias,
    #     routing_data,
    #     scatter_indx=scatter_indx,
    #     precision_config=w2_precision_config,
    #     gammas=None if apply_router_weight_on_input else gammas,
    #     y=output_tensor,
    # )


    if (x_q_dtype_base == "fp8"):
        # do not default to mxfp4
        logger.warning("second stage")
        from aiter.ops.triton.utils._triton.arch_info import get_arch

        x_dtype = torch.float8_e4m3fn # model is fp8_e4m3 type

        if x_dtype == torch.float8_e4m3fn and get_arch() == "gfx942":
            x_dtype = torch.float8_e4m3fnuz

        # logger.warning("check intermediate cache is not all gone")
        # logger.warning(intermediate_cache)

        # logger.warning("moving_scale")
        # logger.warning(moving_scale)

        # make an adjustment to moving scale to have it be useful for typical interm cache numbers
        # moving_scale = moving_scale / moving_scale # reset to 1 -- new scalev for second pass
        # #moving_scale = 

        # logger.warning("moving_scale")
        # logger.warning(moving_scale)

        interm_cache = downcast_to_static_fp8(intermediate_cache.view(M * topk, half_N), a2_scale)
        logger.warning("interm_cache")
        logger.warning(interm_cache)
        output_tensor = moe_gemm_a8w4(
            interm_cache,
            w2,
            None,
            w2_scale,
            a2_scale,
            None,
            w2_bias,
            routing_data,
            scatter_indx=scatter_indx,
            swizzle_mx_scale="CDNA4_SCALE", # simplicity
        )
        logger.warning("output")
        logger.warning(output_tensor)
        #logger.warning(output_tensor * static_scale)
    else:
        interm_cache, x_scale = mxfp4_quant(intermediate_cache.view(M * topk, half_N))
        output_tensor = moe_gemm_a4w4(
            interm_cache, # intermediate_cache.view(M * topk, half_N)
            w2, # w2
            x_scale, # x scales
            w2_scale, # w2 scale
            None, # x static scale
            None, # quant static scale
            w2_bias, # bias
            routing_data, # routing data
            scatter_indx=scatter_indx,
            swizzle_mx_scale="CDNA4_SCALE", # ?
        )

    # logger.warning("output")
    # logger.warning(output_tensor.shape)
    # logger.warning(output_tensor.dtype)
    # logger.warning(output_tensor)
    # logger.warning(output_tensor.view(M, K))

    output_tensor = output_tensor.view(M, K) 
    return output_tensor
