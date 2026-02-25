import torch.distributed as dist
import os
import torch
from mmq.distributed import process_groups
from mmq_kernels.utils.enable_kernel_log import enable_kernel_log
from _C import TKParallelTensor, tk_all_to_all as _tk_all_to_all_cpp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common import (
    check_diff,
    benchmark_l2_clear,
    benchmark_no_l2_clear,
    clean_print
)

def single_all_to_all(input, scatter_idx, gather_idx, group):
    seq_world_size = dist.get_world_size(group)
    inp_shape = list(input.shape)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
    if scatter_idx < 1:
        input_t = input.reshape([seq_world_size, inp_shape[scatter_idx]] + inp_shape[scatter_idx + 1 :]).contiguous()
    else:
        # transpose groups of heads with the seq-len parallel dimension
        # so that we can scatter them!
        input_t = (
            input.reshape([-1, seq_world_size, inp_shape[scatter_idx]] + inp_shape[scatter_idx + 1 :])
            .transpose(0, 1)
            .contiguous()
        )

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    # if scattering the seq-dim, transpose the heads back to the original dimension
    if scatter_idx < 1:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[:gather_idx]
        + [
            inp_shape[gather_idx] * seq_world_size,
        ]
        + inp_shape[gather_idx + 1 :]
    ).contiguous()


def make_tk_input(input_, scatter_idx, gather_idx, group):
    ap_rank = dist.get_rank(group)
    global_rank = dist.get_rank()
    ap_world_size = dist.get_world_size(group)
    group_id = global_rank // ap_world_size

    if input_.shape[1] // ap_world_size < 16 or (input_.shape[1] // ap_world_size) % 16 != 0:
        input_shape = [input_.shape[0], input_.shape[1] * input_.shape[2]]
        output_shape = [input_.shape[0], input_.shape[1] * input_.shape[2]]
    else:
        input_shape = list(input_.shape)
        output_shape = list(input_.shape)
    output_shape[scatter_idx] = output_shape[scatter_idx] // ap_world_size
    output_shape[gather_idx] = output_shape[gather_idx] * ap_world_size

    input_tk = TKParallelTensor(
        input_shape,
        dtype=torch.bfloat16,
        local_rank=ap_rank,
        local_world_size=ap_world_size,
        multicast=False,
        group_id=group_id,
    )

    output_tk = TKParallelTensor(
        output_shape,
        dtype=torch.bfloat16,
        local_rank=ap_rank,
        local_world_size=ap_world_size,
        multicast=False,
        group_id=group_id,
    )
    output_tk.data_.zero_()
    
    barrier_tk = TKParallelTensor(
        (1, 1),
        dtype=torch.int,
        local_rank=ap_rank,
        local_world_size=ap_world_size,
        multicast=True,
        group_id=group_id
    )
    barrier_tk.data_.zero_()
    torch.distributed.barrier()
    return input_tk, output_tk, barrier_tk


@enable_kernel_log
def tk_all_to_all(
    input_: torch.Tensor,
    input_tk: TKParallelTensor,
    output_tk: TKParallelTensor,
    barrier_tk: TKParallelTensor,
    scatter_idx: int,
    gather_idx: int,
) -> torch.Tensor:
    input_tk.data_.copy_(input_.view(input_tk.data_.shape))
    _tk_all_to_all_cpp(output_tk, input_tk, barrier_tk, scatter_idx, gather_idx)
    return output_tk.data_.view(output_tk.data_.shape[0], -1, input_.shape[-1])

def all2all_test(n_tokens, n_heads, n_dim, scatter_idx, gather_idx):
    clean_print(f"===============================================================================", print_once=True)
    clean_print(f"<Ulysses | scatter_idx {scatter_idx} gather_idx {gather_idx} | {n_tokens}x{n_heads}x{n_dim}>", print_once=True)
    num_warmup_iters = 2
    num_iters = 5

    global_rank = dist.get_rank()
    ap_world_size = process_groups.ap_world_size()
    input_ = torch.ones(n_tokens // ap_world_size, n_heads, n_dim, device="cuda", dtype=torch.bfloat16) * global_rank
    input_tk, output_tk, barrier_tk = make_tk_input(input_, scatter_idx, gather_idx, process_groups.ap_process_group())

    output = tk_all_to_all(input_, input_tk, output_tk, barrier_tk, scatter_idx, gather_idx)
    output_ref = single_all_to_all(input_, scatter_idx, gather_idx, process_groups.ap_process_group())
    torch.testing.assert_close(output, output_ref)
    # clean_print(f"Max diff:  {((output - output_ref).abs().max().item()):.10f}")
    # clean_print(f"Mean diff: {((output - output_ref).abs().mean().item()):.10f}")
    
    tk_run = lambda: tk_all_to_all(input_, input_tk, output_tk, barrier_tk, scatter_idx, gather_idx)
    nccl_run = lambda: single_all_to_all(input_, scatter_idx, gather_idx, process_groups.ap_process_group())

    tk_avg_ms = benchmark_l2_clear(tk_run, num_warmup_iters, num_iters)
    nccl_avg_ms = benchmark_l2_clear(nccl_run, num_warmup_iters, num_iters)
    clean_print(f"TK: {tk_avg_ms:.3f} ms")
    clean_print(f"NCCL: {nccl_avg_ms:.3f} ms")

def base_test():
    n_tokens = 8192*2
    n_heads = 64
    n_dim = 256
    ap_group = process_groups.ap_process_group()
    rank = dist.get_rank()
    ap_rank = process_groups.ap_rank()
    ap_world_size = process_groups.ap_world_size()
    group_id = rank // ap_world_size
    print(f"[Rank {rank}] AP Rank: {ap_rank}, Group ID: {group_id}, AP World Size: {ap_world_size}")

    def test1():
        x = TKParallelTensor(
            (n_tokens, n_heads, n_dim),
            dtype=torch.bfloat16,
            local_rank=ap_rank,
            local_world_size=ap_world_size,
            multicast=False,
            group_id=group_id,
        )
        print(f"[Rank {rank}] device {x.data_.device.index} {x.get_info_string()}")


    def test2():
        barrier_g0 = TKParallelTensor(
            (1, 1),
            dtype=torch.int,
            local_rank=ap_rank,
            local_world_size=ap_world_size,
            multicast=True,
            group_id=group_id
        )
        print(f"[Rank {rank}] device {barrier_g0.data_.device.index} {barrier_g0.get_info_string()}")

    def test3():
        a = torch.randn(n_tokens, n_heads, n_dim, device="cuda", dtype=torch.bfloat16)

        tka = TKParallelTensor(
            tensor=a,
            local_rank=ap_rank,
            local_world_size=ap_world_size,
            multicast=False,
            group_id=group_id
        )
        print(f"[Rank {rank}] device {tka.data_.device.index} {tka.get_info_string()}")
    
    test1()
    # test2()
    # test3()

    

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    process_groups.init_process_groups(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        tp_size=1,
        pp_size=1,
        dp_size=8,
        virtual_pp_stage_size=1,
        attention_parallel_size=4,
        ring_attention_size=1,
        ep_size=1,
        vp_size=1,
        local_rank=local_rank,
        timeout=300,
    )
    dist.barrier()
    rank = dist.get_rank()
    ap_rank = process_groups.ap_rank()
    ap_world_size = process_groups.ap_world_size()
    print(f"[Rank {rank}] AP Rank: {ap_rank}, AP World Size: {ap_world_size}")
    dist.barrier()
    # base_test()
    for n_tokens, n_heads, n_dim in [
        (8192*2, 2, 256),
        (8192*2, 24, 256),
        (8192*2, 8, 128),
        (8192*2, 96, 128),
        (8192*4, 2, 256),
        (8192*4, 24, 256),
        (8192*4, 8, 128),
        (8192*4, 96, 128),
    ]:
        if n_heads < ap_world_size:
            continue
        all2all_test(n_tokens, n_heads, n_dim, 1, 0)
    torch.distributed.destroy_process_group()

# python -m torch.distributed.run --nproc_per_node=8 test.py