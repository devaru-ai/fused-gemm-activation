import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('../kernels')

from triton_fused_gemm import (
    fused_gemm_bias_gelu,
    fused_gemm_bias_gelu_sharedmem,
    pad_matrices,
)

def torch_gemm_bias_gelu(A, B, bias):
    x = torch.matmul(A, B)
    x = x + bias
    return torch.nn.functional.gelu(x)

def benchmark_kernel(fn, A, B, bias, n_iter=10, Residual=None):
    if Residual is not None:
        y = fn(A, B, bias, Residual)
    else:
        y = fn(A, B, bias)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        start = time.time()
        if Residual is not None:
            y = fn(A, B, bias, Residual)
        else:
            y = fn(A, B, bias)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    return np.mean(times), y

def autotune_block_and_occupancy(A, B, bias, Residual=None, block_sizes=[16,32,64,128], warps=[2,4,8], stages=[2,3], n_iter=5):
    best_time = float('inf')
    best_cfg = None
    best_out = None
    times_dict = {}
    for blk in block_sizes:
        padA, padB, padBias, M, N = pad_matrices(A, B, bias, blk, blk)
        for nw in warps:
            for ns in stages:
                try:
                    fn = lambda a, b, bias, Residual=None: fused_gemm_bias_gelu_sharedmem(a, b, bias, Residual=Residual, block_m=blk, block_n=blk, num_warps=nw, num_stages=ns)
                    t, out = benchmark_kernel(fn, padA, padB, padBias, n_iter=n_iter, Residual=Residual)
                    times_dict[(blk, nw, ns)] = t
                    print(f"Block {blk}, Warps {nw}, Stages {ns}: {t*1000:.2f} ms")
                    if t < best_time:
                        best_time = t
                        best_cfg = (blk, nw, ns)
                        best_out = out[:M, :N]
                except Exception as e:
                    print(f"Block {blk}, Warps {nw}, Stages {ns}: failed ({e})")
    return best_cfg, best_time, best_out, times_dict

def compute_flops(M, N, K):
    # For FP32 GEMM, 2 * M * N * K ops
    return 2 * M * N * K

def compute_tflops(time_s, flops):
    return flops / time_s / 1e12

def print_benchmark_results(names, times, tflops, baseline_time):
    for i in range(len(names)):
        print(f"{names[i]:35s} {times[i]*1000:.2f} ms | {tflops[i]:.2f} TFLOPS | Speedup vs cuBLAS: {baseline_time/times[i]:.2f}x")

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda"

    # === Main test shape (square) ===
    M, K, N = 1024, 1024, 1024
    A = torch.randn((M, K), device=device, dtype=torch.float32) * 0.01
    B = torch.randn((K, N), device=device, dtype=torch.float32) * 0.01
    bias = torch.randn((N,), device=device, dtype=torch.float32) * 0.01
    Residual = torch.randn((M, N), device=device, dtype=torch.float32) * 0.01  # for fused residual

    FLOPs = compute_flops(M, N, K)

    # Pad for shared-mem kernels (square)
    padBlkM, padBlkN = 64, 64
    padA, padB, padBias, Mpad, Npad = pad_matrices(A, B, bias, padBlkM, padBlkN)
    padA_resid, padB_resid, padBias_resid, _, _ = pad_matrices(A, B, bias, padBlkM, padBlkN)

    # === Baseline & tuned runs ===
    t_cublas, out_cublas = benchmark_kernel(torch_gemm_bias_gelu, A, B, bias)
    t_fused, out_fused = benchmark_kernel(lambda a, b, bias: fused_gemm_bias_gelu(a, b, bias, block_m=32, block_n=32), A, B, bias)
    t_shared, out_shared = benchmark_kernel(lambda a, b, bias: fused_gemm_bias_gelu_sharedmem(a, b, bias, block_m=padBlkM, block_n=padBlkN), padA, padB, padBias)
    t_fused_resid, out_fused_resid = benchmark_kernel(
        lambda a, b, bias, resid: fused_gemm_bias_gelu_sharedmem(a, b, bias, Residual=resid, block_m=padBlkM, block_n=padBlkN),
        padA, padB, padBias, Residual=Residual)
    padBlkM2, padBlkN2 = 128, 32
    padA2, padB2, padBias2, _, _ = pad_matrices(A, B, bias, padBlkM2, padBlkN2)
    t_asymm, out_asymm = benchmark_kernel(
        lambda a, b, bias: fused_gemm_bias_gelu_sharedmem(a, b, bias, block_m=padBlkM2, block_n=padBlkN2),
        padA2, padB2, padBias2)

    # Autotuning (on square, no residual)
    best_cfg, best_time, best_out, times_dict = autotune_block_and_occupancy(
        A, B, bias, block_sizes=[16,32,64], warps=[2,4,8], stages=[2,3], n_iter=3)

    blk_str = f"blk={best_cfg[0]},w={best_cfg[1]},s={best_cfg[2]}"

    # === Compute Throughput (TFLOPS) ===
    times = [t_cublas, t_fused, t_shared, t_fused_resid, t_asymm, best_time]
    tflops = [compute_tflops(t, FLOPs) for t in times]
    labels = [
        "cuBLAS FP32",
        "Triton Fused FP32",
        "Triton SharedMem FP32 (blk64x64)",
        "Fused Postproc (residual)",
        "Asymm Tiling (blk128x32)",
        f"Triton SharedMem FP32 (auto: {blk_str})"
    ]
    colors = ['#2ca02c','#ff7f0e','#1f77b4','#d62728', '#17becf','#9467bd']

    # Print Latency, TFLOPS, Speedup vs cuBLAS baseline
    print("\n=== Benchmark Results ===")
    print_benchmark_results(labels, times, tflops, t_cublas)

    print(f"\nmax diff: shared-cublas {(out_shared[:M,:N] - out_cublas).abs().max().item():.2e}")
    print(f"max diff (fused residual): {(out_fused_resid[:M,:N] - out_cublas).abs().max().item():.2e}")
    print(f"max diff (asymm): {(out_asymm[:M,:N] - out_cublas).abs().max().item():.2e}")
    print(f"max diff (autotune-cublas): {(best_out - out_cublas).abs().max().item():.2e}")

    # Bar chart: Latency and TFLOPS
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax2 = ax1.twiny()

    y_pos = np.arange(len(times))
    ax1.barh(y_pos, [t*1000 for t in times], color=colors, alpha=0.9)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=12)
    ax1.set_xlabel('Latency (ms)', fontsize=13)
    ax1.set_title('cuBLAS vs Triton: Latency and TFLOPS', fontsize=14)

    # TFLOPS bar overlay
    ax2.barh(y_pos, tflops, color='grey', alpha=0.3)
    ax2.set_xlabel('Throughput (TFLOPS)', fontsize=13)
    ax2.set_xlim(0, max(tflops)*1.2)

    # Annotate bars (latency and tflops)
    for i, (v, t) in enumerate(zip([t*1000 for t in times], tflops)):
        ax1.text(v + 5, i, f"{v:.2f} ms", va='center', fontsize=12)
        ax2.text(t + 0.05, i, f"{t:.2f} TFLOPS", va='center', fontsize=12, color='black')

    plt.tight_layout()
    plt.subplots_adjust(left=0.35)  
    plt.savefig("benchmark_all_latency_tflops.png", dpi=120)
    plt.show()

    # Best run per block size (autotune, for reference)
    block_sizes_unique = sorted(set(k[0] for k in times_dict))
    best_runtimes = [min([times_dict[k] for k in times_dict if k[0]==blk])*1000 for blk in block_sizes_unique]
    best_tflops = [compute_tflops(min([times_dict[k] for k in times_dict if k[0]==blk]), FLOPs) for blk in block_sizes_unique]
    plt.figure(figsize=(7,4))
    plt.plot(block_sizes_unique, best_runtimes, marker='o', linewidth=2, color='#1f77b4', label="Latency (ms)")
    plt.plot(block_sizes_unique, best_tflops, marker='^', linestyle='--', linewidth=2, color='#2ca02c', label="TFLOPS")
    plt.xlabel("Block Size", fontsize=12)
    plt.ylabel("Latency (ms) / TFLOPS", fontsize=12)
    plt.grid(True, linestyle='dashed', alpha=0.7)
    plt.title("Autotune: Best Latency and TFLOPS per Block Size", fontsize=13)
    for x, y, tf in zip(block_sizes_unique, best_runtimes, best_tflops):
        plt.text(x, y+1, f"{y:.2f} ms", ha='center', fontsize=10)
        plt.text(x, tf+0.05, f"{tf:.2f} TFLOPS", ha='center', fontsize=10, color='#2ca02c')
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmarks/autotune_blk_lineplot_latency_tflops.png", dpi=120)
    plt.show()
