import torch
import triton
import triton.language as tl

# 1. Simple baseline (no explicit caching/tiling; can tune num_warps/stages)
@triton.jit
def fused_gemm_bias_gelu_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    Residual_ptr,
    stride_res_m, stride_res_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_RESIDUAL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        idx_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + idx_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (idx_k[None, :] < K), other=0.0)
        b = tl.load(B_ptr + idx_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(idx_k[:, None] < K) & (offs_n[None] < N), other=0.0)
        acc += tl.dot(a, b)
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    # 1.a (Optional) Fused residual add
    if USE_RESIDUAL:
        res = tl.load(Residual_ptr + offs_m[:, None] * stride_res_m + offs_n[None, :] * stride_res_n, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        acc = acc + res
    acc_cubed = acc * acc * acc
    scaled = 0.7978845608 * (acc + 0.044715 * acc_cubed)
    scaled = tl.maximum(tl.minimum(scaled, 10.0), -10.0)
    e2x = tl.exp(2 * scaled)
    tanh_approx = (e2x - 1) / (e2x + 1 + 1e-6)
    acc = 0.5 * acc * (1.0 + tanh_approx)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None] < N))

def fused_gemm_bias_gelu(
    A, B, Bias, Residual=None,
    block_m=32, block_n=32, block_k=32,
    num_warps=4, num_stages=2
):
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    K_, N = B.shape
    assert K == K_
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    use_residual = Residual is not None
    res_ptr = Residual if use_residual else A  # dummy, not used
    stride_res_m, stride_res_n = (Residual.stride(0), Residual.stride(1)) if use_residual else (0, 0)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    fused_gemm_bias_gelu_kernel[grid](
        A, B, Bias, C,
        M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        res_ptr, stride_res_m, stride_res_n,
        block_m, block_n, block_k,
        use_residual,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return C

# 2. Shared-memory tiled version (asymmetric tiling, fused residual)
@triton.jit
def fused_gemm_bias_gelu_sharedmem_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    Residual_ptr,
    stride_res_m, stride_res_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_RESIDUAL: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        idx_k = k + tl.arange(0, BLOCK_K)
        a_tile = tl.load(A_ptr + offs_m[:, None] * stride_am + idx_k[None, :] * stride_ak,
                         mask=(offs_m[:, None] < M) & (idx_k[None, :] < K), other=0.0)
        b_tile = tl.load(B_ptr + idx_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                         mask=(idx_k[:, None] < K) & (offs_n[None] < N), other=0.0)
        acc += tl.dot(a_tile, b_tile)
    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    # 2.a (Optional) Fused residual add
    if USE_RESIDUAL:
        res = tl.load(Residual_ptr + offs_m[:, None] * stride_res_m + offs_n[None, :] * stride_res_n, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        acc = acc + res
    acc_cubed = acc * acc * acc
    scaled = 0.7978845608 * (acc + 0.044715 * acc_cubed)
    scaled = tl.maximum(tl.minimum(scaled, 10.0), -10.0)
    e2x = tl.exp(2 * scaled)
    tanh_approx = (e2x - 1) / (e2x + 1 + 1e-6)
    acc = 0.5 * acc * (1.0 + tanh_approx)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None] < N))

def fused_gemm_bias_gelu_sharedmem(
    A, B, Bias, Residual=None,
    block_m=64, block_n=64, block_k=32,
    num_warps=4, num_stages=2
):
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    K_, N = B.shape
    assert K == K_
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    use_residual = Residual is not None
    res_ptr = Residual if use_residual else A
    stride_res_m, stride_res_n = (Residual.stride(0), Residual.stride(1)) if use_residual else (0, 0)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    fused_gemm_bias_gelu_sharedmem_kernel[grid](
        A, B, Bias, C,
        M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        res_ptr, stride_res_m, stride_res_n,
        block_m, block_n, block_k,
        use_residual,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return C

# 3. Padding/Alignment for memory-efficient loads and adaptive tiling
def aligned_size(x, blk):
    return ((x + blk - 1) // blk) * blk

def pad_matrices(A, B, bias, block_m, block_n):
    """Zero pad matrices so last dims are divisible by the chosen blocks."""
    M, K = A.shape
    K_, N = B.shape
    assert K == K_
    M_aligned = aligned_size(M, block_m)
    N_aligned = aligned_size(N, block_n)
    pad_A = torch.zeros((M_aligned, K), dtype=A.dtype, device=A.device)
    pad_B = torch.zeros((K, N_aligned), dtype=B.dtype, device=B.device)
    pad_bias = torch.zeros(N_aligned, dtype=bias.dtype, device=bias.device)
    pad_A[:M, :K] = A
    pad_B[:K, :N] = B
    pad_bias[:N] = bias
    return pad_A, pad_B, pad_bias, M, N
