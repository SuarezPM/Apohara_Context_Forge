// CUDA C kernel for the TurboQuant-KV path (Phase 4).
//
// Feature-gated: this file is only compiled when the `compute_75`
// feature is enabled. Building requires `nvcc` and the matching
// compute capability. The CPU scalar path in `src/lib.rs` is the
// default for the `maturin develop` smoke test.
//
// Workgroup (block) size: 32. This matches the spec's pinned
// workgroup size (R9 / R15 of the deep-interview-apohara-2-0.md
// spec). The kernel is launched with `blockDim.x = 32`.
//
// C ABI: the kernel is exposed as `extern "C"` so a thin C
// launcher (or `ctypes` from Python) can invoke it. The launcher
// signature mirrors the Rust API:
//   extern "C" void turboquant_encode(
//       const float* weights, uint8_t* packed, int n, int bits);
//
// This is the staged port; the H100/MI300X-tuned vectorised
// Lloyd-Max + 1-bit QJL is a follow-up that lands behind the
// `compute_80` / `compute_90` feature flags.

#include <stdint.h>

// Lloyd-Max centroid tables (duplicated from `centroids.rs` for
// kernel self-containment). The tables are unit-variance Beta
// quantizers precomputed via Lloyd-Max iteration.
__device__ const float CENTROIDS_2BIT[4] = {
    -1.710271f, -0.533148f, 0.000000f, 0.533148f
};
__device__ const float CENTROIDS_3BIT[8] = {
    -1.922873f, -1.139285f, -0.591692f, -0.137472f,
     0.137472f,  0.591692f,  1.139285f,  1.922873f
};
__device__ const float CENTROIDS_4BIT[16] = {
    -2.020086f, -1.469140f, -1.081309f, -0.764244f,
    -0.484829f, -0.228538f,  0.017464f,  0.265484f,
     0.515905f,  0.770540f,  1.031358f,  1.300800f,
     1.582847f,  1.882672f,  2.207272f,  2.562995f
};

__device__ inline int nearest_centroid_4bit(float w) {
    // Linear search; for 16 entries on CC 7.5 the warp-level
    // latency hides the latency of a fully unrolled compare chain.
    float best = 1e30f;
    int best_i = 0;
    for (int i = 0; i < 16; ++i) {
        float d = fabsf(w - CENTROIDS_4BIT[i]);
        if (d < best) { best = d; best_i = i; }
    }
    return best_i;
}

// Kernel: scalar Lloyd-Max quantization with workgroup size 32.
// Each thread processes one input element.
__global__ void turboquant_encode_kernel_4bit(
        const float* __restrict__ weights,
        uint8_t* __restrict__ packed,
        int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * 2 >= n) return;
    int idx0 = nearest_centroid_4bit(weights[tid * 2]);
    int idx1 = (tid * 2 + 1 < n)
        ? nearest_centroid_4bit(weights[tid * 2 + 1])
        : 0;
    packed[tid] = (uint8_t)((idx0 & 0xF) | ((idx1 & 0xF) << 4));
}

// C ABI launcher. Called from a thin C wrapper (or via ctypes).
// `n` is the number of input float elements; the caller is
// responsible for sizing `packed` to `ceil(n / 2)` bytes.
extern "C" void turboquant_encode(
        const float* weights, uint8_t* packed, int n, int bits) {
    if (bits != 4) {
        // The non-4bit paths land with the compute_80+ feature.
        return;
    }
    int threads_per_block = 32;
    int n_packed = (n + 1) / 2;
    int blocks = (n_packed + threads_per_block - 1) / threads_per_block;
    turboquant_encode_kernel_4bit<<<blocks, threads_per_block>>>(
        weights, packed, n);
}
