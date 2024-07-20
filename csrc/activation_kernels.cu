#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

//silu_and_mul_kernel
template <typename T> __device__ __forceinline__ T silu(const T &x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

  
template<typename scalar_t>
__global__ void silu_and_mul_kernel(
  scalar_t* __restrict__ out,               // [..., d]
  const scalar_t* __restrict__ input,       // [..., 2 * d]
  const int d) {

  const int token_idx = blockIdx.x;
  const int64_t token_idx_d = token_idx * int64_t(d);
  const int64_t token_idx_2d = token_idx_d * 2;
  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = __ldg(&input[token_idx_2d + idx]);
    const scalar_t y = __ldg(&input[token_idx_2d + d + idx]);
    out[token_idx_d + idx] = silu(x) * y;
  }
}

// Activation and gating kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = ACT_FN(x) * y;
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
}

}  // namespace vllm

// Launch activation and gating kernel.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL)                            \
  int d = input.size(-1) / 2;                                            \
  int64_t num_tokens = input.numel() / input.size(-1);                   \
  dim3 grid(num_tokens);                                                 \
  dim3 block(std::min(d, 1024));                                         \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));      \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();          \
  VLLM_DISPATCH_FLOATING_TYPES(                                          \
      input.scalar_type(), "act_and_mul_kernel", [&] {                   \
        vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>>             \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),       \
                                         input.data_ptr<scalar_t>(), d); \
      });


// from qerve ativation_kernels.cu
// dequant int32 input, apply silu and mul, then per token quant to int8
template <typename scale_type, bool use_per_token_quant>
__global__ void dequant_silu_and_mul_quant_kernel(
    int8_t *__restrict__ out,          // [..., d]
    const int32_t *__restrict__ input, // [..., 2 * d]
    const int d, const float scale_gate, const float scale_up,
    scale_type scale_out,                  // [num_tokens]
    float *__restrict__ tmp = nullptr // [num_tokens, d]
) {
  const int token_idx = blockIdx.x;
  if constexpr (use_per_token_quant) {
    float amax_val = 0.0f;
    const float zero = 0.0f;

    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const float x =
          (float)__ldg(&input[token_idx * 2 * d + idx]) * scale_gate;
      const float y =
          (float)__ldg(&input[token_idx * 2 * d + d + idx]) * scale_up;
      float t = silu(x) * y;
      tmp[token_idx * d + idx] = t;
      t = t > zero ? t : -t;
      if (t > amax_val)
        amax_val = t;
    }

    __shared__ float s_amax;
    const float block_amax_val = blockReduceMax(amax_val);
    if (threadIdx.x == 0) {
      s_amax = block_amax_val;
      scale_out[token_idx] = block_amax_val / 127.0f;
    }
    __syncthreads();

    float tmp_scale = 127.0f / s_amax;
    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
      out[token_idx * d + idx] =
          float_to_int8_rn(tmp_scale * tmp[token_idx * d + idx]);
    }
  } else {
    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const float x =
          (float)__ldg(&input[token_idx * 2 * d + idx]) * scale_gate;
      const float y =
          (float)__ldg(&input[token_idx * 2 * d + d + idx]) * scale_up;
      out[token_idx * d + idx] = float_to_int8_rn(silu(x) * y / scale_out);
    }
  }
}
} // namespace vllm

void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel);
}

void gelu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel);
}

void gelu_tanh_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel);
}


// from qerve activation_kernels.cu
void silu_and_mul_qerve(
  torch::Tensor& out,      // [..., d]
  torch::Tensor& input)    // [..., 2 * d]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_and_mul_kernel", [&] {
    vllm::silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);
  });
}

void invoke_dequant_silu_and_mul_quant(
    torch::Tensor &out,   // [..., d]
    torch::Tensor &input, // [..., 2 * d]
    const float scale_gate, const float scale_up, const float scale_out) {
  int64_t num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::dequant_silu_and_mul_quant_kernel<float, false><<<grid, block, 0, stream>>>(
      out.data_ptr<int8_t>(), input.data_ptr<int32_t>(), d, scale_gate,
      scale_up, scale_out);
}


void invoke_dequant_silu_and_mul_quant(
    torch::Tensor &out,   // [..., d]
    torch::Tensor &input, // [..., 2 * d]
    const float scale_gate, const float scale_up,
    torch::Tensor &scale_out, // [num_tokens]
    torch::Tensor &tmp // [..., d]
) {
  int64_t num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::dequant_silu_and_mul_quant_kernel<float*, true><<<grid, block, 0, stream>>>(
      out.data_ptr<int8_t>(), input.data_ptr<int32_t>(),
       d, scale_gate, scale_up, scale_out.data_ptr<float>(), tmp.data_ptr<float>());
}


namespace vllm {

// Element-wise activation kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void activation_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * d + idx]);
    out[token_idx * d + idx] = ACT_FN(x);
  }
}

}  // namespace vllm

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                       \
  int d = input.size(-1);                                                      \
  int64_t num_tokens = input.numel() / d;                                      \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "activation_kernel", [&] { \
    vllm::activation_kernel<scalar_t, KERNEL<scalar_t>>                        \
        <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),                 \
                                     input.data_ptr<scalar_t>(), d);           \
  });

namespace vllm {

template <typename T>
__device__ __forceinline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T& x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_quick_kernel(const T& x) {
  // x * sigmoid(1.702 * x)
  return (T)(((float)x) / (1.0f + expf(-1.702f * (float)x)));
}

}  // namespace vllm

void gelu_new(torch::Tensor& out,    // [..., d]
              torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(torch::Tensor& out,    // [..., d]
               torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}

void gelu_quick(torch::Tensor& out,    // [..., d]
                torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_quick_kernel);
}
