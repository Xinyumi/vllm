#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_utils.h"
#include "reduction_utils.cuh"
#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
#endif

namespace vllm {

// from TRTLLM
template <typename Tf, typename T>
__inline__ __device__ Tf compute_layernorm(Tf val, float s_mean, float s_variance, const T* gamma, const T* beta, int i)
{
    Tf ret = (val - s_mean) * s_variance * cuda_cast<Tf>(gamma[i]);
    if (beta != nullptr)
    {
        ret = ret + cuda_cast<Tf>(beta[i]);
    }
    return ret;
}


// from TRTLLM
/* Computes the layernorm https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
 * normed_output <- ( (input - E[input]) / Sqrt(Var[input] + eps) ) * gamma + beta
 * input is [tokens, hidden_dim]. Mean and Variance are per-row (i.e. per-token)
 *
 * One CTA handles one row.
 *
 * with USE_DIFF_OF_SQUARES set to false:
 * First pass (loop) computes the mean.
 * Second computes the variance via Var[x] = E[(x - E[x])²].
 * Third pass computes and writes normed_output
 *
 * with USE_DIFF_OF_SQUARES set to true (may be faster but less accurate):
 * First pass (loop) computes the mean and variance via Var[x] = E[x²] - E[x]²
 * Second pass computes and writes normed_output
 *
 * use_shmem controls if we cache input values into shared memory
 *
 * Optional: with dynamic scaling, the last pass doesn't write immediately but finds the
 *           amax per row. A final pass scales to int8 accordingly, and writes output to
 *           normed_output_quant.
 */
template <typename T, typename scale_type, bool USE_DIFF_OF_SQUARES = false>
__global__ void generalLayerNorm(const T* input, const T* gamma, const T* beta, T* normed_output, const float eps,
    int tokens, int hidden_dim, const scale_type* scale_orig_quant_per_tensor, scale_type* scale_orig_quant_per_token,
    int8_t* normed_output_quant, bool use_shmem)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using int8_packed_t = typename packed_as<int8_t, num_elems_T>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);
    __shared__ float s_mean;
    __shared__ float s_variance;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    float mean = 0.0f;
    float variance = 0.0f;
    float local_sum = 0.0f;
    float local_var_sum = 0.0f;

    const int n_elems = hidden_dim / num_elems_T;
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const T val = input[bidx * n_elems + i];
        if (use_shmem)
        {
            shmem[i] = val;
        }

        const float_packed_t val_f = cuda_cast<float_packed_t>(val);
        local_sum += cuda_sum<float>(val_f);
        if (USE_DIFF_OF_SQUARES)
        {
            local_var_sum += cuda_sum<float>(val_f * val_f);
        }
    }

    if (USE_DIFF_OF_SQUARES)
    {
        float packed[2] = {local_sum, local_var_sum};
        blockReduceSumV2<float, 2>(packed);
        mean = packed[0];
        variance = packed[1];
    }
    else
    {
        mean = blockReduceSum(local_sum);
    }

    if (threadIdx.x == 0)
    {
        mean = mean / hidden_dim;
        s_mean = mean;
        if (USE_DIFF_OF_SQUARES)
        {
            variance = (variance / hidden_dim) - (mean * mean); // Var[x] = E[x²] - E[x]²
            s_variance = rsqrtf(variance + eps);
        }
    }
    __syncthreads();

    if (!USE_DIFF_OF_SQUARES)
    {
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const T val = use_shmem ? shmem[i] : input[bidx * n_elems + i];
            float_packed_t diff = cuda_cast<float_packed_t>(val) - s_mean;
            local_var_sum += cuda_sum<float>(diff * diff);
        }
        variance = blockReduceSum(local_var_sum);

        if (threadIdx.x == 0)
        {
            s_variance = rsqrtf(variance / hidden_dim + eps);
        }
        __syncthreads();
    }

    const bool with_per_token_scaling = scale_orig_quant_per_token != nullptr;
    const bool with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    const float_packed_t scale_orig_quant
        = cuda_cast<float_packed_t>(with_per_tensor_scaling ? __half2float(*scale_orig_quant_per_tensor) : 0.0f);
    T_scalar amax = 1e-6f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const int index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
        const T val = cuda_cast<T>(compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i));

        if (with_per_token_scaling)
        {
            amax = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
            if (use_shmem)
            {
                shmem[i] = val;
            }
        }
        else if (with_per_tensor_scaling)
        {
            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        }
        else
        {
            normed_output[index] = val;
        }
    }

    if (with_per_token_scaling)
    {
        float abs_max_f = blockAllReduceMax(cuda_cast<float>(amax));
        const float dynamic_per_token_scale = 127.f / abs_max_f;
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const int index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
            if (!use_shmem)
            {
                val_f = compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i);
            }

            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0)
        {
            scale_orig_quant_per_token[bidx] = abs_max_f / 127.f;
        }
    }
}


template <typename T, typename scale_type, bool USE_DIFF_OF_SQUARES = false>
__global__ void generalLayerNorm_fuse_sum(const T* input, const T* gamma, const T* beta, T* normed_output, const float eps,
    int tokens, int hidden_dim, scale_type* input_sum, const scale_type* scale_orig_quant_per_tensor, scale_type* scale_orig_quant_per_token,
    int8_t* normed_output_quant, bool use_shmem)
{
    constexpr auto num_elems_T = num_elems<T>::value;
    using int8_packed_t = typename packed_as<int8_t, num_elems_T>::type;
    using float_packed_t = typename packed_as<float, num_elems_T>::type;
    using T_scalar = typename packed_as<T, 1>::type;

    extern __shared__ __align__(sizeof(float)) char _shmem[];
    T* shmem = reinterpret_cast<T*>(_shmem);
    __shared__ float s_mean;
    __shared__ float s_variance;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    float mean = 0.0f;
    float variance = 0.0f;
    float local_sum = 0.0f;
    float local_var_sum = 0.0f;

    const int n_elems = hidden_dim / num_elems_T;
    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const T val = input[bidx * n_elems + i];
        if (use_shmem)
        {
            shmem[i] = val;
        }

        const float_packed_t val_f = cuda_cast<float_packed_t>(val);
        local_sum += cuda_sum<float>(val_f);
        if (USE_DIFF_OF_SQUARES)
        {
            local_var_sum += cuda_sum<float>(val_f * val_f);
        }
    }

    if (USE_DIFF_OF_SQUARES)
    {
        float packed[2] = {local_sum, local_var_sum};
        blockReduceSumV2<float, 2>(packed);
        mean = packed[0];
        variance = packed[1];
    }
    else
    {
        mean = blockReduceSum(local_sum);
    }

    if (threadIdx.x == 0)
    {
        mean = mean / hidden_dim;
        s_mean = mean;
        if (USE_DIFF_OF_SQUARES)
        {
            variance = (variance / hidden_dim) - (mean * mean); // Var[x] = E[x²] - E[x]²
            s_variance = rsqrtf(variance + eps);
        }
    }
    __syncthreads();

    if (!USE_DIFF_OF_SQUARES)
    {
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const T val = use_shmem ? shmem[i] : input[bidx * n_elems + i];
            float_packed_t diff = cuda_cast<float_packed_t>(val) - s_mean;
            local_var_sum += cuda_sum<float>(diff * diff);
        }
        variance = blockReduceSum(local_var_sum);

        if (threadIdx.x == 0)
        {
            s_variance = rsqrtf(variance / hidden_dim + eps);
        }
        __syncthreads();
    }

    const bool with_per_token_scaling = scale_orig_quant_per_token != nullptr;
    const bool with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
    const float_packed_t scale_orig_quant
        = cuda_cast<float_packed_t>(with_per_tensor_scaling ? __half2float(*scale_orig_quant_per_tensor) : 0.0f);
    T_scalar amax = 1e-6f;
    T_scalar sum = 0.0f;

    for (int i = tidx; i < n_elems; i += blockDim.x)
    {
        const int index = bidx * n_elems + i;
        const float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
        const T val = cuda_cast<T>(compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i));

        if (with_per_token_scaling)
        {
            amax = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
            sum += cuda_sum<float>(val);
            if (use_shmem)
            {
                shmem[i] = val;
            }
        }
        else if (with_per_tensor_scaling)
        {
            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
        }
        else
        {
            normed_output[index] = val;
        }
    }

    if (with_per_token_scaling)
    {
        float abs_max_f = blockAllReduceMax(cuda_cast<float>(amax));
        float sum_f = blockAllReduceSum(cuda_cast<float>(sum));
        const float dynamic_per_token_scale = 127.f / abs_max_f;
        for (int i = tidx; i < n_elems; i += blockDim.x)
        {
            const int index = bidx * n_elems + i;
            float_packed_t val_f = cuda_cast<float_packed_t>(use_shmem ? shmem[i] : input[index]);
            if (!use_shmem)
            {
                val_f = compute_layernorm(val_f, s_mean, s_variance, gamma, beta, i);
            }

            reinterpret_cast<int8_packed_t*>(normed_output_quant)[index]
                = cuda_cast<int8_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
        }
        if (tidx == 0)
        {
            scale_orig_quant_per_token[bidx] = abs_max_f / 127.f;
            input_sum[bidx] = sum_f;
        }
    }
}

// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

/* Converter structs for the conversion from torch types to HIP/CUDA types,
   and the associated type conversions within HIP/CUDA. These helpers need
   to be implemented for now because the relevant type conversion
   operators/constructors are not consistently implemented by HIP/CUDA, so
   a generic conversion via type casts cannot be implemented.

   Each struct should have the member static constexpr bool `exists`:
   If false, the optimized kernel is not used for the corresponding torch type.
   If true, the struct should be fully defined as shown in the examples below.
 */
template <typename torch_type>
struct _typeConvert {
  static constexpr bool exists = false;
};

#if defined(USE_ROCM) || (defined(CUDA_VERSION) && (CUDA_VERSION >= 12000))
// CUDA < 12.0 runs into issues with packed type conversion
template <>
struct _typeConvert<c10::Half> {
  static constexpr bool exists = true;
  using hip_type = __half;
  using packed_hip_type = __half2;

  __device__ static inline float convert(hip_type x) { return __half2float(x); }
  __device__ static inline float2 convert(packed_hip_type x) {
    return __half22float2(x);
  }
  __device__ static inline hip_type convert(float x) {
    return __float2half_rn(x);
  }
  __device__ static inline packed_hip_type convert(float2 x) {
    return __float22half2_rn(x);
  }
};

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// CUDA_ARCH < 800 does not have BF16 support
// TODO: Add in ROCm support once public headers handle bf16 maturely
template <>
struct _typeConvert<c10::BFloat16> {
  static constexpr bool exists = true;
  using hip_type = __nv_bfloat16;
  using packed_hip_type = __nv_bfloat162;

  __device__ static inline float convert(hip_type x) {
    return __bfloat162float(x);
  }
  __device__ static inline float2 convert(packed_hip_type x) {
    return __bfloat1622float2(x);
  }
  __device__ static inline hip_type convert(float x) {
    return __float2bfloat16(x);
  }
  __device__ static inline packed_hip_type convert(float2 x) {
    return __float22bfloat162_rn(x);
  }
};
  #endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#endif    // defined(USE_ROCM) || (defined(CUDA_VERSION) && (CUDA_VERSION >=
          // 12000))

/* Vector POD struct to generate vectorized and packed FP16/BF16 ops
   for appropriate specializations of fused_add_rms_norm_kernel.
   Only functions that are necessary in that kernel are implemented.
   Alignment to 16 bytes is required to use 128-bit global memory ops.
 */
template <typename scalar_t, int width>
struct alignas(16) _f16Vec {
  /* Not theoretically necessary that width is a power of 2 but should
     almost always be the case for optimization purposes */
  static_assert(width > 0 && (width & (width - 1)) == 0,
                "Width is not a positive power of 2!");
  using Converter = _typeConvert<scalar_t>;
  using T1 = typename Converter::hip_type;
  using T2 = typename Converter::packed_hip_type;
  T1 data[width];

  __device__ _f16Vec& operator+=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        T2 temp{data[i], data[i + 1]};
        temp += T2{other.data[i], other.data[i + 1]};
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) data[i] += other.data[i];
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const _f16Vec<scalar_t, width>& other) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        T2 temp{data[i], data[i + 1]};
        temp *= T2{other.data[i], other.data[i + 1]};
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) data[i] *= other.data[i];
    }
    return *this;
  }

  __device__ _f16Vec& operator*=(const float scale) {
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 temp_f = Converter::convert(T2{data[i], data[i + 1]});
        temp_f.x *= scale;
        temp_f.y *= scale;
        T2 temp = Converter::convert(temp_f);
        data[i] = temp.x;
        data[i + 1] = temp.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        float temp = Converter::convert(data[i]) * scale;
        data[i] = Converter::convert(temp);
      }
    }
    return *this;
  }

  __device__ float sum_squares() const {
    float result = 0.0f;
    if constexpr (width % 2 == 0) {
#pragma unroll
      for (int i = 0; i < width; i += 2) {
        float2 z = Converter::convert(T2{data[i], data[i + 1]});
        result += z.x * z.x + z.y * z.y;
      }
    } else {
#pragma unroll
      for (int i = 0; i < width; ++i) {
        float x = Converter::convert(data[i]);
        result += x * x;
      }
    }
    return result;
  }
};

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,         // [..., hidden_size]
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = input_v[id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }
  /* Keep the following if-else block in sync with the
     calculation of max_block_size in fused_add_rms_norm */
  if (num_tokens < 256) {
    variance = blockReduceSum<float, 1024>(variance);
  } else
    variance = blockReduceSum<float, 256>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[id] = temp;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,         // [..., hidden_size]
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * hidden_size + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }
  /* Keep the following if-else block in sync with the
     calculation of max_block_size in fused_add_rms_norm */
  if (num_tokens < 256) {
    variance = blockReduceSum<float, 1024>(variance);
  } else
    variance = blockReduceSum<float, 256>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

}  // namespace vllm

void rms_norm(torch::Tensor& out,     // [..., hidden_size]
              torch::Tensor& input,   // [..., hidden_size]
              torch::Tensor& weight,  // [hidden_size]
              double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
    vllm::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
  });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                       \
  VLLM_DISPATCH_FLOATING_TYPES(                                                \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {                  \
        vllm::fused_add_rms_norm_kernel<scalar_t, width>                       \
            <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),           \
                                         residual.data_ptr<scalar_t>(),        \
                                         weight.data_ptr<scalar_t>(), epsilon, \
                                         num_tokens, hidden_size);             \
      });

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}

void rms_norm_general(torch::Tensor &out,    // [..., hidden_size]
              torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &scaling, // [tokens] or [1]
              float epsilon,
              bool use_per_token_quant) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  block.x = 32 * ((block.x + 31) / 32);
  
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "generalLayerNorm", [&] {
    using T = typename FloatTypeConverter<scalar_t>::Type;
    if (use_per_token_quant) {
      // per-token
      vllm::generalLayerNorm<T, at::Half><<<grid, block, 0, stream>>>(
        reinterpret_cast<T*>(input.data_ptr<scalar_t>()), 
        reinterpret_cast<T*>(weight.data_ptr<scalar_t>()), nullptr,
        nullptr, epsilon, num_tokens, hidden_size, nullptr, scaling.data_ptr<at::Half>(),
        out.data_ptr<int8_t>(), false
      );
      // input, gamma, beta, normed_output, eps, tokens, hidden_dim, per_tensor_scale, per_token_scale
      // normed_output_quant, use_shmem
        // out.data_ptr<int8_t>(), input.data_ptr<scalar_t>(),
        // weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
    } else {
      // per-tensor
      vllm::generalLayerNorm<T, at::Half><<<grid, block, 0, stream>>>(
        reinterpret_cast<T*>(input.data_ptr<scalar_t>()), 
        reinterpret_cast<T*>(weight.data_ptr<scalar_t>()), nullptr,
        nullptr, epsilon, num_tokens, hidden_size, scaling.data_ptr<at::Half>(), nullptr,
        out.data_ptr<int8_t>(), false
      );
    }
  });
}

void rms_norm_general_fuse_sum(torch::Tensor &out,    // [..., hidden_size]
              torch::Tensor &input,  // [..., hidden_size]
              torch::Tensor &weight, // [hidden_size]
              torch::Tensor &input_sum, // [tokens] or [1]
              torch::Tensor &scaling, // [tokens] or [1]
              float epsilon,
              bool use_per_token_quant) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  block.x = 32 * ((block.x + 31) / 32);
  
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "generalLayerNorm_fuse_sum", [&] {
    using T = typename FloatTypeConverter<scalar_t>::Type;
    if (use_per_token_quant) {
      // per-token
      vllm::generalLayerNorm_fuse_sum<T, at::Half><<<grid, block, 0, stream>>>(
        reinterpret_cast<T*>(input.data_ptr<scalar_t>()), 
        reinterpret_cast<T*>(weight.data_ptr<scalar_t>()), nullptr,
        nullptr, epsilon, num_tokens, hidden_size, input_sum.data_ptr<at::Half>(), nullptr, scaling.data_ptr<at::Half>(),
        out.data_ptr<int8_t>(), false
      );
      // input, gamma, beta, normed_output, eps, tokens, hidden_dim, per_tensor_scale, per_token_scale
      // normed_output_quant, use_shmem
        // out.data_ptr<int8_t>(), input.data_ptr<scalar_t>(),
        // weight.data_ptr<scalar_t>(), epsilon, num_tokens, hidden_size);
    } else {
      // per-tensor
      // Rasing error here
      // Not implemented per-tensor input_sum
      assert(false);
      
      vllm::generalLayerNorm_fuse_sum<T, at::Half><<<grid, block, 0, stream>>>(
        reinterpret_cast<T*>(input.data_ptr<scalar_t>()), 
        reinterpret_cast<T*>(weight.data_ptr<scalar_t>()), nullptr,
        nullptr, epsilon, num_tokens, hidden_size, nullptr, scaling.data_ptr<at::Half>(), nullptr,
        out.data_ptr<int8_t>(), false
      );
    }
  });
}



void invoke_dequant_add_residual_rms_norm_quant(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &gamma,    // [hidden_size]
    at::Half scale,
    float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      residual.scalar_type(), "dequant_add_residual_rms_norm_quant_kernel",
      [&] {
          vllm::dequant_add_residual_rms_norm_quant_kernel<scalar_t, at::Half, false>
            <<<grid, block, 0, stream>>>(
                input.data_ptr<int32_t>(), residual.data_ptr<scalar_t>(),
                out.data_ptr<int8_t>(), gamma.data_ptr<scalar_t>(), epsilon,
                scale, num_tokens, hidden_size);
      });
}

void invoke_dequant_add_residual_rms_norm_quant(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &gamma,    // [hidden_size]
    torch::Tensor &scale,    // [num_tokens]
    float epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      residual.scalar_type(), "dequant_add_residual_rms_norm_quant_kernel",
      [&] {
          vllm::dequant_add_residual_rms_norm_quant_kernel<scalar_t, at::Half*, true>
            <<<grid, block, 0, stream>>>(
                input.data_ptr<int32_t>(), residual.data_ptr<scalar_t>(),
                out.data_ptr<int8_t>(), gamma.data_ptr<scalar_t>(), epsilon,
                scale.data_ptr<at::Half>(), num_tokens, hidden_size);
      });
}
