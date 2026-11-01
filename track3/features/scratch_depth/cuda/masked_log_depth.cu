// Original two-pass fp32 reduction; no atomics or shared output races.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

// Every block owns one partial sum/count/error slot. All threads participate
// in each barrier, including threads without an input pixel.
__global__ void forward_partials(const float* prediction, const float* target,
                                 const bool* valid, bool* mask, int64_t size,
                                 float* sums, int64_t* counts, int64_t* errors) {
  __shared__ float block_sums[kThreads];
  __shared__ int64_t block_counts[kThreads];
  __shared__ int64_t block_errors[kThreads];
  float sum = 0.0f;
  int64_t count = 0;
  int64_t error = 0;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < size; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float label = target[index];
    const bool selected = valid[index] && isfinite(label) && label > 0.0f;
    mask[index] = selected;
    if (selected) {
      ++count;
      const float value = prediction[index];
      if (!isfinite(value) || value <= 0.0f) {
        ++error;
      } else {
        sum += fabsf(logf(value) - logf(label));
      }
    }
  }
  block_sums[threadIdx.x] = sum;
  block_counts[threadIdx.x] = count;
  block_errors[threadIdx.x] = error;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      block_sums[threadIdx.x] += block_sums[threadIdx.x + stride];
      block_counts[threadIdx.x] += block_counts[threadIdx.x + stride];
      block_errors[threadIdx.x] += block_errors[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sums[blockIdx.x] = block_sums[0];
    counts[blockIdx.x] = block_counts[0];
    errors[blockIdx.x] = block_errors[0];
  }
}

// One block consumes partials on the same stream after the first kernel.
__global__ void finish_forward(const float* sums, const int64_t* counts,
                               const int64_t* errors, int blocks, float* loss,
                               int64_t* count_out, int64_t* error_out) {
  __shared__ float block_sums[kThreads];
  __shared__ int64_t block_counts[kThreads];
  __shared__ int64_t block_errors[kThreads];
  float sum = 0.0f;
  int64_t count = 0;
  int64_t error = 0;
  for (int index = threadIdx.x; index < blocks; index += blockDim.x) {
    sum += sums[index];
    count += counts[index];
    error += errors[index];
  }
  block_sums[threadIdx.x] = sum;
  block_counts[threadIdx.x] = count;
  block_errors[threadIdx.x] = error;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      block_sums[threadIdx.x] += block_sums[threadIdx.x + stride];
      block_counts[threadIdx.x] += block_counts[threadIdx.x + stride];
      block_errors[threadIdx.x] += block_errors[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int64_t total = block_counts[0];
    *count_out = total;
    *error_out = block_errors[0];
    *loss = total > 0 ? block_sums[0] / static_cast<float>(total) : 0.0f;
  }
}

__global__ void backward_pixels(const float* upstream, const float* prediction,
                                const float* target, const bool* mask,
                                const int64_t* count, int64_t size, float* output) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < size; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    // No log/division/upstream multiplication on excluded pixels, even with
    // NaN/Inf inputs or upstream gradient. Zero is written to every such pixel.
    float result = 0.0f;
    if (mask[index] && *count > 0) {
      const float difference = logf(prediction[index]) - logf(target[index]);
      const float sign = difference > 0.0f ? 1.0f : (difference < 0.0f ? -1.0f : 0.0f);
      if (sign != 0.0f) {
        result = (upstream[0] / static_cast<float>(*count)) * sign / prediction[index];
      }
    }
    output[index] = result;
  }
}

int blocks_for(int64_t size) {
  return static_cast<int>(std::min<int64_t>((size - 1) / kThreads + 1, kMaxBlocks));
}
}  // namespace

std::vector<torch::Tensor> masked_log_depth_forward_cuda(
    const torch::Tensor& prediction, const torch::Tensor& target,
    const torch::Tensor& valid) {
  const c10::cuda::CUDAGuard guard(prediction.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(prediction.get_device());
  const int blocks = blocks_for(prediction.numel());
  const auto float_options = prediction.options().requires_grad(false);
  const auto integer_options = float_options.dtype(at::kLong);
  auto sums = torch::empty({blocks}, float_options);
  auto counts = torch::empty({blocks}, integer_options);
  auto errors = torch::empty({blocks}, integer_options);
  auto mask = torch::empty(prediction.sizes(), float_options.dtype(at::kBool));
  auto loss = torch::empty({}, float_options);
  auto count = torch::empty({}, integer_options);
  auto error = torch::empty({}, integer_options);
  forward_partials<<<blocks, kThreads, 0, stream.stream()>>>(
      prediction.data_ptr<float>(), target.data_ptr<float>(), valid.data_ptr<bool>(),
      mask.data_ptr<bool>(), prediction.numel(), sums.data_ptr<float>(),
      counts.data_ptr<int64_t>(), errors.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  finish_forward<<<1, kThreads, 0, stream.stream()>>>(
      sums.data_ptr<float>(), counts.data_ptr<int64_t>(), errors.data_ptr<int64_t>(),
      blocks, loss.data_ptr<float>(), count.data_ptr<int64_t>(), error.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // Host scalar reads synchronize this device's current stream and surface
  // errors before returning any loss or saved backward state to Python.
  TORCH_CHECK(count.item<int64_t>() > 0, "Training batch contains no usable depth labels");
  TORCH_CHECK(error.item<int64_t>() == 0,
              "Selected depth predictions must be finite and strictly positive");
  return {loss, mask, count};
}

torch::Tensor masked_log_depth_backward_cuda(
    const torch::Tensor& gradient, const torch::Tensor& prediction,
    const torch::Tensor& target, const torch::Tensor& mask,
    const torch::Tensor& count) {
  const c10::cuda::CUDAGuard guard(prediction.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(prediction.get_device());
  auto output = torch::empty(prediction.sizes(), prediction.options().requires_grad(false));
  backward_pixels<<<blocks_for(prediction.numel()), kThreads, 0, stream.stream()>>>(
      gradient.data_ptr<float>(), prediction.data_ptr<float>(), target.data_ptr<float>(),
      mask.data_ptr<bool>(), count.data_ptr<int64_t>(), prediction.numel(), output.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
