// Original scratch-depth extension binding; local sources only.
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> masked_log_depth_forward_cuda(
    const torch::Tensor& prediction, const torch::Tensor& target,
    const torch::Tensor& valid);
torch::Tensor masked_log_depth_backward_cuda(
    const torch::Tensor& gradient, const torch::Tensor& prediction,
    const torch::Tensor& target, const torch::Tensor& mask,
    const torch::Tensor& count);

namespace {
void require_tensor(const torch::Tensor& tensor, at::ScalarType dtype,
                    const char* name) {
  TORCH_CHECK(tensor.defined() && tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.layout() == at::kStrided && tensor.is_contiguous(),
              name, " must be contiguous and strided");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype");
}

void require_inputs(const torch::Tensor& prediction, const torch::Tensor& target,
                    const torch::Tensor& mask) {
  require_tensor(prediction, at::kFloat, "prediction");
  require_tensor(target, at::kFloat, "target");
  require_tensor(mask, at::kBool, "mask");
  TORCH_CHECK(prediction.sizes() == target.sizes() && prediction.sizes() == mask.sizes(),
              "prediction, target and mask must have identical shapes");
  TORCH_CHECK(prediction.device() == target.device() && prediction.device() == mask.device(),
              "prediction, target and mask must use the same CUDA device");
  TORCH_CHECK(prediction.numel() > 0, "Training batch contains no usable depth labels");
}

std::vector<torch::Tensor> forward(const torch::Tensor& prediction,
                                 const torch::Tensor& target,
                                 const torch::Tensor& valid) {
  require_inputs(prediction, target, valid);
  return masked_log_depth_forward_cuda(prediction, target, valid);
}

torch::Tensor backward(const torch::Tensor& gradient,
                       const torch::Tensor& prediction, const torch::Tensor& target,
                       const torch::Tensor& mask, const torch::Tensor& count) {
  require_inputs(prediction, target, mask);
  require_tensor(gradient, at::kFloat, "gradient");
  require_tensor(count, at::kLong, "count");
  TORCH_CHECK(gradient.dim() == 0 && count.dim() == 0,
              "gradient and saved count must be scalars");
  TORCH_CHECK(gradient.device() == prediction.device() && count.device() == prediction.device(),
              "gradient and saved count must use the prediction CUDA device");
  return masked_log_depth_backward_cuda(gradient, prediction, target, mask, count);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &forward, "Masked log-depth MAE forward (CUDA)");
  module.def("backward", &backward, "Masked log-depth MAE prediction gradient (CUDA)");
}
