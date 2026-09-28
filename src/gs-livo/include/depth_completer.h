/*
Depth completion with SPNet (Wang et al., https://github.com/Wang-xjtu/SPNet) through a TensorRT engine,
used to initialize Gaussians where the LiDAR has no returns (as Gaussian-LIC2 does). The engine is built
from the ONNX export of Gaussian-LIC2 (ckpt/export_onnx_512_640.py): inputs rgb (1,3,H,W) in [0,1],
depth (1,1,H,W) in metres / 200, mask (1,1,H,W); output pred (1,1,H,W) in metres / 200.
*/
#ifndef DEPTH_COMPLETER_H_
#define DEPTH_COMPLETER_H_

#include <torch/torch.h>

#include <memory>
#include <string>

namespace gslivo
{

class DepthCompleter
{
public:
  explicit DepthCompleter(const std::string &engine_path);  // throws std::runtime_error
  ~DepthCompleter();
  DepthCompleter(const DepthCompleter &) = delete;
  DepthCompleter &operator=(const DepthCompleter &) = delete;

  int width() const { return width_; }
  int height() const { return height_; }

  // rgb: (3, H, W) float CUDA tensor in [0, 1]; sparse: (H, W) float CUDA tensor, metric depth, 0 = unknown.
  // Returns the completed metric depth (H, W) on the GPU. Runs on PyTorch's current CUDA stream.
  torch::Tensor complete(const torch::Tensor &rgb, const torch::Tensor &sparse);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int width_ = 0, height_ = 0;
};

}  // namespace gslivo

#endif  // DEPTH_COMPLETER_H_
