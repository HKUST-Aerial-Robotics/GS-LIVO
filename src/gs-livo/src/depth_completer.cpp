#include "depth_completer.h"

#include <stdexcept>

#ifdef WITH_TENSORRT
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <NvInfer.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace gslivo
{

namespace
{
class TrtLogger : public nvinfer1::ILogger
{
  void log(Severity severity, const char *msg) noexcept override
  {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << msg << std::endl;
  }
};

constexpr float kDepthScale = 200.f;  // SPNet works on depth / 200
}  // namespace

struct DepthCompleter::Impl
{
  TrtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
};

DepthCompleter::DepthCompleter(const std::string &engine_path) : impl_(new Impl)
{
  std::ifstream f(engine_path, std::ios::binary);
  if (!f) throw std::runtime_error("depth completion: cannot open engine " + engine_path);
  const std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
  if (!impl_->runtime) throw std::runtime_error("depth completion: createInferRuntime failed");
  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
  if (!impl_->engine) throw std::runtime_error("depth completion: cannot deserialize " + engine_path + " (TensorRT version / GPU mismatch?)");
  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) throw std::runtime_error("depth completion: createExecutionContext failed");

  for (const char *name : {"rgb", "depth", "mask", "pred"})
  {
    bool found = false;
    for (int i = 0; i < impl_->engine->getNbIOTensors(); i++) found |= std::string(impl_->engine->getIOTensorName(i)) == name;
    if (!found) throw std::runtime_error(std::string("depth completion: engine has no tensor '") + name + "'");
  }
  // The ONNX export has dynamic H, W; the engine is built for one shape (min = opt = max).
  const nvinfer1::Dims d = impl_->engine->getProfileShape("rgb", 0, nvinfer1::OptProfileSelector::kOPT);
  if (d.nbDims != 4) throw std::runtime_error("depth completion: unexpected rgb shape");
  height_ = static_cast<int>(d.d[2]);
  width_ = static_cast<int>(d.d[3]);
  const nvinfer1::Dims4 rgb_shape(1, 3, height_, width_), one_shape(1, 1, height_, width_);
  if (!impl_->context->setInputShape("rgb", rgb_shape) || !impl_->context->setInputShape("depth", one_shape) ||
      !impl_->context->setInputShape("mask", one_shape))
    throw std::runtime_error("depth completion: setInputShape failed");
}

DepthCompleter::~DepthCompleter()
{
  if (impl_)
  {
    impl_->context.reset();  // destroy in reverse order of creation
    impl_->engine.reset();
    impl_->runtime.reset();
  }
}

torch::Tensor DepthCompleter::complete(const torch::Tensor &rgb, const torch::Tensor &sparse)
{
  torch::NoGradGuard ng;
  // TensorRT synchronizes extra on the legacy default stream: use a pool stream, then join it
  const c10::cuda::CUDAStream caller = c10::cuda::getCurrentCUDAStream();
  const c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool();
  stream.synchronize();  // (cheap) nothing pending on the pool stream
  caller.synchronize();  // rgb / sparse were produced on the caller's stream
  c10::cuda::CUDAStreamGuard guard(stream);
  if (rgb.size(1) != height_ || rgb.size(2) != width_ || sparse.size(0) != height_ || sparse.size(1) != width_)
    throw std::runtime_error("depth completion: image size differs from the engine's");
  const auto cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  torch::Tensor in_rgb = rgb.to(cuda).reshape({1, 3, height_, width_}).contiguous();
  torch::Tensor in_depth = (sparse.to(cuda) / kDepthScale).reshape({1, 1, height_, width_}).contiguous();
  torch::Tensor in_mask = (sparse.to(cuda) > 0).to(torch::kFloat32).reshape({1, 1, height_, width_}).contiguous();
  torch::Tensor out = torch::empty({1, 1, height_, width_}, cuda);
  auto *ctx = impl_->context.get();
  if (!ctx->setTensorAddress("rgb", in_rgb.data_ptr()) || !ctx->setTensorAddress("depth", in_depth.data_ptr()) ||
      !ctx->setTensorAddress("mask", in_mask.data_ptr()) || !ctx->setTensorAddress("pred", out.data_ptr()))
    throw std::runtime_error("depth completion: setTensorAddress failed");
  if (!ctx->enqueueV3(stream.stream())) throw std::runtime_error("depth completion: inference failed");
  torch::Tensor depth = out.reshape({height_, width_}) * kDepthScale;
  stream.synchronize();  // the result is used on the caller's stream
  return depth;
}

}  // namespace gslivo

#else  // built without TensorRT (cmake -DGSLIVO_TENSORRT_DIR=... enables it)

namespace gslivo
{
struct DepthCompleter::Impl
{
};
DepthCompleter::DepthCompleter(const std::string &) { throw std::runtime_error("built without TensorRT (GSLIVO_TENSORRT_DIR)"); }
DepthCompleter::~DepthCompleter() = default;
torch::Tensor DepthCompleter::complete(const torch::Tensor &, const torch::Tensor &) { throw std::runtime_error("built without TensorRT"); }
}  // namespace gslivo

#endif
