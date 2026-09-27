#include "gs_map.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>

#include "rasterizer.cuh"

namespace gslivo
{

namespace
{
constexpr float kSHC0 = 0.28209479177387814f;

double msSince(std::chrono::steady_clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

torch::Tensor eigenToTensor(const Eigen::Matrix4f &m)
{
  // Eigen is column-major: from_blob reads the transpose, which is the layout the 3DGS rasterizer expects.
  return torch::from_blob(const_cast<float *>(m.data()), {4, 4}, torch::kFloat32).clone().to(torch::kCUDA);
}

torch::Tensor cudaLong(const std::vector<int64_t> &v)
{
  return torch::from_blob(const_cast<int64_t *>(v.data()), {static_cast<int64_t>(v.size())}, torch::kLong).clone().to(torch::kCUDA);
}
}  // namespace

torch::Tensor renderGaussians(const torch::Tensor &params, const GSCamera &cam, torch::Tensor *alpha)
{
  const int W = cam.width, H = cam.height;
  const auto cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  if (params.size(0) == 0)
  {
    if (alpha) *alpha = torch::zeros({H, W}, cuda);
    return torch::zeros({3, H, W}, cuda);
  }
  Eigen::Matrix4f w2c = Eigen::Matrix4f::Identity();
  w2c.block<3, 3>(0, 0) = cam.Rcw.cast<float>();
  w2c.block<3, 1>(0, 3) = cam.Pcw.cast<float>();
  // Projection with principal point: ndc2Pix(P * p) == (fx x / z + cx, fy y / z + cy).
  const float znear = 0.01f, zfar = 100.f;
  Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
  P(0, 0) = 2.f * cam.fx / W;
  P(0, 2) = (2.f * cam.cx + 1.f) / W - 1.f;
  P(1, 1) = 2.f * cam.fy / H;
  P(1, 2) = (2.f * cam.cy + 1.f) / H - 1.f;
  P(2, 2) = zfar / (zfar - znear);
  P(2, 3) = -zfar * znear / (zfar - znear);
  P(3, 2) = 1.f;
  const Eigen::Matrix4f full = P * w2c;
  const Eigen::Vector3f campos = (-cam.Rcw.transpose() * cam.Pcw).cast<float>();

  GaussianRasterizationSettings settings = {
      .image_height = H,
      .image_width = W,
      .tanfovx = static_cast<float>(W / (2.0 * cam.fx)),
      .tanfovy = static_cast<float>(H / (2.0 * cam.fy)),
      .bg = torch::zeros({3}, cuda),
      .scale_modifier = 1.f,
      .viewmatrix = eigenToTensor(w2c),
      .projmatrix = eigenToTensor(full),
      .sh_degree = 0,
      .camera_center = torch::from_blob(const_cast<float *>(campos.data()), {3}, torch::kFloat32).clone().to(torch::kCUDA),
      .prefiltered = false};
  GaussianRasterizer rasterizer(settings);

  const int64_t n = params.size(0);
  torch::Tensor xyz = params.slice(1, col::xyz, col::xyz + 3).contiguous();
  torch::Tensor scales = torch::exp(params.slice(1, col::log_scale, col::log_scale + 3)).contiguous();
  torch::Tensor rots = torch::nn::functional::normalize(params.slice(1, col::rot, col::rot + 4),
                                                        torch::nn::functional::NormalizeFuncOptions().dim(1))
                           .contiguous();
  torch::Tensor sh = params.slice(1, col::sh_dc, col::sh_dc + 3).reshape({n, 1, 3}).contiguous();
  torch::Tensor opacity = torch::sigmoid(params.slice(1, col::opacity, col::opacity + 1)).contiguous();
  torch::Tensor means2d = torch::zeros_like(xyz);

  auto [image, radii] = rasterizer.forward(xyz, means2d, opacity, sh, torch::Tensor(), scales, rots, torch::Tensor());
  if (alpha)
  {
    torch::NoGradGuard ng;
    GaussianRasterizer acc(settings);
    auto [ones_img, r2] = acc.forward(xyz.detach(), torch::zeros_like(xyz), opacity.detach(), torch::Tensor(),
                                      torch::ones({n, 3}, cuda), scales.detach(), rots.detach(), torch::Tensor());
    *alpha = ones_img[0];
  }
  return image;
}

torch::Tensor imageToTensor(const cv::Mat &rgb)
{
  cv::Mat c = rgb.isContinuous() ? rgb : rgb.clone();
  return torch::from_blob(c.data, {c.rows, c.cols, 3}, torch::kUInt8)
      .to(torch::kCUDA)
      .permute({2, 0, 1})
      .to(torch::kFloat32)
      .div_(255.f)
      .contiguous();
}

cv::Mat tensorToImage(const torch::Tensor &img)
{
  torch::Tensor t = img.detach().clamp(0, 1).mul(255).round().to(torch::kUInt8).permute({1, 2, 0}).contiguous().cpu();
  cv::Mat out(static_cast<int>(t.size(0)), static_cast<int>(t.size(1)), CV_8UC3);
  std::memcpy(out.data, t.data_ptr<uint8_t>(), t.numel());
  return out;
}

GSMap::GSMap(const GSMapParams &params) : p_(params)
{
  leaves_per_axis_ = 1 << p_.max_layer;
  leaf_size_ = p_.root_voxel_size / leaves_per_axis_;
  const auto cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  std::vector<float> lr(kGaussianFloats);
  for (int i = 0; i < 3; i++) lr[col::xyz + i] = p_.lr_xyz, lr[col::log_scale + i] = p_.lr_scale, lr[col::sh_dc + i] = p_.lr_color;
  for (int i = 0; i < 4; i++) lr[col::rot + i] = p_.lr_rot;
  lr[col::opacity] = p_.lr_opacity;
  lr_ = torch::from_blob(lr.data(), {1, kGaussianFloats}, torch::kFloat32).clone().to(torch::kCUDA);
  setParams(torch::zeros({0, kGaussianFloats}, cuda));
  adam_m_ = torch::zeros({0, kGaussianFloats}, cuda);
  adam_v_ = torch::zeros({0, kGaussianFloats}, cuda);
  adam_t_ = torch::zeros({0, 1}, cuda);
}

GSMap::~GSMap()
{
  for (auto &kv : sht_) delete kv.second;
}

void GSMap::setParams(torch::Tensor p) { params_ = p.detach().requires_grad_(true); }

VOXEL_LOCATION GSMap::keyOf(const Eigen::Vector3d &p) const
{
  return VOXEL_LOCATION(static_cast<int64_t>(std::floor(p[0] / p_.root_voxel_size)),
                        static_cast<int64_t>(std::floor(p[1] / p_.root_voxel_size)),
                        static_cast<int64_t>(std::floor(p[2] / p_.root_voxel_size)));
}

GSVoxel *GSMap::voxelAt(const VOXEL_LOCATION &key, bool create)
{
  auto it = sht_.find(key);
  if (it != sht_.end()) return it->second;
  if (!create) return nullptr;
  GSVoxel *v = new GSVoxel;
  v->origin = Eigen::Vector3d(key.x, key.y, key.z) * p_.root_voxel_size;
  v->leaf_count.assign(leaves_per_axis_ * leaves_per_axis_ * leaves_per_axis_, 0);
  sht_[key] = v;
  return v;
}

int GSMap::leafIndex(const GSVoxel &v, const Eigen::Vector3d &p) const
{
  int idx[3];
  for (int j = 0; j < 3; j++)
    idx[j] = std::clamp(static_cast<int>(std::floor((p[j] - v.origin[j]) / leaf_size_)), 0, leaves_per_axis_ - 1);
  return (idx[0] * leaves_per_axis_ + idx[1]) * leaves_per_axis_ + idx[2];
}

bool GSMap::inFov(const Eigen::Vector3d &p_w, const GSCamera &cam, double margin_px) const
{
  const Eigen::Vector3d pc = cam.Rcw * p_w + cam.Pcw;
  if (pc[2] < p_.min_depth || pc[2] > p_.max_depth) return false;
  const double u = cam.fx * pc[0] / pc[2] + cam.cx, v = cam.fy * pc[1] / pc[2] + cam.cy;
  return u >= margin_px && v >= margin_px && u <= cam.width - 1 - margin_px && v <= cam.height - 1 - margin_px;
}

Gaussian GSMap::makeGaussian(const InitPoint &pt, const cv::Mat &rgb, const GSCamera &cam, const Eigen::Vector2d &uv) const
{
  Gaussian g;
  // Eq. 3: rotation from the surface normal (third axis); the view ray when LIO has no plane here.
  Eigen::Vector3d n = pt.n_w;
  if (n.norm() < 0.5) n = pt.p_w + cam.Rcw.transpose() * cam.Pcw;  // p_w - camera centre
  n.normalize();
  Eigen::Vector3d a = Eigen::Vector3d::UnitX().cross(n);
  if (a.norm() < 1e-3) a = Eigen::Vector3d::UnitY().cross(n);
  a.normalize();
  Eigen::Matrix3d R;
  R.col(0) = a;
  R.col(1) = n.cross(a);
  R.col(2) = n;
  const Eigen::Quaterniond q(R);
  // Eq. 2: planar Gaussian, level-based tangential extent, s_delta across the surface.
  const float s_t = static_cast<float>(p_.scale_factor * leaf_size_);
  for (int j = 0; j < 3; j++) g.xyz[j] = static_cast<float>(pt.p_w[j]);
  g.log_scale[0] = g.log_scale[1] = std::log(s_t);
  g.log_scale[2] = std::log(static_cast<float>(p_.thickness));
  g.rot[0] = q.w(), g.rot[1] = q.x(), g.rot[2] = q.y(), g.rot[3] = q.z();
  // Eq. 7: bilinear colour -> zero-order SH.
  const int u0 = static_cast<int>(std::floor(uv[0])), v0 = static_cast<int>(std::floor(uv[1]));
  const float du = static_cast<float>(uv[0] - u0), dv = static_cast<float>(uv[1] - v0);
  const cv::Vec3b &c00 = rgb.at<cv::Vec3b>(v0, u0), &c01 = rgb.at<cv::Vec3b>(v0, u0 + 1);
  const cv::Vec3b &c10 = rgb.at<cv::Vec3b>(v0 + 1, u0), &c11 = rgb.at<cv::Vec3b>(v0 + 1, u0 + 1);
  for (int j = 0; j < 3; j++)
  {
    const float c = ((1 - du) * (1 - dv) * c00[j] + du * (1 - dv) * c01[j] + (1 - du) * dv * c10[j] + du * dv * c11[j]) / 255.f;
    g.sh_dc[j] = (c - 0.5f) / kSHC0;
  }
  g.logit_opacity = static_cast<float>(std::log(p_.init_opacity / (1.0 - p_.init_opacity)));
  return g;
}

void GSMap::updateWindow(const std::vector<InitPoint> &scan, const cv::Mat &rgb, const GSCamera &cam, int frame)
{
  const auto t0 = std::chrono::steady_clock::now();
  stats.add_voxels = stats.delete_voxels = stats.evict_voxels = stats.new_gaussians = 0;
  stats.h2d_mb = stats.d2h_mb = 0;
  const Eigen::Vector3d centre = -cam.Rcw.transpose() * cam.Pcw;

  // Sec. II-B.1 + Step 3: spatial hash keys of the current LiDAR frame; new Gaussians in free leaves.
  std::vector<GSVoxel *> hit;
  std::vector<std::pair<GSVoxel *, int>> fresh;  // new Gaussians of voxels already in the window
  std::vector<GSVoxel *> touched;  // voxels with n_fresh > 0
  for (const InitPoint &pt : scan)
  {
    const Eigen::Vector3d pc = cam.Rcw * pt.p_w + cam.Pcw;
    if (pc[2] < p_.min_depth || pc[2] > p_.max_depth) continue;
    const Eigen::Vector2d uv(cam.fx * pc[0] / pc[2] + cam.cx, cam.fy * pc[1] / pc[2] + cam.cy);
    if (uv[0] < 0 || uv[1] < 0 || uv[0] >= cam.width - 1 || uv[1] >= cam.height - 1) continue;
    GSVoxel *v = voxelAt(keyOf(pt.p_w), true);
    if (v->last_seen != frame)
    {
      v->last_seen = frame;
      hit.push_back(v);
    }
    uint8_t &count = v->leaf_count[leafIndex(*v, pt.p_w)];
    if (count >= p_.leaf_capacity) continue;
    count++;
    v->gs.push_back(makeGaussian(pt, rgb, cam, uv));
    n_gaussians_++;
    stats.new_gaussians++;
    if (v->in_window)
    {
      if (v->n_fresh++ == 0) touched.push_back(v);
      fresh.emplace_back(v, static_cast<int>(v->gs.size()) - 1);
    }
  }

  // Step 1: window voxels whose centre left the FoV are marked DELETE.
  const double half = 0.5 * p_.root_voxel_size;
  std::vector<GSVoxel *> remove, keep;
  for (GSVoxel *v : window_voxels_)
  {
    if (v->last_seen == frame || inFov(v->origin + Eigen::Vector3d::Constant(half), cam, -0.1 * cam.width))
      keep.push_back(v);
    else
      remove.push_back(v);
  }
  stats.delete_voxels = static_cast<int>(remove.size());

  // Step 3: ADD = voxels hit by this scan that are not in the window, nearest first.
  std::vector<GSVoxel *> add;
  for (GSVoxel *v : hit)
    if (!v->in_window) add.push_back(v);
  std::sort(add.begin(), add.end(), [&](GSVoxel *a, GSVoxel *b) {
    return (a->origin - centre).squaredNorm() < (b->origin - centre).squaredNorm();
  });

  // Window capacity: evict the least recently observed voxels, then skip the farthest additions.
  // (v->rows of a window voxel still excludes this frame's fresh Gaussians.)
  for (GSVoxel *v : remove) v->in_window = false;  // mark; removeRows() does the bookkeeping
  long long rows = windowSize();
  for (GSVoxel *v : remove) rows -= static_cast<long long>(v->rows.size());
  for (GSVoxel *v : keep) rows += v->n_fresh;
  long long need = 0;
  for (GSVoxel *v : add) need += static_cast<long long>(v->gs.size());
  if (rows + need > p_.max_window)
  {
    std::sort(keep.begin(), keep.end(), [](GSVoxel *a, GSVoxel *b) { return a->last_seen < b->last_seen; });
    size_t k = 0;
    for (; k < keep.size() && rows + need > p_.max_window && keep[k]->last_seen < frame; k++)
    {
      rows -= static_cast<long long>(keep[k]->rows.size()) + keep[k]->n_fresh;
      keep[k]->in_window = false;
      remove.push_back(keep[k]);
      stats.evict_voxels++;
    }
    keep.erase(keep.begin(), keep.begin() + k);
    while (!add.empty() && rows + need > p_.max_window)
    {
      need -= static_cast<long long>(add.back()->gs.size());
      add.pop_back();
    }
  }
  // fresh Gaussians of voxels that leave the window stay in RAM only
  fresh.erase(std::remove_if(fresh.begin(), fresh.end(), [](const auto &f) { return !f.first->in_window; }), fresh.end());
  for (GSVoxel *v : touched) v->n_fresh = 0;
  for (auto &f : fresh) f.first->rows.push_back(-1);  // placeholder row, set by appendVoxels

  // Steps 1-2: write back + swap-with-rear compaction.
  removeRows(remove);
  // Step 4: append ADD voxels and the fresh Gaussians of OVERLAP voxels.
  std::vector<GSVoxel *> append = add;
  appendVoxels(append);
  if (!fresh.empty())
  {
    // fresh Gaussians of voxels that stay in the window
    std::vector<Gaussian> staged;
    staged.reserve(fresh.size());
    const int base = windowSize();
    for (size_t i = 0; i < fresh.size(); i++)
    {
      GSVoxel *v = fresh[i].first;
      const int slot = fresh[i].second;
      staged.push_back(v->gs[slot]);
      v->rows[slot] = base + static_cast<int>(i);
      row_owner_.push_back(v);
      row_slot_.push_back(slot);
    }
    torch::Tensor t = torch::from_blob(staged.data(), {static_cast<int64_t>(staged.size()), kGaussianFloats}, torch::kFloat32)
                          .to(torch::kCUDA);
    stats.h2d_mb += staged.size() * sizeof(Gaussian) / 1e6;
    torch::NoGradGuard ng;
    setParams(torch::cat({params_.detach(), t}));
    adam_m_ = torch::cat({adam_m_, torch::zeros_like(t)});
    adam_v_ = torch::cat({adam_v_, torch::zeros_like(t)});
    adam_t_ = torch::cat({adam_t_, torch::zeros({t.size(0), 1}, t.options())});
  }
  stats.add_voxels = static_cast<int>(add.size());
  stats.window_gaussians = windowSize();
  stats.window_voxels = static_cast<int>(window_voxels_.size());
  stats.map_gaussians = n_gaussians_;
  stats.map_voxels = sht_.size();
  stats.t_window_ms = msSince(t0);
}

void GSMap::removeRows(const std::vector<GSVoxel *> &voxels)
{
  if (voxels.empty()) return;
  torch::NoGradGuard ng;
  const int n = windowSize();
  std::vector<char> del(n, 0);
  std::vector<int64_t> rows;
  std::vector<std::pair<GSVoxel *, int>> dst;
  for (GSVoxel *v : voxels)
  {
    for (size_t s = 0; s < v->rows.size(); s++)
    {
      const int r = v->rows[s];
      if (r < 0) continue;  // fresh Gaussian, never uploaded
      del[r] = 1;
      rows.push_back(r);
      dst.emplace_back(v, static_cast<int>(s));
    }
  }
  // Step 1: optimized parameters back to the global map (RAM).
  if (!rows.empty())
  {
    torch::Tensor back = params_.index_select(0, cudaLong(rows)).contiguous().cpu();
    const Gaussian *src = reinterpret_cast<const Gaussian *>(back.data_ptr<float>());
    for (size_t i = 0; i < dst.size(); i++) dst[i].first->gs[dst[i].second] = src[i];
    stats.d2h_mb += back.numel() * sizeof(float) / 1e6;
  }
  // Step 2: swap the DELETE rows with rows from the rear, then truncate.
  const int m = n - static_cast<int>(rows.size());
  std::vector<int64_t> holes, movers;
  for (int r = 0; r < m; r++)
    if (del[r]) holes.push_back(r);
  for (int r = m; r < n; r++)
    if (!del[r]) movers.push_back(r);
  if (!holes.empty())
  {
    torch::Tensor h = cudaLong(holes), s = cudaLong(movers);
    torch::Tensor p = params_.detach();
    for (torch::Tensor *t : {&p, &adam_m_, &adam_v_, &adam_t_}) t->index_copy_(0, h, t->index_select(0, s));
    for (size_t i = 0; i < holes.size(); i++)
    {
      GSVoxel *owner = row_owner_[movers[i]];
      const int slot = row_slot_[movers[i]];
      row_owner_[holes[i]] = owner;
      row_slot_[holes[i]] = slot;
      owner->rows[slot] = static_cast<int>(holes[i]);
    }
  }
  setParams(params_.detach().narrow(0, 0, m));
  adam_m_ = adam_m_.narrow(0, 0, m);
  adam_v_ = adam_v_.narrow(0, 0, m);
  adam_t_ = adam_t_.narrow(0, 0, m);
  row_owner_.resize(m);
  row_slot_.resize(m);
  for (GSVoxel *v : voxels)
  {
    v->in_window = false;
    v->rows.clear();
  }
  window_voxels_.erase(std::remove_if(window_voxels_.begin(), window_voxels_.end(), [](GSVoxel *v) { return !v->in_window; }),
                       window_voxels_.end());
}

void GSMap::appendVoxels(const std::vector<GSVoxel *> &voxels)
{
  // Step 4: CPU Gaussian Buffer = the ADD voxels' Gaussians staged contiguously -> one H2D copy.
  size_t total = 0;
  for (GSVoxel *v : voxels) total += v->gs.size();
  if (total == 0)
  {
    for (GSVoxel *v : voxels)
      if (!v->in_window) v->in_window = true, window_voxels_.push_back(v);
    return;
  }
  std::vector<Gaussian> cgb;
  cgb.reserve(total);
  int row = windowSize();
  for (GSVoxel *v : voxels)
  {
    v->in_window = true;
    v->rows.resize(v->gs.size());
    for (size_t s = 0; s < v->gs.size(); s++)
    {
      cgb.push_back(v->gs[s]);
      v->rows[s] = row++;
      row_owner_.push_back(v);
      row_slot_.push_back(static_cast<int>(s));
    }
    window_voxels_.push_back(v);
  }
  torch::Tensor t = torch::from_blob(cgb.data(), {static_cast<int64_t>(total), kGaussianFloats}, torch::kFloat32).to(torch::kCUDA);
  stats.h2d_mb += total * sizeof(Gaussian) / 1e6;
  torch::NoGradGuard ng;
  setParams(torch::cat({params_.detach(), t}));
  adam_m_ = torch::cat({adam_m_, torch::zeros_like(t)});
  adam_v_ = torch::cat({adam_v_, torch::zeros_like(t)});
  adam_t_ = torch::cat({adam_t_, torch::zeros({t.size(0), 1}, t.options())});
}

float GSMap::optimize(const cv::Mat &rgb, const GSCamera &cam, int iterations)
{
  const auto t0 = std::chrono::steady_clock::now();
  stats.loss = 0;
  if (windowSize() == 0 || iterations <= 0) return 0;
  const torch::Tensor gt = imageToTensor(rgb);
  torch::Tensor alpha;
  {
    torch::NoGradGuard ng;
    renderGaussians(params_, cam, &alpha);
  }
  // only pixels the window covers carry photometric information about it
  const torch::Tensor mask = (alpha > 0.5f).to(torch::kFloat32).unsqueeze(0);
  const float n_px = mask.sum().item<float>();
  if (n_px < 100) return 0;
  const float b1 = 0.9f, b2 = 0.999f, eps = 1e-15f;
  const float max_log_scale = std::log(static_cast<float>(p_.max_scale_ratio * leaf_size_));
  float loss_value = 0;
  for (int it = 0; it < iterations; it++)
  {
    if (params_.grad().defined()) params_.mutable_grad().zero_();
    torch::Tensor img = renderGaussians(params_, cam);
    torch::Tensor loss = ((img - gt).abs() * mask).sum() / (3.f * n_px);
    loss.backward();
    torch::NoGradGuard ng;
    const torch::Tensor g = params_.grad();
    adam_t_.add_(1);
    adam_m_.mul_(b1).add_(g, 1 - b1);
    adam_v_.mul_(b2).addcmul_(g, g, 1 - b2);
    const torch::Tensor m_hat = adam_m_ / (1 - torch::pow(b1, adam_t_));
    const torch::Tensor v_hat = adam_v_ / (1 - torch::pow(b2, adam_t_));
    params_.sub_(lr_ * m_hat / (v_hat.sqrt() + eps));
    params_.slice(1, col::log_scale, col::log_scale + 3).clamp_max_(max_log_scale);
    if (it == iterations - 1) loss_value = loss.item<float>();
  }
  stats.loss = loss_value;
  stats.t_optimize_ms = msSince(t0);
  return loss_value;
}

torch::Tensor GSMap::render(const GSCamera &cam, torch::Tensor *alpha)
{
  torch::NoGradGuard ng;
  return renderGaussians(params_, cam, alpha);
}

void GSMap::exportAll(std::vector<Gaussian> &out, std::vector<uint8_t> *in_window)
{
  out.clear();
  out.reserve(static_cast<size_t>(n_gaussians_));
  for (auto &kv : sht_)
    if (!kv.second->in_window) out.insert(out.end(), kv.second->gs.begin(), kv.second->gs.end());
  const size_t n_ram = out.size();
  if (windowSize() > 0)
  {
    torch::NoGradGuard ng;
    torch::Tensor w = params_.detach().contiguous().cpu();
    const Gaussian *src = reinterpret_cast<const Gaussian *>(w.data_ptr<float>());
    out.insert(out.end(), src, src + windowSize());
  }
  if (in_window)
  {
    in_window->assign(out.size(), 1);
    std::fill(in_window->begin(), in_window->begin() + n_ram, 0);
  }
}

bool GSMap::savePly(const std::string &path)
{
  std::vector<Gaussian> gs;
  exportAll(gs);
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f << "ply\nformat binary_little_endian 1.0\nelement vertex " << gs.size() << "\n";
  for (const char *name : {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
                           "rot_0", "rot_1", "rot_2", "rot_3"})
    f << "property float " << name << "\n";
  f << "end_header\n";
  for (const Gaussian &g : gs)
  {
    const float row[17] = {g.xyz[0], g.xyz[1], g.xyz[2], 0, 0, 0, g.sh_dc[0], g.sh_dc[1], g.sh_dc[2], g.logit_opacity,
                           g.log_scale[0], g.log_scale[1], g.log_scale[2], g.rot[0], g.rot[1], g.rot[2], g.rot[3]};
    f.write(reinterpret_cast<const char *>(row), sizeof(row));
  }
  return static_cast<bool>(f);
}

}  // namespace gslivo
