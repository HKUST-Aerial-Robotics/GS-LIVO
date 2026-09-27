/*
Gaussian map of GS-LIVO (Hong et al., "GS-LIVO: Real-Time LiDAR, Inertial, and Visual Multi-sensor
Fused Odometry with Gaussian Mapping", arXiv:2501.08672):

  Sec. II-A  global Gaussian map: hash-indexed octree. Root voxels of edge v_s are found through a
             spatial hash of floor(p / v_s) (Eq. 1); each root voxel is a complete octree of depth
             `max_layer` whose leaves hold at most `leaf_capacity` Gaussians (LoD / downsampling).
  Sec. II-B  LiDAR-camera joint initialization: planar Gaussians attached to the surface, rotation
             from the LiDAR normal (Eq. 3), scale from the leaf level (Eq. 2), zero-order SH from the
             bilinearly interpolated image colour (Eq. 7); photometric optimization with Adam (Eq. 9).
  Sec. II-C  sliding window of Gaussians: the Gaussians of the root voxels inside the camera FoV live
             contiguously on the GPU (GPU Gaussian Buffer); the Spatial Hash Table maps voxel keys to
             the voxels and their rows. Per frame: (1) voxels leaving the FoV are written back to RAM
             and marked DELETE, (2) their rows are swapped with the rear rows and the buffer is
             truncated, (3) LiDAR hash keys of the new frame give the OVERLAP / ADD voxels, (4) ADD
             voxels are staged contiguously (CPU Gaussian Buffer) and appended with one host-device copy.
*/
#ifndef GS_MAP_H_
#define GS_MAP_H_

#include <torch/torch.h>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "voxel_map.h"  // VOXEL_LOCATION + std::hash

namespace gslivo
{

// One Gaussian as stored in RAM and, row by row, in the GPU buffer (14 floats, raw parameters).
struct Gaussian
{
  float xyz[3];
  float log_scale[3];
  float rot[4];  // quaternion w x y z (normalized when rendering)
  float sh_dc[3];  // zero-order SH coefficients, RGB
  float logit_opacity;
};
constexpr int kGaussianFloats = 14;
static_assert(sizeof(Gaussian) == kGaussianFloats * sizeof(float), "Gaussian must be tightly packed");

// Column blocks of a (N, 14) parameter tensor.
namespace col
{
constexpr int xyz = 0, log_scale = 3, rot = 6, sh_dc = 10, opacity = 13;
}

struct GSMapParams
{
  double root_voxel_size = 0.1;  // v_s in Eq. 1
  int max_layer = 2;  // octree depth; leaf edge = v_s / 2^max_layer
  int leaf_capacity = 1;  // Gaussians per leaf voxel
  double scale_factor = 0.5;  // tangential std-dev = scale_factor * leaf edge  (s_y, s_z in Eq. 2)
  double thickness = 0.002;  // s_delta in Eq. 2 [m], along the surface normal
  double init_opacity = 0.9;
  double min_depth = 0.3, max_depth = 50.0;  // camera depth range for initialization / the window
  int max_window = 100000;  // Gaussians in the sliding window (GPU)
  int iterations = 10;  // Adam iterations per frame
  double lr_xyz = 1e-4, lr_scale = 5e-3, lr_rot = 1e-3, lr_color = 5e-3, lr_opacity = 2.5e-2;
  double max_scale_ratio = 3.0;  // tangential scale clamp: max_scale_ratio * leaf edge
};

// Pinhole camera, world -> camera (the undistorted image).
struct GSCamera
{
  Eigen::Matrix3d Rcw = Eigen::Matrix3d::Identity();
  Eigen::Vector3d Pcw = Eigen::Vector3d::Zero();
  double fx = 1, fy = 1, cx = 0, cy = 0;
  int width = 0, height = 0;
};

// LiDAR point in the world frame; n_w = 0 when LIO has no plane for it.
struct InitPoint
{
  Eigen::Vector3d p_w;
  Eigen::Vector3d n_w;
};

// Render (N, 14) raw Gaussian parameters (on the GPU) from `cam`; returns the (3, H, W) RGB image.
// With alpha != nullptr, also renders the (H, W) accumulated opacity (second pass, no gradient).
torch::Tensor renderGaussians(const torch::Tensor &params, const GSCamera &cam, torch::Tensor *alpha = nullptr);

// 8UC3 image (RGB) -> (3, H, W) float CUDA tensor in [0, 1], and back (clamped).
torch::Tensor imageToTensor(const cv::Mat &rgb);
cv::Mat tensorToImage(const torch::Tensor &img);

struct GSVoxel
{
  Eigen::Vector3d origin;  // min corner
  std::vector<Gaussian> gs;  // RAM copy (authoritative while the voxel is outside the window)
  std::vector<uint8_t> leaf_count;  // Gaussians per leaf of the depth-max_layer octree
  std::vector<int> rows;  // GPU buffer row of every Gaussian while in the window
  bool in_window = false;
  int last_seen = -1;  // last frame whose LiDAR scan hit this voxel
  int n_fresh = 0;  // Gaussians created this frame while in the window (not on the GPU yet)
};

struct GSFrameStats
{
  int add_voxels = 0, delete_voxels = 0, evict_voxels = 0, new_gaussians = 0;
  int window_gaussians = 0, window_voxels = 0;
  long long map_gaussians = 0;
  size_t map_voxels = 0;
  double t_window_ms = 0, t_optimize_ms = 0, h2d_mb = 0, d2h_mb = 0;
  float loss = 0;
};

class GSMap
{
public:
  explicit GSMap(const GSMapParams &params);
  ~GSMap();

  // Sec. II-B.1 + II-C: initialize Gaussians from the LiDAR scan (RGB image, 8UC3) in empty leaves and
  // update the sliding window to the voxels inside `cam`'s field of view.
  void updateWindow(const std::vector<InitPoint> &scan, const cv::Mat &rgb, const GSCamera &cam, int frame);

  // Sec. II-B.2: `iterations` Adam steps of the photometric L1 loss (Eq. 9) on the window.
  float optimize(const cv::Mat &rgb, const GSCamera &cam, int iterations);

  // Render the window (no gradient); alpha: optional accumulated opacity.
  torch::Tensor render(const GSCamera &cam, torch::Tensor *alpha = nullptr);

  // All Gaussians (RAM + GPU window); in_window: 1 for window rows.
  void exportAll(std::vector<Gaussian> &out, std::vector<uint8_t> *in_window = nullptr);
  // Standard 3DGS PLY (SH degree 0).
  bool savePly(const std::string &path);

  long long numGaussians() const { return n_gaussians_; }
  int windowSize() const { return static_cast<int>(row_owner_.size()); }
  const GSMapParams &params() const { return p_; }
  GSFrameStats stats;

private:
  VOXEL_LOCATION keyOf(const Eigen::Vector3d &p) const;
  GSVoxel *voxelAt(const VOXEL_LOCATION &key, bool create);
  int leafIndex(const GSVoxel &v, const Eigen::Vector3d &p) const;
  bool inFov(const Eigen::Vector3d &p_w, const GSCamera &cam, double margin_px) const;
  Gaussian makeGaussian(const InitPoint &pt, const cv::Mat &rgb, const GSCamera &cam, const Eigen::Vector2d &uv) const;
  void removeRows(const std::vector<GSVoxel *> &voxels);  // write back + swap-with-rear + truncate
  void appendVoxels(const std::vector<GSVoxel *> &voxels);  // stage (CGB) + one H2D copy
  void setParams(torch::Tensor p);

  GSMapParams p_;
  double leaf_size_;
  int leaves_per_axis_;
  std::unordered_map<VOXEL_LOCATION, GSVoxel *> sht_;  // spatial hash table
  long long n_gaussians_ = 0;

  // GPU Gaussian Buffer: (N, 14) leaf tensor + Adam moments and per-row step counts
  torch::Tensor params_, adam_m_, adam_v_, adam_t_;
  torch::Tensor lr_;  // (14,) learning rate per column
  std::vector<GSVoxel *> row_owner_;  // voxel of every row
  std::vector<int> row_slot_;  // index into owner->gs
  std::vector<GSVoxel *> window_voxels_;
};

}  // namespace gslivo

#endif  // GS_MAP_H_
