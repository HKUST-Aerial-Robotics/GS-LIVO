#ifndef GS_OCTREE_HPP_
#define GS_OCTREE_HPP_

#include "common_lib.h"
#include <Eigen/Dense>
#include <fstream>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <pcl/common/io.h>
#include <ros/ros.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#include "debug_utils.cuh"
#include "gaussian.cuh"
#include "loss_monitor.cuh"
#include "loss_utils.cuh"
#include "parameters.cuh"
#include "render_utils.cuh"
#include "scene.cuh"

#define GSMAP_HASH_P 116101
#define GSMAP_MAX_N 10000000000



// Forward declaration
class GSVoxelOctree;

class GSPointList
{
public:
  std::vector<GS_point*> gs_point_list;
  int count;
  GSPointList(int num) : count(num) {}
};


class GSVoxelOctree
{
public:
  GSVoxelOctree() = default;
  std::vector<GS_point*> gs_points_;
  std::vector<GSVoxelOctree*> leaf_node_list;

  int layer_;
  int max_layer_;
  // std::unordered_map<VOXEL_LOCATION, GSPointList*> gs_map_points_;

  GSVoxelOctree *leaves_[8];
  GSVoxelOctree *root_voxel_;
  double voxel_center_[3]; // x, y, z
  float quater_length_;

  GSVoxelOctree(int max_layer, int layer, GSVoxelOctree *root_voxel)
      : max_layer_(max_layer), layer_(layer), root_voxel_(root_voxel)
  {
    gs_points_.clear();
    leaf_node_list.clear();
    for (int i = 0; i < 8; i++)
    {
      leaves_[i] = nullptr;
    }
    if(root_voxel == nullptr) root_voxel_ = this;
  }

  ~GSVoxelOctree() {
    for (int i = 0; i < 8; i++) {
        if (leaves_[i] != nullptr) {
            delete leaves_[i]; 
            leaves_[i] = nullptr; 
        }
    }
  }
   void get_all_gs_points(std::vector<GS_point*>& all_points)
    {
        // If this is a leaf node with points
        if (layer_ == max_layer_)
        {
            for (auto* pt : gs_points_)
                all_points.push_back(pt);
        }
        else
        {
            // Recursively collect points from all non-null child nodes
            for (int i = 0; i < 8; i++)
                if (leaves_[i] != nullptr)
                    leaves_[i]->get_all_gs_points(all_points);
        }
    }
  void UpdateGSOctree(GS_point* pv)
  {
    // cout << "layer_: " << layer_ << endl;
    if (layer_ < max_layer_) 
    {
      int xyz[3] = {0, 0, 0};
      if (pv->_points.x > voxel_center_[0]) { xyz[0] = 1; }
      if (pv->_points.y > voxel_center_[1]) { xyz[1] = 1; }
      if (pv->_points.z > voxel_center_[2]) { xyz[2] = 1; }
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
      if (leaves_[leafnum] != nullptr) { leaves_[leafnum]->UpdateGSOctree(pv); }
      else
      {
        leaves_[leafnum] = new GSVoxelOctree(max_layer_, layer_ + 1, root_voxel_);
        leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
        leaves_[leafnum]->quater_length_ = quater_length_ / 2;
        leaves_[leafnum]->UpdateGSOctree(pv);
      }
    }
    else
    {
      // 已经切割到了 叶子节点
      if(gs_points_.size() < 1)
      {


        if(root_voxel_->leaf_node_list.size()<1)
        {

        pv->index = -1;

        gs_points_.push_back(pv);
        root_voxel_->leaf_node_list.push_back(this);

        }



      }
    }
  }
};


class GSMapManager
{
public:
  GSMapManager() = default;
  std::unordered_map<VOXEL_LOCATION, GSVoxelOctree*> gs_map_;
  // std::unordered_map<VOXEL_LOCATION, GSPointList*> gs_map_points_;
  float voxel_size_;
  int max_layer_;

  GSMapManager(std::unordered_map<VOXEL_LOCATION, GSVoxelOctree*> &gs_map, float voxel_size, int max_layer)
      : gs_map_(gs_map), voxel_size_(voxel_size), max_layer_(max_layer)
  {
  };
  void UpdateGSMap(GS_point* p_v)
  {

    V3D pt_w(p_v->_points.x, p_v->_points.y, p_v->_points.z);

    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = floor(pt_w[j] / voxel_size_);
    }

    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = gs_map_.find(position);
    if (iter != gs_map_.end()) { gs_map_[position]->UpdateGSOctree(p_v); }
    else
    {
      GSVoxelOctree *octo_tree = new GSVoxelOctree(max_layer_, 0, nullptr);
      gs_map_[position] = octo_tree;
      gs_map_[position]->quater_length_ = voxel_size_ / 4;
      gs_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size_;
      gs_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size_;
      gs_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size_;

      gs_map_[position]->UpdateGSOctree(p_v);
    }
    // }
  }
};
typedef std::shared_ptr<GSMapManager> GSMapManagerPtr;

#endif // GS_OCTREE_HPP_
