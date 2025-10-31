/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "spatial.h"
#include "simple_knn.h"

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  // _xyz 的大小为 [9443, 3]，所以 points.size(0) 返回的是 9443，也就是说：
  //  P = 9443;
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  // 这行代码创建一个一维张量 means，长度为 P，并初始化为全0，数据类型为 
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  
  // 代码调用 SimpleKNN::knn 函数，计算每个点到最近点的平方距离，并将结果存储在 means 张量中。
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());

  return means;
}