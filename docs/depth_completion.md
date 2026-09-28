# 深度补全用于 GS-LIVO 高斯初始化：可行性分析与方案

分支 `gs_livo_depth_complementation`（基于 `complete` @ caf68e2）。第 1–5 节是实现前的分析，第 6 节是实现和 CBD_Building_01 上的实测结果。

## 结论
可以接入，而且接口很自然。Gaussian-LIC2 的做法可以拆成四步：稀疏深度图 → SPNet 补全 → 对齐与筛选 → 反投影得到新点。
补全出来的点正好可以作为 GS-LIVO 已有的初始化输入 `gslivo::InitPoint`，交给 `GSMap::updateWindow` 的叶子容量机制去重，不需要改动地图、滑窗和优化的结构。

真正需要做的只有三件：
1. 在 bjb1 的 CUDA 12.8 上跑起 SPNet 推理；
2. 为补全点算法向，并控制补全点对光度 EKF 的影响；
3. 用实验确认收益。收益预计比 Gaussian-LIC2 小，原因见第 4 节。

## 1. Gaussian-LIC2 是怎么做的（`src/gaussian.cpp` Dataset::addFrame、`src/depth_completer.cpp`）
- **频率**：只在关键帧上做，每 `select_every_k_frame: 5` 帧一次。
- **输入**：
  - RGB（640×512，归一化到 [0,1]）；
  - 当前帧 LiDAR 投影得到的稀疏深度（米，除以 200 归一化）；
  - 有效掩码（depth>0）。
- **网络**：SPNet Large（`Large_300.pth`，942 MB，ConvNeXt 结构），导出 ONNX 后用 TensorRT 8.6 以 FP16 推理。
  - bjb2 上 4090 实测（`trtexec_512_640.log`）：每帧 7.6 ms，显存峰值约 0.5 GB。
- **对齐与筛选**：
  1. 在有 LiDAR 的像素上计算"补全深度 − LiDAR 深度"的平均差。差的绝对值 ≥ 0.1 m 时整帧放弃，差值小于该阈值时整体减去这个差。
  2. 用 Sobel 梯度去掉深度边缘（梯度 ≥ 0.1），避免在前后景交界处生成悬空点。
  3. 把图像分成 `patch_size: 10` 的小块。块内没有 LiDAR 点时，取块内补全深度最小的像素，深度不超过 `max_depth: 20` m 才反投影成新点。
- **生成高斯**：各向同性球，尺度 = 深度 / f，不透明度 0.1，颜色取像素值。
- **许可证**：Gaussian-LIC2 是 GPL-3。GS-LIVO 继承 FAST-LIVO2，`src/gs-livo/LICENSE` 是 GPL-2。
  - GPL-2-only 的代码不能直接并入 GPL-3 代码。所以本方案**不复制**它的源码，而是按上面的算法自行实现。
  - TensorRT 推理封装是通用样板代码，也自行实现。

## 2. 映射到 GS-LIVO
| Gaussian-LIC2 | GS-LIVO 对应位置 |
|---|---|
| 稀疏深度图 depth_msg（Coco-LIC 发布） | `processFrameGS` 里已有稠密扫描 `scan`（世界坐标）和位姿：投影到去畸变图像，取 z-buffer 最小深度，约 1 ms |
| 相机 640×512，fx≈646.8 | 完全相同：GS-LIVO 内部就是在 640×512 的去畸变图像上工作，可直接用 512×640 的 engine |
| 新点 `pointcloud_` | 新增 `InitPoint`，交给 `GSMap::updateWindow`（或新接口 `GSMap::insertPoints`） |
| "块内无 LiDAR 点才补" | GS-LIVO 的叶子容量机制天然去重：已被 LiDAR 高斯占满的叶子不会再加点；块筛选只用来控制数量 |
| 各向同性球 | GS-LIVO 是贴表面的扁平高斯，需要法向：补全深度是稠密的，可由相邻像素反投影点的叉积得到法向，比现在无平面时用视线方向代替更好 |

建议的调用位置：在 `processFrameGS` 里，**光度 EKF 更新之后、窗口优化之前**，每 k 帧做一次。这时位姿最准，反投影误差最小。

```
updateWindow(LiDAR scan)  →  光度 EKF  →  [每 k 帧: 稀疏深度 → SPNet → 对齐/筛选 → InitPoint(p_w, n_w) → GSMap::insertPoints]  →  optimize
```

## 3. 需要解决的问题
1. **推理后端**。bjb1 是 CUDA 12.8，bjb2 上 TensorRT 8.6 的 cu11.8 版本和已构建的 engine 不能直接搬过来。两种方案：
   - **(a) TensorRT**：推荐。可用 NVIDIA apt 源或 pip 的 `tensorrt-cu12`（TRT 10）。TRT 10 删除了 `getNbBindings/executeV2`，要改用 `setTensorAddress/enqueueV3`；从 ONNX（942 MB，可从 bjb2 拷贝）在 bjb1 上重新构建 FP16 engine，构建约 6 分钟。
   - **(b) LibTorch TorchScript**：GS-LIVO 已经链接了 LibTorch，不需要新依赖；但推理可能明显慢于 TRT FP16，需要实测。
2. **补全点污染位姿估计**。光度 EKF 的参考图是窗口渲染图。如果补全点深度错了，渲染会在 LiDAR 测量点前面画出错误表面，从而把位姿带偏。缓解手段（按代价从小到大）：
   - 补全点用低初始不透明度（Gaussian-LIC2 用 0.1）；
   - EKF 选点时要求该像素的渲染深度与 LiDAR 深度一致；
   - 给高斯加一个"来源"标记，渲染 EKF 参考图时排除补全点。这需要在 GPU 缓冲区旁维护一个并行的标记列，`removeRows/appendVoxels` 都要跟着改。
3. **补全深度的可靠性**。天空、玻璃、远处的补全深度不可信：沿用 20 m 上限、边缘剔除和整帧平均差检查；可以再加"补全深度与已渲染深度差太大则丢弃"。
4. **实时性**。GS-LIVO 目前每帧约 97 ms（LIO 43 + GS 54），已经在 10 Hz 边缘。每帧都补全会增加约 10 ms。每 5 帧一次，平均只增加约 2 ms；也可以放到单独线程和 CUDA stream 上异步做，结果在下一帧插入。

## 4. 预期收益：可能比 Gaussian-LIC2 小
- 视场角：相机水平 52.6°、竖直 43.2°，完全落在 Livox Avia 的 70.4°×77.2° 视场内；而且 Avia 是非重复扫描，GS-LIVO 每帧都用稠密扫描初始化，几帧之后角度覆盖就补齐了。
- 当前 CBD_Building_01 上窗口渲染覆盖率约 95%。剩下 5% 的空洞主要可能来自：天空（补全也无能为力）、LiDAR 没有回波的表面（玻璃、黑色物体）、超出量程的远处、近处盲区（0.8 m 以内）。
- 所以**第 0 步应先量化空洞的来源**，再决定是否值得做：统计未覆盖像素中，属于天空（图像上方、无任何 LiDAR 回波）和属于"本应有表面"的各占多少。后一部分才是深度补全能改善的上限。

## 5. 实施步骤与验收
0. **量化空洞**：在 `processFrameGS` 里按帧输出 alpha 图，统计空洞像素按行/列的分布，以及空洞处邻近是否有 LiDAR 点。
1. **推理**：在 bjb1 上装 TensorRT，从 ONNX 构建 512×640 FP16 engine，实现 `DepthCompleter`（TRT 10 API），先离线测单帧耗时和精度（与 LiDAR 深度的误差）。
2. **初始化**：实现稀疏深度投影、对齐、边缘剔除、分块取点、法向计算，新增 `GSMap::insertPoints`；补全点不透明度可配。
3. **配置开关**：`gs/depth_completion_en`、`gs/dc_every_k`、`gs/dc_max_depth`、`gs/dc_patch`、`gs/dc_init_opacity`，默认关闭，保证与 `complete` 行为一致。
4. **验收**：用 CBD_Building_01 做开/关对比，看四个数：
   - 优化前预测 PSNR 和覆盖率（`gs_stats.py`）；
   - 每帧耗时；
   - 闭环首尾误差（`evo/pose_output_en`），以及与不开补全时轨迹的差异。
   另外用 viewer 从新视角检查补全区域有没有漂浮物。
5. 在单独的工作区（如 `gslivo_ws_dc`）编译运行，不影响已部署的 `gslivo_ws2`。

## 6. 实现与实测（2026-09-29）
### 实现
- `DepthCompleter`（`include/depth_completer.h`、`src/depth_completer.cpp`）：自行实现的 TensorRT 10 推理封装。
  - 输入输出直接用 PyTorch 的 GPU 张量，在独立的 CUDA stream 上执行。
  - 编译时不提供 `-DGSLIVO_TENSORRT_DIR` 也能编过，只是运行时报"未启用"。
- `VIOManager::depthCompletionGS`：每 k 帧一次，在光度 EKF 更新之后、窗口优化之前执行：
  1. LiDAR 扫描投影成稀疏深度（z-buffer 取最近）；
  2. SPNet 补全，与 LiDAR 深度对齐，平均差 ≥ `dc_align_thresh` 时整帧放弃；
  3. 剔除深度边缘；
  4. 每个没有 LiDAR 点的 `dc_patch` 小块取最近像素反投影，法向由补全深度的相邻像素叉积得到；
  5. 以 `dc_init_opacity` 作为初始不透明度，交给 `GSMap::insertPoints`。
- `GSMap::insertPoints`：与 LiDAR 点共用初始化和叶子容量去重（`initFrom`）。补全点落入的体素记为本帧已观测，窗口有空位时加入窗口。
- 与 Gaussian-LIC2 的一处不同：新增相对边缘判据 `dc_edge_rel`（每像素深度变化 / 深度，默认 0.05）。
  - 原判据是 Sobel 幅值 ≥ 0.1 的绝对阈值。远处倾斜的表面即使没有遮挡边缘，每像素深度也会变化十几厘米，所以原判据会把这些表面几乎全部当作边缘剔除。
  - 设 `dc_edge_rel: 0` 可恢复原判据。
- 引擎：TensorRT 10.9（CUDA 12.8）从 Gaussian-LIC2 导出的 ONNX 构建，FP16，512×640。
  - 4090 上推理 7.95 ms，显存 0.5 GB；加上前后处理，每次补全共约 17 ms，按每 5 帧一次平均约 3.5 ms/帧。
  - 构建脚本：`/root/autodl-tmp/deps/spnet/build_engine.sh`。

### 实测（CBD_Building_01 整段，1182 帧，同条件 A/B：`gsviz/slam/gslivo/ab_run_gslivo.sh`）
| 配置 | 优化前预测 PSNR | 整图 PSNR | 覆盖率 | 补全新增高斯 | GS 耗时/帧 |
|---|---|---|---|---|---|
| 窗口 10 万，关补全 | 25.02 | 23.46 | 95% | – | 56 ms |
| 窗口 10 万，补全（Gaussian-LIC2 默认参数） | 25.11 | 23.41 | 95% | 9.3k | 59 ms |
| 窗口 10 万，补全 + 相对边缘，patch 10 | 25.07 | 23.42 | 95% | 18.5k | 59 ms |
| 窗口 10 万，补全 + 相对边缘，patch 4 | 24.73 | 23.00 | 94% | 211k | 59 ms |
| **窗口 20 万，关补全** | **26.01** | **28.12** | **98%** | – | 60 ms |
| 窗口 20 万，补全 + 相对边缘，patch 10 | 26.00 | 28.39 | 98% | 18.8k | 63 ms |

- **跟踪**：所有配置与"关补全"轨迹的平均差 ≤ 0.7 cm，闭环首尾误差 3.7–4.8 cm。补全不影响位姿。
- **空洞来源**（补全帧上统计窗口渲染 alpha < 0.5 的像素，占整幅图的百分比，窗口 10 万）：
  - Gaussian-LIC2 边缘判据：空洞 5.5%，其中可补 1.07%，超过 20 m 1.53%，被判为边缘 2.90%；
  - 相对边缘判据：可补 3.90%，边缘仅 0.06%。
- **可补的空洞仍补不上的原因**：每次约 566 个候选点里，只有约 78 个落在空叶子里；其余位置地图里**已经有高斯**，只是它们所在的体素不在 GPU 滑窗里。窗口 10 万时每帧都是满的，本帧没被 LiDAR 扫到的视野内体素会被换出；窗口满时 `insertPoints` 也加不进去。
- 密集补点（patch 4）让窗口更拥挤，空洞反而增加到 6.6%，PSNR 下降。

### 结论
1. 深度补全已按 Gaussian-LIC2 的方式接入并能正常工作（对齐误差 −0.3 cm，无跳帧），但在 CBD_Building_01 上**没有可测出的收益**：预测 PSNR 的变化在运行间噪声内（约 ±0.05 dB），整图 PSNR 最多 +0.3 dB。
2. 原因是 Avia 的非重复扫描几帧内就把相机视场的几何补齐了，剩下的空洞主要是**窗口容量**问题，而不是缺几何。窗口从 10 万加到 20 万，预测 PSNR +1 dB、整图 PSNR +4.7 dB，覆盖率 95%→98%，每帧 GS 耗时 +4 ms。
3. 建议：
   - 本数据集优先调大窗口（或改进窗口策略：视野内未被扫到的体素也保留在窗口里）；
   - 深度补全保留为可选功能（默认关闭）。它更可能在 LiDAR 覆盖差的场景有用：机械式 LiDAR 的竖直视场窄于相机，或大片玻璃、黑色、远处表面。可以用 scene_data 等数据再验证。
