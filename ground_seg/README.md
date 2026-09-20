# ground_seg — 可通行地面提取（DINOv3 ViT-B + Seed Similarity + SAM2）

机器人前视相机图片 → **可通行地面区域**，全程无标注无训练：
DINOv3 ViT-B 冻结特征以画面底部为种子做余弦相似度与连通域传播；随后用 SAM2.1
按这些候选区域生成正/负点提示，贴合到像素级边界。SAM2 只负责边界，不负责把任意
物体解释为地面；其结果与 DINO 候选 IoU 过低时会回退到 DINO 掩码。

每张图输出（`outputs/<照片名>/`；STAIRS=True 时 4 个文件，False 时前 2 个）：

- `<stem>_dino_mask.png` — SAM2 精修前的 DINO 候选，便于对照与调参
- `<stem>_ground_mask.png` — **SAM2 精修后的地面二值掩码**
- `<stem>_traversable_mask.png` — 当前与地面相同；未来加入经过验证的楼梯语义头后会合并
- `<stem>_ground.png` — 绿色叠加的最终结果

## 环境搭建

```bash
cd /home/dow/DOW/dinov3
source .venv/bin/activate   # 已建好；重建见 requirements.txt 头部注释
pip install -r ground_seg/requirements.txt   # 纯地面（STAIRS=False）的话，open_clip 那行可不装
```

包结构是单层的（`ground_seg/` 直接含 `__init__.py`），在仓库根目录下
`import ground_seg` / `python ground_seg/main.py` 都不需要设 PYTHONPATH。

## 使用

```bash
# 推荐入口：改 ground_seg/main.py 顶部配置（模型/图片/分辨率）后直接跑
python ground_seg/main.py

# 命令行版（同一功能，参数在命令行；额外输出相似度热图 _simmap.jpg）
python ground_seg/scripts/extract_ground.py test_images -o outputs

# 特征质量检查（PCA 可视化）
python ground_seg/scripts/pca_check.py test_images/street_traffic.jpg
```

`main.py` 顶部配置块：

`SAM2_REFINE=True` 默认开启，使用本仓库的
`third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt`。楼梯的“重复横线”规则仍仅保留为
调试候选（默认 `STAIRS=False`），不能安全地当作语义标签；需要稳定的楼梯类别时，应
使用仓库中 ADE20K 头并补齐匹配的 ViT-7B backbone，或针对真实场景标注并训练轻量头。

| 配置 | 默认 | 说明 |
|---|---|---|
| `CKPT` | ViT-B 权重 | 变体从文件名自动推断；注释里有 ViT-S 路径（快 2.8×） |
| `IMAGES` | `test_images` | 目录 / 单张 / 列表都行 |
| `OUTPUT` | `outputs` | 每张图一个子文件夹 |
| `RESOLUTION` | `448` | 推理分辨率；896 边缘略细但慢一倍，地面检测 448 已足够 |
| `DEVICE` | `auto` | auto / cuda / cpu |
| `STAIRS` | `True` | 楼梯识别 + 安全扣除；False 回到纯地面（只出 2 个文件，快 4×） |
| `STAIRS_THRESHOLD` 等 | 见文件 | 楼梯头调参，2026-09-18 已调好（见下节），一般不用动 |
| `SEED_ROWS` 等 | 见文件 | 地面种子/阈值调参，一般不用动 |

## Python API（供导航模块调用）

```python
from ground_seg import GroundSegmenter

gseg = GroundSegmenter(resolution=448, ckpt="dinov3/checkpoints/…vitb16….pth")
prepared = gseg.prepare("photo.jpg")     # 一次 DINOv3 前向
ground = gseg.segment_prepared(prepared) # .mask (H,W) uint8{0,255}, .coverage, .warnings, …
# 或一步到位：ground = gseg.segment("photo.jpg")
```

## 算法（`segmenter.py`）

1. 预处理：短边等比缩放至 448、拉伸到 16 的倍数、ImageNet 归一化
2. `get_intermediate_layers` 取 ViT 最后一层 patch 特征（L2 归一化）
3. 底部 10% 行 × 中间 60% 列为种子（前视相机先验：地面在画面底部）；
   裁掉与种子均值相似度最低的 20% 后取均值 → 种子向量
4. 全图与种子的余弦相似度 → Otsu 阈值阶梯（-0.05/-0.10 兜底）
5. 8 连通域生长，只保留与种子连通的区域（排除远处相似面）→ 形态学开 + 去碎块
6. 双线性上采样 ×16 → 中值滤波 → 原分辨率掩码

失败兜底：所有阈值都不达标时返回底部种子矩形，保证导航始终有"脚下地面"答案。

## 楼梯识别与安全扣除（`stairs.py`）

为什么需要：地面头凭**外观**（立面阴影 + 透视）把楼梯排除在可通行区域外，
通常正确但那不是"识别"——系统不知道非地面里哪个是楼梯，也没有保证
（同材质缓步台可能被误并入地面）。楼梯头显式识别楼梯，并做**安全兜底**：
`subtract_from_ground` 把楼梯区域膨胀 2% 图高后从地面掩码强制扣除
（膨胀带吸收两个头各自的上采样边界误差）。扣除超过原地面 30% 会打强警告
（楼梯误检啃地面的金丝雀）。

机制与门检测完全相同（CLIP ViT-B-32 零样本多尺度滑窗，与门头共享同一份
模型缓存），只有提示词和几何默认不同：窗口加到 224（近距大楼梯整段上下文）、
无纵横比约束（楼梯可宽可窄）、最小面积更大（楼梯是大目标）。

**2026-09-18 调参结论**（3 张楼梯实拍 + 7 张无楼梯对照）：

- **阈值 0.48**：7 张对照热峰全部 ≤0.42，中距室内楼梯 0.54、近距 0.96，
  两侧余量各 ~0.06
- 负提示词三组缺一不可：室内杂物（floor/wall/railing/…）＋ **室外**
  （road/sidewalk/plaza/facade… 不加的话广场地砖栅格能冲到 0.68）＋
  **办公栅格**（glass partition/bookcase/window frame——玻璃隔断栅格最
  像楼梯，加前 0.45、加后 0.42）
- 正提示词里的 "a short flight of steps" / "interior stairs" 把中距短梯段
  从 0.50 抬到 0.54，是 0.48 阈值能同时覆盖近距 + 中距的关键

实测：galileo_test-4（近距花岗岩梯）橙色贴满梯段、覆盖率 41%→38%（安全
扣除生效）；galileo_test_2（走廊短梯段）检出并扣除；7 张对照楼梯掩码全空、
地面掩码逐字节不变。

## 性能（RTX 5060 Ti 8GB，默认 ViT-B/16 + 448 分辨率）

- GPU 稳态纯地面 **~0.15s/图**；开楼梯头（STAIRS=True）**~0.7s/图**
  （CLIP 滑窗是主要开销，与 DINOv3 共享同一次前向的 prepared 图）
- 换 ViT-S/16 更快；CPU ~7s/图（结果与 GPU 一致）
- 显存峰值 ~0.5GB（DINOv3）+ ~0.6GB（CLIP，门/楼梯头共享一份）

## 扩展工具（可选，默认不用）

包里保留了之前做的完整流水线（`scripts/extract_scene.py`），如需门检测/场景
分割再启用：

- **场景分割** `SceneSegmenter`：patch 特征 K-means（k=8），只有外观区域、
  无"地毯/墙"这类语义标签
- **门检测** `DoorDetector`：CLIP ViT-B-32 零样本滑窗（需 `open_clip_torch`，
  权重首次运行自动下载）；调参结论：阈值 0.75、滑窗 96/128/160 固定尺寸 +
  896 分辨率可分开相邻门（窗口随分辨率放大反而并框）；墙面小面板误检用
  `--door-min-area 0.02` 滤
- **官方 ADE20K 语义头** `ground_seg/semseg.py`：DINOv3 ViT-L + Mask2Former
  Adapter（门/地板/台阶三类），需手动下载两个门禁权重放进
  `dinov3/checkpoints/`；8GB 卡需 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

这些头与地面头共享同一次 DINOv3 前向，详见各模块 docstring。

## 已知局限

- 极强反光的地面（镜面倒影）可能把倒影里的"墙"判成地面（连通域传播已缓解大部分）
- 远处地面与墙面颜色/材质几乎一致时，边界可能略靠前（保守，不越界到墙上）
- **昏暗楼梯间（galileo_test_3 类：隔门帘远望）楼梯头可能漏检**——安全不
  依赖楼梯头召回：地面头已凭外观排除楼梯；楼梯头职责是显式识别 + 同材质
  缓步台兜底
- K-means 场景分割无语义；走廊纵深远处小门漏检、白门白墙混淆（仅扩展工具相关）
