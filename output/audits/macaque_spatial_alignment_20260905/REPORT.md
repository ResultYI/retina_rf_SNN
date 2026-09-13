# 22-cell macaque Canonical V1 spatial alignment audit

- 日期：2026-09-05（Asia/Tokyo）
- branch：`rgc-readout-v2`
- HEAD：`fea28de038821fadee279b93728688b34bcb3bac`
- 与指定的此前审计版本：branch、HEAD 均相同；不以此代替实际 checkpoint/source 核验。
- 审计开始前 `git status --short`：

```text
?? "earch code and final audit evidence\357\200\242"
```

- 原有 untracked 文件未触碰。完成后的新增文件仅位于本审计目录；完整状态见 `evidence_manifest.json` 的 `final_verification`。
- 实际读取：指定 shared-BC lineage 的 22 个 `cells/<cell>/model-trained.pt`；指定 LN lineage 的 22 个最终 `ln-trained.pt`；双方共 44 个 `validation-predictions.pt`；逐 cell results/config；88 个已冻结 LN inner 候选 checkpoint（仅补充中心一致性，不替代最终 refit）；官方本地 README、model notebooks、`retinatools/library.py`、两份 DOCX catalog、Fig3 Excel。
- 无法读取/未取得：Fig2 Excel 是 64-byte annex pointer，缺少 workbook payload；检查到的本地材料没有原论文完整正文。22 cells 的独立生理 RF-center 坐标及其不确定度没有恢复出来。Fig3 可读，但不是当前 22 个 MC/PC cells 的 RF-center ground truth。
- 本轮训练次数：**0**。
- 修改 production source/model/data：**false**。无 fit、调参、模型 forward、新 RF 估计或 illusion 重跑；只构造模型并 strict-load 检查几何、读取冻结参数、重算已保存 logits 的 NLL。
- 已检查输入文件：273 个，分析前后 SHA256 全部一致。分析脚本及补充证据单独列在 manifest 中。

## 1. Executive conclusion

**GO: targeted spatial-alignment experiment justified**

当前证据不足以支持“所有 22 个 cell 的 RF center 都可固定在 crop `(0,0)`”。固定中心是当前实现事实；它不是这批数据已验证的实验事实。最终 LN refit 的 radial offset 中位数为 **1.6488 pooled pixels = 0.08888°**，18/22 超过一个 pooled pixel，12/22 超过相应 Canonical direct-BC support 半径。这里的比较尺度来自实际输入和固定几何，不是显著性阈值。

偏移与 `Canonical − LN` validation NLL 差值相关：Pearson **0.8273**、Spearman **0.7267**；排除最大偏移且最大 gap 的 `68#10` 后为 **0.6222 / 0.6857**。固定中心因此是一个有实际尺度、值得针对性检验的预测和机制解释限制。**这些结果没有量化 centering 的因果贡献，也没有证明改中心能改善预测。**

两个证据边界必须同时保留：LN center 是训练得到的 **effective center**；官方模型的 `xoff/yoff` 也是模型参数。两者均不能代替独立生理 RF ground truth。已保存的 inner 候选与 final refit 提供有限的一致性证据，不构成独立种子重复或 RF 定位置信区间。原 validation 已长期用于 development，本报告所有 gap 统计均属 **exploratory/development evidence**。

## 2. Canonical spatial implementation facts

### 2.1 实际数据路径与坐标

[data/schottdorf_lee_multirecording.py](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_multirecording.py) 的 `load_schottdorf_recording` 和 `load_schottdorf_cell` 均直接写入 `torch.zeros((1,2))`。22 个最终 Canonical checkpoint 的 `cell_positions_degs` 均为 `[[0.0,0.0]]`。这些是分别拟合的 N=1 模型，不能解释成 22 个 RGC 物理共位。

[data/schottdorf_lee_2021.py](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py) 的 `_pooled_lm_signal` 对同一 256×256 movie 做统一 51×51 中央 crop，再对每个 3×3 block 求均值，得到 17×17 L+M Weber drive。没有 cell-specific crop、平移或 RF alignment。`load_schottdorf_movie_drive` 在 cell 循环前只生成一次 movie drive；[当前 lineage run.py](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/run.py) 的 `main/factory` 将该 grid 和每 cell 的零坐标直接传给 `build_mechanistic_retina`。

`_cone_positions` 使用：

\[
p_{native}=4.6/256=0.01796875^\circ,\qquad
p_{pooled}=3p_{native}=0.05390625^\circ,
\]
\[
x_j=(j-8)p_{pooled},\qquad y_i=(8-i)p_{pooled},\qquad i,j=0,\ldots,16.
\]

x 坐标从左到右为：

```text
[-0.43125, -0.37734375, -0.3234375, -0.26953125, -0.215625,
 -0.16171875, -0.1078125, -0.05390625, 0,
  0.05390625, 0.1078125, 0.16171875, 0.215625, 0.26953125,
  0.3234375, 0.37734375, 0.43125] degree
```

y 按图像行顺序是上述数组的相反数。存储 flatten 顺序为 row-major。中心到中心的跨度为 0.8625°，51 个 native pixels 的宽度为 0.91640625°；二者不要混用。

还有一个较小的数字坐标约定：`(256−51)//2=102`，crop 覆盖原图 index 102…152，所以 crop center 是原图 index `(127,127)`；256×256 图像几何中点是 `(127.5,127.5)`。二者每轴差半个 native pixel，即 0.008984375°。本报告 LN offset 均相对**当前 crop 原点**，未擅自调整该合同，也未把半像素差称为实验配准误差。

### 2.2 Support、Gaussian basis 与自由度

[support_partition.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/support_partition.py) 的 `build_support_partition` 用 `cdist(cell_positions, cone_positions)` 生成以下 full disks；`partition_spatial_basis` 在 mask 内重新归一化。

| cell type | direct BC radius | AC/broad-BC radius | 当前 BC / AC grid pixels | 未截断 Gaussian σ（degree） |
|---|---:|---:|---:|---|
| midget / PC | 0.06° | 0.13° | 5 / 21 | 0.05, 0.14 |
| parasol / MC | 0.10° | 0.15° | 9 / 21 | 0.09, 0.20 |

虽然 MC、PC 的 AC 半径不同，在该粗网格上都恰好选中 21 个 pixels。AC disk 与 BC disk 重叠并包含它，不能写成不相交 annulus。

[bipolar_subunits.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/bipolar_subunits.py) 的 `_spatial_basis` 根据相同 cell center 计算 `exp(−distance²/(2σ²))`，σ 按类型固定；`PathFeatureBank.__init__` 将 `spatial_basis`、`path_spatial_basis` 和 support 注册为 buffer。没有可学习的中心、平移、宽度或逐 pixel 自由权重。22 个 checkpoint 的这些 buffer 与当前默认构造器逐 tensor **完全相等**，不是只看配置字符串。

[spatial_contract.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/spatial_contract.py) 的 `register_spatial_contract/_check_loaded_geometry` 检查空间 identity 和加载几何。22/22 strict-load 通过；causal identity 为 `h1-shared-bc-direct-broad-ac`、spatial identity 为 `bc-central-disk_ac-overlapping-full-disk`、schema 为 `schottdorf_canonical_v1_shared_bc_development`、revision=4、stage=`trained`。本报告没有加载旧 independent-AC Canonical checkpoint。

| 部件 | 固定空间性质 | 可学习量及实际边界 |
|---|---|---|
| H1 | [graph.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/graph.py) `ConeGraph`：289 cone 节点间 0.18° 邻域、Gaussian 权重、行归一化及连接均固定 | [h1_pathway.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/h1_pathway.py) `H1Pathway` 的 τ/delay；[pathway_gates.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/pathway_gates.py) 的有界 H1 amplitude。没有空间平移/半径学习 |
| BC | 相对 RGC 零中心的 fixed disks、两个固定 Gaussian modes | `BipolarSubunits.positive_weights` 学习 spatial×temporal basis 的 softmax mixture；τ/delay 可学习；cell-specific BC gain 可学习。有效宽度/时间形状可随混合改变，但 basis 不移动 |
| AC | 相同零中心的 broad-BC disk；没有独立可移动的 AC spatial encoder | [amacrine_pathways.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/amacrine_pathways.py) `presynaptic_states/forward` 接收 shared `BC_broad`，学习后续 τ/delay、local/transient gates 及 AC gain；共享 BC mixture |

H1 需要单独区分：**其图邻域围绕每个 cone，不是围绕 RGC 零点建一个单一 Gaussian。** `feature_bank.h1_support` 虽按 RGC center 和 0.18° 生成，实际 H1 forward 是 `G → temporal filter → Gᵀ`，再从 cones 中减去 surround；该 RGC-centered support buffer 没有被用于裁剪这条 H1 forward。因此 H1 经 BC/AC 读出后的有效 RF 可以超出 direct-BC/AC disk；不能把 0.18° 的 buffer 误当所有 H1 路径的最终边界。

[shared_subunits.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/shared_subunits.py) 的 N=1 `connection_matrix` 是固定 `[[1]]`，不能通过邻 cell 混合移动 RF。全部实际 trainable parameter 名称已逐 checkpoint 保存在 manifest；没有 translation/center 参数。

## 3. Independent RF-center evidence

证据检索限定为本地 repository 材料，没有网络补取或联系作者。官方本地 repository HEAD 为 `cffefb08c760f04c9a951da46061b361d2288e9b`，本轮 status 为空；使用临时命令参数处理 Git ownership 检查，没有改 Git global config。

| 来源 | 实际观察 | 能支持什么 / 不能支持什么 |
|---|---|---|
| [官方 README](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/README.md)，Cell/Recording list | recording、class、cell key、eccentricity；没有每 cell 的 movie-plane x/y 或定位误差 | eccentricity 是相对视网膜参考点的偏心度，不能换成 crop 内 RF 偏移；没有严格零中心保证 |
| [Cell List.docx](<D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/data/Cell List.docx>) | recording/type/cell key 对照 | 没有每 cell RF x/y |
| [CellsList.docx](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/data/CellsList.docx) | 原始记录清单包含一次 “Repeat centered” 记述，并有质量/重复记录备注 | 提示实验中存在重新居中的操作；缩写 `C/LR/RR` 未找到可靠释义，不把它们解读为精确居中证明；没有定位坐标及误差 |
| [官方 library.py](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/retinatools/library.py) `get_RF/filter_video_MC/filter_video_PC`；`run_model` notebooks | 在完整 movie 上按每组 `xoff/yoff` 生成 RF，不是统一固定零点；可对应当前 cohort 的 13 个参数组均非零 | 一手**作者模型**证据；没有证实每个 offset 都来自独立 RF 定位测量，不能作为 ground truth |
| [Fig2publication.xlsx](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/exampleRFs/Fig2publication.xlsx) | 内容是 `/annex/objects/MD5-s20606799--d23fd7777e29a2c74fb9221fd2d33b7f`；文件 64 bytes | workbook 未物化，无法读其中 RF maps；不得推定覆盖哪些 cells |
| [Fig3publication.xlsx](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/exampleRFs/Fig3publication.xlsx) | 可读 3,092,078 bytes，7 sheets；例如 `rn101lum!B2:C2=(126,131)`、`gn292lum!B2:C2=(127,128)`，标签为 MxX/MxY | 示例 101、292 不在当前22-cell recording集合；231 是 excluded S-cone `68#12`。这些是示例峰位置，坐标基准/跨通道含义尚未完整确认，不转换成当前 cohort 的生理中心 |
| 原论文正文 | 本地 inspected materials 中未找到原论文完整正文；现有 timing 报告给出正文链接，但没有可核验的 centering 段落 | “原论文明确说部分 RF 偏离 movie center/定位不充分”在本轮为 **UNVERIFIED**；不能把旧报告或未读论文当一手全文证据 |

官方数据/论文来源标识保存在本地 `datacite.yml` 和 README；这里没有将在线版本状态称为已验证。全部 DOCX 提取文本、Excel sheet/range 及 notebook cell 索引保存在 `independent_evidence.json`。

### 3.1 作者模型中心的交叉参照

作者 `get_RF` 使用 `linspace(−128,128,256)`，其相邻坐标间隔为 256/255，不能直接把 `xoff/3` 当精确 pooled offset。与当前 crop 对齐的数字变换为：

\[
i_x=127.5+(255/256)x_{off},\qquad
x_{pooled}=[0.5+(255/256)x_{off}]/3,
\]

y 同理先得到 image-down 坐标，再乘负号转 Canonical degree y。这个变换依据作者数字 grid 和当前 crop 实现，不是额外实验校准。

| cell | 官方 notebook | xoff, yoff（作者坐标） | 与 final LN 中心的距离（pooled pixels） |
|---|---|---:|---:|
| 68#3 | MC_off, code cell 3 | −10.0, −3.4 | 0.5030 |
| 67#6 | MC_off, 8 | −8.4, 2.5 | 0.3385 |
| 69#6 | MC_off, 13 | 1.1, 4.1 | 0.2848 |
| 69#4 | MC_on, 3 | 3.5, −2.0 | 0.3516 |
| 70#34 | MC_on, 8 | −2.5, −5.5 | 0.3723 |
| 69#7 | MC_on, 13 | 3.4, 1.0 | 0.2060 |
| 70#7 | PC_Gon, 3 | 0.0, −1.0 | 0.1634 |
| 67#21 | PC_Gon, 8 | 2.9, −2.2 | 0.2108 |
| 70#15 | PC_Gon, 13 | −5.7, 2.9 | 0.7090 |
| 67#34 | PC_Ron, 3 | −1.6, 6.0 | 0.5774 |
| 68#7 | PC_Ron, 8 | −6.8, −6.5 | 0.4647 |
| 67#14 | PC_Ron, 13 | 2.9, −1.8 | 0.0766 |
| 68#11 | PC_off, 2 | −1, 1 | 1.5988 |

cell 索引均为 notebook 的 zero-based cell index。映射依据 notebook 的 recording 文件与当前 catalog，完整路径在 `official_model_centers.csv`。12/13 距离小于一个 pooled pixel，是有限的跨模型一致性；`68#11` 明显不一致，也不应隐藏。`68#10`、`69#21` 没有这些 notebooks 中对应的中心参数。

### 3.2 为什么当前项目置零

可证实的实现沿革是：minimal adapter 已使用零位置，multi-recording adapter 继续直接赋零，而 catalog 只保存 cell type、polarity、eccentricity 等信息。当前 Git 的 `log -S 'cell_positions_degs=torch.zeros'` 只定位到整批代码纳入的 `fea28de`，不足以恢复当时作者的科学理由。

因此最准确的定性是：**当前 loader 中硬编码的统一居中假设，配合未接入每 cell RF-center metadata。** 没找到支持精确零点的实验依据；“历史简化”符合实现形态，但原始决策动机为 **UNVERIFIED**。也不能说“官方完全没有空间信息”：作者模型 offsets 和部分示例 RF 文件确实存在，只是尚不能作为当前22 cells的独立配准合同。

## 4. 22-cell LN center table

[CenterSurroundLN.__init__/gaussians](D:/PythonProject/retina_rf_SNN/baselines/center_surround_ln.py) 定义 `grid_xy` 为每轴 −8…8 的 **pooled pixels**，x 向右、y 向图像下方。`center_xy` 不经过 tanh 等变换，也没有代码中的位置边界；Gaussian 用它直接计算 squared distance，两个空间成分共用同一 center。

[select_and_refit_ln](D:/PythonProject/retina_rf_SNN/training/mechanistic_retina/center_surround_ln.py) 在 training-only inner split 选择 regularization 和步数，然后 fresh full-train refit；[run_ln_cell](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/schottdorf_center_surround_ln.py) 将 `selection.refit.model.state_dict()` 保存到 `ln-trained.pt`。本轮22/22均核验该路径、schema、cell_id、`refit_steps=best_step` 及 `full_train_fresh_refit=true`。表中没有使用 inner-best 或旧 R4 comparator NLL。

定义 `d=hypot(x,y)`、`Δ=NLL_Canonical−NLL_LN`。pixel 坐标栏就是 `center_xy` 原始值的显示舍入；degree y 已翻转。CSV 保留原 float32 参数展开后的精度。NLL 单位为 nats/valid bin，Δ>0 表示 Canonical 较差。

| cell | group | raw x,y（pixels；y↓） | d（pixels） | x,y（degree；y↑） | d（degree） | LN NLL | Canonical NLL | Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 67#4 | PC_OFF | 1.4166, 1.1041 | 1.7960 | 0.07636, -0.05952 | 0.09682 | 0.479696 | 0.464896 | -0.014799 |
| 67#6 | MC_OFF | -2.2959, 0.9076 | 2.4687 | -0.12376, -0.04893 | 0.13308 | 0.430726 | 0.442264 | 0.011538 |
| 67#7 | MC_ON | 1.6148, -0.4886 | 1.6871 | 0.08705, 0.02634 | 0.09095 | 0.428922 | 0.456887 | 0.027965 |
| 67#14 | PC_ON | 1.0699, -0.4791 | 1.1723 | 0.05768, 0.02583 | 0.06319 | 0.366129 | 0.372925 | 0.006796 |
| 67#21 | PC_ON | 1.0519, -0.7598 | 1.2976 | 0.05670, 0.04096 | 0.06995 | 0.298146 | 0.309125 | 0.010979 |
| 67#26 | PC_ON | 0.9772, 0.0235 | 0.9775 | 0.05268, -0.00127 | 0.05269 | 0.515616 | 0.512516 | -0.003100 |
| 67#33 | MC_OFF | -0.0369, -1.8225 | 1.8229 | -0.00199, 0.09824 | 0.09826 | 0.373482 | 0.369917 | -0.003565 |
| 67#34 | PC_ON | -0.2859, 1.5868 | 1.6124 | -0.01541, -0.08554 | 0.08692 | 0.377447 | 0.386382 | 0.008935 |
| 68#3 | MC_OFF | -2.7970, -0.6076 | 2.8622 | -0.15078, 0.03275 | 0.15429 | 0.486941 | 0.519420 | 0.032479 |
| 68#4 | PC_ON | -0.1510, -0.1615 | 0.2211 | -0.00814, 0.00870 | 0.01192 | 0.423185 | 0.427100 | 0.003915 |
| 68#7 | PC_ON | -1.6275, -1.9608 | 2.5482 | -0.08773, 0.10570 | 0.13736 | 0.356707 | 0.385218 | 0.028511 |
| 68#10 | MC_ON | -4.5704, -6.2054 | 7.7068 | -0.24637, 0.33451 | 0.41545 | 0.315079 | 0.421071 | 0.105992 |
| 68#11 | PC_OFF | -0.2158, -1.0993 | 1.1203 | -0.01163, 0.05926 | 0.06039 | 0.387451 | 0.372822 | -0.014630 |
| 69#3 | PC_OFF | 0.2687, 0.6301 | 0.6850 | 0.01449, -0.03397 | 0.03693 | 0.563169 | 0.553158 | -0.010011 |
| 69#4 | MC_ON | 1.1303, -0.2072 | 1.1491 | 0.06093, 0.01117 | 0.06194 | 0.427979 | 0.433251 | 0.005272 |
| 69#6 | MC_OFF | 0.7037, 1.3008 | 1.4790 | 0.03794, -0.07012 | 0.07973 | 0.430207 | 0.435078 | 0.004870 |
| 69#7 | MC_ON | 1.2837, 0.7044 | 1.4642 | 0.06920, -0.03797 | 0.07893 | 0.458074 | 0.463303 | 0.005229 |
| 69#21 | PC_OFF | 4.5529, 1.1893 | 4.7057 | 0.24543, -0.06411 | 0.25366 | 0.472138 | 0.506465 | 0.034327 |
| 70#1 | PC_ON | 0.1481, 5.6389 | 5.6408 | 0.00798, -0.30397 | 0.30407 | 0.525037 | 0.545601 | 0.020564 |
| 70#7 | PC_ON | 0.3094, -0.0858 | 0.3211 | 0.01668, 0.00463 | 0.01731 | 0.476248 | 0.472401 | -0.003846 |
| 70#15 | PC_ON | -1.9555, 1.8004 | 2.6581 | -0.10542, -0.09705 | 0.14329 | 0.430160 | 0.439213 | 0.009053 |
| 70#34 | MC_ON | -0.9334, -1.4032 | 1.6853 | -0.05032, 0.07564 | 0.09085 | 0.349418 | 0.368022 | 0.018604 |

## 5. Prediction-gap analysis

NLL 从双方最终逐 cell `results.json` 提取，并用冻结 logits 计算 `mean(softplus(z)−qz)` 独立核对；22 pairs 的 target/mask/source IDs/trial indices 全部相同，NLL 与原记录误差均小于 1e−7。每 cell 等权，未把多 trial 当更多独立 cells。Pearson 用原值，Spearman 用平均秩 Pearson；本数据无 ties，另用 Python standard-library correlation 与排序秩交叉核对，结果一致。没有计算或发明 significance threshold。

| subset | n | d 中位数（pixels） | d 范围（pixels） | d 中位数（degree） | mean Δ | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 22 | 1.6488 | 0.2211–7.7068 | 0.08888 | 0.012958 | 0.8273 | 0.7267 |
| MC_OFF | 4 | 2.1458 | 1.4790–2.8622 | 0.11567 | 0.011331 | 0.8453 | 0.8000 |
| MC_ON | 5 | 1.6853 | 1.1491–7.7068 | 0.09085 | 0.032612 | 0.9852 | 0.9000 |
| PC_OFF | 4 | 1.4581 | 0.6850–4.7057 | 0.07860 | −0.001278 | 0.9443 | 0.2000 |
| PC_ON | 9 | 1.2976 | 0.2211–5.6408 | 0.06995 | 0.009090 | 0.6995 | 0.8333 |
| all except 68#10 | 21 | 1.6124 | 0.2211–5.6408 | 0.08692 | 0.008528 | 0.6222 | 0.6857 |

全体 d 的 Q25/Q75 为 1.1549/2.5283 pixels，即 0.06226/0.13629°；均值 2.1401 pixels=0.11536°。6/22 的 LN effective center 距原点超过相应 AC disk 半径，但这不代表 H1 经图传播后对该位置完全无响应。Canonical 在6/22 cells的 NLL优于LN。

offset 最大的5个依次为 `68#10, 70#1, 69#21, 68#3, 70#15`；gap 最大的5个为 `68#10, 69#21, 68#3, 68#7, 67#7`，两组重叠3个。按 offset 上半组（11 cells）看，gap 最大5个均在其中；这个排序描述不能代替因果检验。`70#1` 的第二大 offset 并未对应第二大 gap，`70#15` 也只呈中等 gap，关系不是一一对应。

### 5.1 指定代表与冻结中心一致性

| cell | 本轮关注点 | 已保存4个 inner 候选的 radial offset（pixels） | final 与候选中心距离（pixels） |
|---|---|---:|---:|
| 68#10 | 最大 offset=0.41545°，最大 Δ=0.105992；LN中心 σ≈2.023 pixels，靠近crop上侧，有限窗口截断也应考虑 | 6.9095–6.9268 | 0.8316–0.8480 |
| 68#3 | offset=0.15429°，Δ=0.032479；作者 notebook 也有非零同方向中心 | 2.6700–2.6705 | 0.2102–0.2121 |
| 68#7 | offset=0.13736°，Δ=0.028511；超过PC的0.06° BC半径；作者中心相近 | 2.4163–2.4166 | 0.1459–0.1475 |
| 69#21 | offset=0.25366°，Δ=0.034327；没有可对应的独立定位数据 | 4.7294–4.7342 | 0.0497–0.0526 |
| 67#4 | Canonical优势最大的cell：Δ=−0.014799，offset仍有0.09682°，说明偏移不是预测优劣的充分判据 | 1.5160–1.5225 | 0.2831–0.2906 |
| 70#1 | offset=0.30407°、Δ=0.020564；同样有有限窗口/宽度解释的不确定性 | 5.6076–5.6114 | 0.0636–0.0821 |

17/22 cells 的4个已保存 inner 候选均超过1 pixel。指定4个疑似错位代表的候选方向与 final 的 cosine 均高于0.999。这只描述同 seed、相同数据体系、不同 regularization 与 refit 数据范围下的一致性，**不是独立重复**。全部88个结果已保留，未为显示一致性挑选某个 candidate。没有新增训练来检验跨seed或跨时间段稳定性。

LN与Canonical的模型family不同：LN有两个自由的60-bin temporal kernels、可学习宽度与共同中心；Canonical有固定空间basis、受约束动态路径与状态递归。LN的60-bin kernel、Canonical的16-bin basis/RF报告窗口和递归状态记忆不能混为同一时间长度。center可能与宽度、temporal kernel、gain、history、裁剪范围或其他失配共同变化。小group中的相关系数尤其不稳定，例如PC_OFF的Pearson与Spearman分歧很大。不能把全部平均NLL差距归因于空间配准。

## 6. Consequences for RF/circuit interpretation

下表为**从实际forward推导的合理机制**；本轮没有重新计算RF、拟合参数变化或偏移干预，故没有直接证据表明这些补偿已经发生。

[model.py](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/model.py) 的 `forward_sequence` 依次执行：

\[
C'=C-a_{H1}G^\top L_{H1}(GC),\quad
B_{direct}=W_{BC}\Phi_{disk}(C'),\quad
B_{broad}=W_{BC}\Phi_{broad}(C'),
\]
\[
I=g_{BC}(B_s+B_t)-g_{AC}[\alpha_s L_{AC,s}(B_{broad,s})+
\alpha_t L_{AC,t}(B_{broad,t})],\quad z=\mathrm{RGCState}(I,history).
\]

其中 Φ 含固定 spatial basis 与可学习 temporal basis，`W_BC` 同时用于direct与broad两种view；delay包含在相应时序滤波中。这是结构示意，不把它当无状态的完整输出公式。

| 量 | 固定中心发生失配时可能的影响 | 推断边界 |
|---|---|---|
| global effective RF | 真实空间投影可能不在当前固定居中basis可表达范围中，影响RF位置、形状、norm及与外部RF的比较；标量动态再改变输入依赖的Jacobian | [effective_rf](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/rf_effective.py) 测的是固定geometry下logit对stimulus的导数。模型RF“居中”不是数据证明RF居中 |
| H1 RF | H1从整个cone图汇聚后经居中BC/AC读出。H1 amplitude/τ/delay可改变surround贡献，可能帮助利用偏移位置与中心位置的时空相关 | 它不能把固定graph或读出中心直接平移。H1是上游共享路径，normal−H1-clamp RF含其经BC和AC传播的总效应，不能当独立可加的一根输出支路 |
| direct-BC RF | 5/9个直接输入pixels及Gaussian中心固定，无法让直接局部disk追随真实偏移；空间mode mixture和gain可改变幅度、有效宽度 | 对原始cones的BC路径RF仍含H1预处理，不等于只在BC mask内非零 |
| AC RF | fixed broad disk从相同BC encoder取样，潜在错位可改变center/surround对输入内容的分工；可能通过AC gate、gain、滤波时间改变净贡献 | AC没有独立平移或独立spatial encoder；其改变也不能唯一归因于真实AC生理变化 |
| learned BC/AC gain | 输入投影偏弱/偏强时，exp(log_gain)与相消可以补偿总current，改变pathway相对幅度和RF norm | 本轮没有将offset与gain变化做因果检验，不能称已发生gain补偿 |
| temporal τ / delay | 自然movie中的空间与时间相关使错误位置的signal有可能用不同temporal mixture、τ或delay近似目标响应 | temporal机制无法一般性替代二维空间平移；时间参数仍受各自bounds约束，不将τ、delay、lag窗口与history shift合并解释 |
| illusion probe response | probe放置于crop中心时，它相对真实RF的位置可能错误；训练时的补偿在新空间图案下可能改变响应或pathway对照 | 当前结果仍是指定中心、指定history与冻结参数下的模型response，不能升格为真实cell或知觉现象；本轮未运行新的probe |

因此“prediction接近，但pathway decomposition被扭曲”在结构上可能：相互抵消的BC/AC贡献和H1预处理能保持相似总logit，却分配不同内部贡献。不过这只是可辨识性风险，**发生与否、方向及幅度均为UNVERIFIED**。现有路径RF定义也需区分 [pathway_base_rfs](D:/PythonProject/retina_rf_SNN/models/mechanistic_retina/pathway_rf.py) 的current Jacobian和[effective_pathway_rf](D:/PythonProject/retina_rf_SNN/evaluation/mechanistic_retina/pathway_decomposition.py) 的输出敏感度加权Jacobian，不能用一张居中RF图反证空间配准没有问题。

## 7. GO / NO-GO / UNKNOWN

最终决策：**GO: targeted spatial-alignment experiment justified**。

理由限定为：多个cell的effective offsets达到输入采样和BC support尺度，在冻结inner candidates/final refit之间保持非零；官方模型本身保留非零中心、metadata未支持统一精确零点；prediction gap存在描述性关联，且实际forward提供明确的潜在机制解释限制。符合“值得做针对性实验”，不等于支持某个新architecture。

独立生理中心仍是 **UNKNOWN**；原论文是否明确写过定位不足为 **UNVERIFIED**；centering对当前NLL、RF、pathway的实际因果贡献为 **UNVERIFIED**。这些未知不妨碍做受控的模型配准敏感性实验，但阻止生理RF恢复结论和“已解释预测差距”的结论。

## 8. 仅设计最小后续实验，本轮不执行

**首选设计：同一Canonical两个配准条件的配对比较。** 对照center=(0,0)；实验条件仅将cell center换为独立定位metadata经验证坐标变换后的固定值。Gaussian basis和BC/AC masks按同一center重新生成，support半径、σ、H1 cone graph和其余architecture保持现有定义。独立中心未获得或不能对应当前cell时，不用作者model参数冒充该metadata。此方案新增的训练自由度为 **0**，只改变外部测量的几何输入。

如果没有独立定位，只能做训练数据驱动的配准实验：将每cell的 `(δx,δy)` 明确计作 **2个新增位置自由度**，center候选仅由原training内的数据产生，在既有training-only inner-dev上选择；中心搜索范围、候选数、优化预算必须在接触后续评估结果前冻结。当前support是硬disk且basis为构造时buffer，不能仅把center tensor设为requires_grad就宣称实现正确的可训练translation；最小可审查方式是对固定center候选做profile comparison，每个候选重新构造同一定义的disk/basis，不改变production模型。本轮不指定由当前validation挑出的最佳offset，不实现搜索器，不做fit。

两种设计任选其一，不同时铺开。共同预注册内容：

1. 同一22-cell集合，配对seed/初始化规则与训练预算；相同stimulus、L+M front-end、原split、Bernoulli loss、history、mask和trial处理。固定现有τ/delay范围及其余选择协议，不增加baseline、illusion目标或额外调参。
2. 原validation禁止用于选择center、边界、步数或停止条件；原validation已暴露的事实无法通过改名消除，任何最终一次性比较仍标为development evidence。
3. 保持现有crop。候选若导致支持为空、AC无外延或靠边裁剪，按预注册几何可行性规则处理并报告；不能为了该cell静默扩大crop或改变输入。独立中心超出现有可行区域时，该cell的此最小实验记为无法评估。
4. primary量为配对 `NLL_aligned−NLL_zero`，按cell等权；同时按原4组完整报告。预先规定：仅training NLL降低不算成功；未用于center选择的评估部分上平均差值<0才支持预测改善，≥0则不支持。若收益只来自`68#10`而其余21-cell均值不改善，只允许cell-specific结论。这个符号标准是下一实验的操作判据，不是当前相关系数的显著性阈值，也不代表统计确定性。
5. 最小机制读出固定为现有global/pathway RF、BC/AC gain、H1 amplitude和τ/delay，不以其数值方向选模型。若prediction相近但这些量明显改变，只报告配准条件下的机制敏感性；没有独立RF时仍不能宣称生理解释改善。只有取得独立RF，才可预注册与其位置误差的配对比较。
6. 成功只授权“此模型在此评估条件下对alignment敏感/改善”的结论；失败后停止，不继续移动中心、换loss或根据原validation改搜索范围。

### Artifact与复核记录

- `per_cell_alignment.csv`：22个最终refit中心、单位换算、双方NLL与checkpoint路径。
- `alignment_vs_prediction.csv`：整体、4组和去掉`68#10`后的分布/相关性。
- `frozen_inner_center_consistency.csv`：88个已存inner候选，只作补充一致性证据。
- `official_model_centers.csv`、`independent_evidence.json`：作者模型中心的来源/坐标变换，catalog与示例RF可用性。
- `evidence_manifest.json`：实际读取文件的SHA256、22个identity/geometry checks、22组冻结NLL复算及最终只读核对。
- `audit.py`：只读数值分析的复现入口。

工具限制记录：首次写manifest被自动审批误判为违反只读，核对用户明确要求创建该文件和脚本写入边界后，同一操作获准。初次SciPy导入因现有NumPy ABI不兼容失败；未修复环境，最终统计使用NumPy，并用标准库独立核对。Fig2读取因其为annex pointer失败；Fig3只读提取时openpyxl提示未知扩展，但没有保存/重写workbook，SHA256未变。没有因这些问题修改已有文件。
