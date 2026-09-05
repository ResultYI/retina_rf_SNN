# Schottdorf–Lee timing contract final confirmation

日期：2026-08-31。范围：两个原始 recording、官方说明、现行 spike parser / movie loader，以及无模型的时间映射核对。

**最终判定：UNVERIFIED。** Spike timestamp 的单位、live-relative 使用、不重复减 Video Start(s)、六次重复的 movie identity 和 150 Hz 数值 binning 均有直接证据；但原始 MPEG 的**顺序解码内容边界**与 README / loader 的首帧编号相差一帧。正确的 acquisition `t=0` 应对应哪个 decoded frame 尚不能最终确认。发现该疑点后，只完成最小顺序读取复现并停止；未修 parser、未运行模型、未训练。

## A. 数据文件、recording / cell ID 与路径

工作区根目录：`D:/PythonProject/retina_rf_SNN`。以下链接为绝对路径。

| 项目 | 文件 / 身份 | 直接证据 |
|---|---|---|
| 连续 10 min | [lSS01300.txt](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/data/lSS01300.txt) | 官方 README 第 86 行：MC on，cell `70#34`，10 min |
| 6×1 min | [lSS01299.txt](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/data/lSS01299.txt) | 官方 README 第 85 行：MC on，cell `70#34`，6×1 min，eccentricity 5.66° |
| 官方说明副本 | [README.md](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/README.md) | 原始数据仓库 README，不是本项目研究总结 |
| 实际可解码 movie | [1x10_256.mpg](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_macaque/1x10_256.mpg) | 265,338,036 bytes；MD5 `d64bdae05eb07895a8f30cda287c5a74` |
| 官方 movie annex 标识 | [stimuli/1x10_256.mpg](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/stimuli/1x10_256.mpg) | 此副本是 65-byte pointer，不是 movie payload；内容 `/annex/objects/MD5-s265338036--d64bdae05eb07895a8f30cda287c5a74`，与实际 movie 的 size / MD5 一致 |
| 官方实现参考 | [make_movie_MC.py](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/run_model_HPC/make_movie_MC.py:42)、[library.py](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/retinatools/library.py:145) | 前者 `dt=1000/150.`；后者使用顺序 `VideoCapture.read()` |

官方数据 DOI：[10.12751/g-node.xage77](https://doi.gin.g-node.org/10.12751/g-node.xage77/)。该 DOI 页面确认本数据及代码随 Schottdorf / Lee 2021 论文发布。其指向 [官方 repository](https://gin.g-node.org/Manuel/Macaque-ganglion-cells) 和 [DOI archive](https://gin.g-node.org/doi/Macaque-ganglion-cells)。本轮 DOI 页面可读取，GIN repository / archive 页面返回 HTTP 403；详细证据取自本地原始仓库文件。未据此声称本轮重新验证了远端 README 的 byte identity。

当前 shared-BC run 的 movie 路径由 [run.py 第 36 行](D:/PythonProject/retina_rf_SNN/output/real_data/schottdorf_canonical_v1_shared_bc_development_22cell_20260830/run.py:36) 直接指定为上表实际 movie。此处仅核对输入路径，未读取实验指标或 checkpoints。

### 本轮文件指纹

| 文件 | SHA256 |
|---|---|
| 原始 README | `aec20c21a4b00d58afaa4144f44c77b8b850f555871e20dc57f1562b4a831ef2` |
| lSS01299.txt | `f26bdf244b693b5dff575392e2a886878f3ad677baaed185b500db12ad17005b` |
| lSS01300.txt | `ade77687b9c32bf2ad6d80bb92a1dc29617fb860a643b770e793c43282c2afe6` |
| data/schottdorf_lee_spikes.py | `04993cd009000dab362c352530e5a68d5dfda39403785e25aa6a53dcd9c2b6d4` |
| data/schottdorf_lee_2021.py | `ae07cb57443ce95fcef2208060638c1cc12168ec9e1a1f3f9e58c82e24ddf764` |
| data/schottdorf_lee_multirecording.py | `3b6b1ae2ddb2c35d1c9ae133017ff5016b79960d56a194f1722b0079ec1716db` |

## B. 官方说明与原始字段证据

### B1. README 明确规定的合同

| 位置 | 内容事实及边界 |
|---|---|
| [README:36](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/README.md:36) | 所有数据时间分辨率为 **0.1 ms**。 |
| [README:38](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/README.md:38) | 10 min spike time list 已相对于 **live video 起点**，且已校正 video 与 acquisition computer 之间的小 clock-rate difference。Video Start 是随 cell 变化的 internal control。 |
| 同上 | `Spikes/5 sec` 的 0 行是 live 前的 spike count；1–120 行是后续 5 s epoch counts；121 行被说明为 live 结束后的 count。此表不是逐 spike timestamp。 |
| [README:40](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/README.md:40) | 6×1 min 格式相似；**10 min video 的第一分钟重复六次**。第一次前有可变 blank delay，重复之间有约 5 s blank。第 7 列为六次重复后的 maintained activity。 |
| [README:94](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/README.md:94) | 描述为 751 blank frames，live 从第 752 帧开始，共 90,000 frames，之后 750 blank frames；实验播放 **150 frames/s**，为 acquisition rate 的三倍；audio timing pulses 用于 acquisition synchronization。 |
| [README:96](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/README.md:96) | `6x1_256.mpg` 结构相同，但 live 部分为 9,000 frames。 |

README 没有给出 `Video Start(s)` internal clock 的完整时钟零点定义，也没有列出六次重复各自的绝对 acquisition start。不能把一个 `Video Starts` 标量当成六个 segment start，不能自行拼出严格 `65 s × repeat` 的连续 acquisition 时间线。

已检查的官方 Python model 代码读取 movie / 已处理 firing-rate 文件；未从这些代码取得生成原始 spike export 或校正 timestamp 的公式。因此不声称已独立重建 clock-rate correction。

### B2. 原始字段

**lSS01300**：第 1 行 `Total spikes\t24001`；第 2 行 `Video Start\t41904`；第 128 行 `No\tTime`。第 129 行时间 `−41602`，第 238 / 239 行分别为 `−39 / 36`。

**lSS01299**：

```text
line 1: Total spikes  15590
line 2: Video Starts  46232
line 4: Spikes per rpt
line 5:              2570 2497 2522 2492 2452 2476 581
line 23: Spikes times
line 24: 0           -45918 -49883 -49252 -49562 -49678 -48926 -48786
```

上述列按 whitespace 展示；实际文件为 tab-separated。第一字段是 spike ordinal，之后才是 7 个 spike-time columns。

### B3. 原始辅助 count 表的限制

对 lSS01299，7 个 time columns 按 `q∈[50000(k−1),50000k)` 重数，与原始 `Spikes/5 sec` 的每个正 epoch count 均一致；负时间 counts 也与第 0 行一致。此项直接支持各列各有 local zero，而不是把 column 2–6 当成连续 6-minute timestamps。

lSS01300 的辅助 count 表有原始文件内部不一致：其全部 count 行相加为 **23,380**，而 `Total spikes` 与 time-list 行数均为 **24,001**。epoch 1 的 header / time-list 重数为 `258 / 258`，epoch 2 为 `232 / 233`，epoch 120 为 `196 / 197`。按公开时间单位，time list 的 `[0,600 s)` 事件数为 **23,374**；header epochs 1–120 相加为 **23,255**。另外存在 517 个 `t≥600 s` 的时间戳，最后一个为 634.9805 s。

当前 parser 不使用该辅助 histogram 生成 target。这里记录该不一致，**不以其反推 offset、不修数据、不诊断其生成原因**。该辅助表不能作为全部时间对齐正确的独立证明；其不一致来源为 UNVERIFIED。

## C. 10 min recording 的时间映射

设原始整数 timestamp 为 `q`：

```text
t_ms = 0.1 q
t_s  = q / 10000
live 条件：0 ≤ q < 6,000,000
150 Hz bin j = floor(150 t_s) = floor(3q/200)
bin 区间：[j/150, (j+1)/150) seconds
当前 loader movie decoded index（零基）：m = 751 + j
按 README 的一基 frame 编号：m + 1 = 752 + j
```

原始 `Video Start=41904` 仅保存为 metadata，**没有再次相减**。以文档的时间单位表示该标量是 4190.4 ms；这不是要求再从 live-relative spike list 减去的延迟。

直接调用 `parse_recording_spike_trials(lSS01300)` 得到一个 trial、23,374 个 live timestamps，逐元素等于原始整数筛选后乘 0.1。负时间 110 个被排除，`t≥600 s` 的 517 个被排除。

例如 `q=36 → t=3.6 ms → j=floor(0.54)=0 → 当前 movie index 751`。若错误再次减 Video Start，会得到 `−4186.8 ms` 并丢弃；当前实现没有这样做。

**上述 spike→bin 算术已核对；其中 `m=751+j` 是否等于 acquisition live zero 的正确图像映射，因 G2 的原始 movie 边界冲突保留 UNVERIFIED。**

## D. 6×1 min recording 的时间映射

column 1–6 各表示一次重复。每一列的 `q=0` 对应该 repeat 的 local live 起点；此解释由 README 的“格式相似”、首分钟重复说明、各列负时间 / 约 60 s 正时间范围和原始 5 s counts 共同支持。

```text
repeat r ∈ {1,2,3,4,5,6}
t_local_s = q / 10000        （不减 Video Starts；不减 60r 或 65r）
live 条件：0 ≤ q < 600,000
j = floor(3q/200),  0 ≤ j < 9,000
六个 repeats 都映射到同一个 10 min movie 的第一分钟
当前 decoded movie index m = 751 + j
```

这不是把 10 min movie 切成六个不同的一分钟，也不是用第 2 repeat 读取第 2 分钟。`_make_trial_split` 对同一 segment 给所有 repeats 使用相同 `drive[segment]`，spikes 则选择 `counts[trial, segment]`。

| Spike-time column | 原始 count | 最小 / 最大 q | q<0 | 0≤q<600000，parser 保留 | q≥600000 |
|---|---:|---:|---:|---:|---:|
| 1 | 2570 | −45918 / 652160 | 93 | 2417 | 60 |
| 2 | 2497 | −49883 / 651587 | 79 | 2368 | 50 |
| 3 | 2522 | −49252 / 651710 | 85 | 2382 | 55 |
| 4 | 2492 | −49562 / 651930 | 79 | 2350 | 63 |
| 5 | 2452 | −49678 / 651185 | 75 | 2304 | 73 |
| 6 | 2476 | −48926 / 649917 | 79 | 2340 | 57 |
| 7 maintained activity | 581 | −48786 / 249678 | 78 | 不建立 movie trial | 不适用 |

六个 parser trials 共 14,161 个 live events，均与原始列筛选 / 换算逐元素一致。第 7 列即使有正时间也不会变成第七个 movie trial。

例如 repeat 2 的 `q=599980 → 59.998 s → j=8999`；repeat 6 的 `q=44 → 0.0044 s → j=0`。下一 repeat 的时间轴从自己的列重新开始，不由上一列的时间戳数值推算。

## E. 第 7 列与 Video Start(s)

| 字段 | 含义 / 单位 | 当前使用 | 未确认的边界 |
|---|---|---|---|
| `Video Start`，10 min | live video start 的 internal control；此例整数 41904，按公开时间单位为 4190.4 ms | 读取、保存；不对 spike times 再加减 | internal acquisition clock 的绝对零点、原始 correction 参数没有公开在该字段中 |
| `Video Starts`，6×1 min | 相似格式中的 internal start-control 标量；此例 46232，即 4623.2 ms；不是六个 starts 的数组 | 读取、保存；不对任何列重复施加 offset | 六个 repeat 的绝对 acquisition start、精确 inter-repeat blank 时长不能由此单值恢复 |
| `Spikes per rpt` 的第 7 count | maintained-activity block 共 581 个 events；单位是 spike count | 参与原始 count 总数一致性检查 | 不是 offset、frame index 或 movie duration |
| `Spikes times` 的第 7 time column | 六次重复后的 maintained activity；整数时间单位 0.1 ms；本例范围 −4878.6 至 24967.8 ms | 解析用于检查，然后 `trial_times[:6]` 排除 | 该 maintained block 的 local zero 对应哪个绝对 acquisition 时刻没有单独定义，标记 UNVERIFIED；当前 fitting 不需要使用它 |

“第 7 列”这里指去掉最前面的 spike ordinal 后的第 7 个**时间列**，即 TSV 的第 8 个字段。不能把第 7 个物理 TSV 字段误当 maintained activity；那个字段是 repeat 6。

## F. 当前 parser / loader 实现位置

| 文件位置 | 代码事实 |
|---|---|
| [schottdorf_lee_spikes.py:26](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_spikes.py:26) | `parse_recording_spike_trials` 入口；第 33 / 37 行分别读取 Video Start / Video Starts。 |
| [同文件:39](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_spikes.py:39) | 只使用 `trial_times[:6]`；各 repeat 独立裁剪为 `[0,60000 ms)`。 |
| [同文件:93](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_spikes.py:93)、[126](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_spikes.py:126) | 原始整数仅乘 `0.1` 得到 ms；无 Video Start subtraction、无新的 clock-rate rescaling。 |
| [同文件:115](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_spikes.py:115)、[120](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_spikes.py:120) | 7 列 count 校验；跳过 row ordinal 后读取 7 个字段。 |
| [同文件:138](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_spikes.py:138) | `_live_times` 使用左闭右开范围。 |
| [schottdorf_lee_multirecording.py:176](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_multirecording.py:176) | `_bin_trial` 为 `floor(times_ms * 150 / 1000)`，然后 bincount。 |
| [同文件:80](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_multirecording.py:80)、[193](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_multirecording.py:193) | 六次重复各自 bin；禁止 repeat 窗口超过 9000 frames；各 trial 复用同一 segment stimulus。 |
| [schottdorf_lee_2021.py:14](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py:14)、[178](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py:178) | `_LIVE_START_FRAME=751`；从 0 开始顺序 read；前 751 个 decoded frames 计入 background，从 decoded index 751 写入 `live[0]`。 |

Movie container 报告 **50 fps**，官方实验实际播放为 **150 fps**。当前 loader 使用 frame ordinal 和官方 150 Hz，而不是按 container PTS 的 50 fps 建立 spike 时间轴；这一 rate 选择与官方说明和 `make_movie_MC.py:42` 一致。

## G. 逐项核对与最小疑点复现

### G1. 原始 timestamp → repeat / segment → local time → bin

此表中的 segment 为 **1 s 的算术分段**，用于核对 `j=150k+b`，不改变项目 split。repeat 是独立 movie repetition，不能与 segment 混用。表中 movie index 为**当前 parser 映射**，不是对有争议的 physical t=0 的最终认证。

| Recording；原始行；时间列 | q | repeat / 1 s segment k | live-relative t(s) | segment-local t(s) | 手算 j / segment-bin b | 当前 movie index（零基） | 实际 `_bin_trial` 非零 index |
|---|---:|---|---:|---:|---|---:|---|
| 01300；238；1 | −39 | pre-live | −0.0039 | — | −1 / — | 排除 | [] |
| 01300；239；1 | 36 | 1 / 0 | 0.0036 | 0.0036 | 0 / 0 | 751 | [0] |
| 01300；303；1 | 9000 | 1 / 0 | 0.9000 | 0.9000 | 135 / 135 | 886 | [135] |
| 01300；304；1 | 10090 | 1 / 1 | 1.0090 | 0.0090 | 151 / 1 | 902 | [151] |
| 01300；496；1 | 49228 | 1 / 4 | 4.9228 | 0.9228 | 738 / 138 | 1489 | [738] |
| 01300；497；1 | 51388 | 1 / 5 | 5.1388 | 0.1388 | 770 / 20 | 1521 | [770] |
| 01300；23612；1 | 5999328 | 1 / 599 | 599.9328 | 0.9328 | 89989 / 139 | 90740 | [89989] |
| 01300；23613；1 | 6000239 | post-live | 600.0239 | — | 90003 / — | 排除 | [] |
| 01299；116；1 | −127 | repeat 1 pre-live | −0.0127 | — | −2 / — | 排除 | [] |
| 01299；117；1 | 406 | 1 / 0 | 0.0406 | 0.0406 | 6 / 6 | 757 | [6] |
| 01299；174；1 | 10325 | 1 / 1 | 1.0325 | 0.0325 | 154 / 4 | 905 | [154] |
| 01299；2533；1 | 599126 | 1 / 59 | 59.9126 | 0.9126 | 8986 / 136 | 9737 | [8986] |
| 01299；2534；1 | 600086 | repeat 1 post-live | 60.0086 | — | 9001 / — | 排除 | [] |
| 01299；103；2 | 291 | 2 / 0 | 0.0291 | 0.0291 | 4 / 4 | 755 | [4] |
| 01299；2470；2 | 599980 | 2 / 59 | 59.9980 | 0.9980 | 8999 / 149 | 9750 | [8999] |
| 01299；2471；2 | 600049 | repeat 2 post-live | 60.0049 | — | 9000 / — | 排除 | [] |
| 01299；168；3 | 10009 | 3 / 1 | 1.0009 | 0.0009 | 150 / 0 | 901 | [150] |
| 01299；103；6 | 44 | 6 / 0 | 0.0044 | 0.0044 | 0 / 0 | 751 | [0] |
| 01299；2442；6 | 599996 | 6 / 59 | 59.9996 | 0.9996 | 8999 / 149 | 9750 | [8999] |
| 01299；2443；6 | 600033 | repeat 6 post-live | 60.0033 | — | 9000 / — | 排除 | [] |

上述每一行均真实调用当前 `_bin_trial` 对该单个 timestamp 重算，不是只写出公式。检查还覆盖其余 repeats 的起点、1 s / 5 s 边界、60 s 尾部；所抽查数值 bin 无 mismatch。完整 trial parser 输出与直接原始整数解析后换算逐元素相同。

### G2. 触发停止条件：README / movie decoder 的首帧冲突

运行环境：`D:/anaconda/python.exe`，Python 3.12.7，OpenCV 4.11.0，backend FFMPEG。movie MD5 与官方 annex 标识一致。

先随机 seek 发现疑点，随后**只做一次同 loader 方式的从文件起点顺序读取**，不调用模型。排除了“仅随机 seek 定位不精确”这个解释。三个候选来源的证据状态：

1. 错误 movie payload：size / MD5 匹配官方 annex，未获支持。
2. 单纯随机 seek 偏差：顺序读取仍在 decoded index 750 出现首个明显非 blank 图像，不能解释全部差异。
3. README 编号、导出 / 解码帧排列或 acquisition-trigger→frame 对应关系存在差异：仍待原始 trigger / 作者导出约定确认。

| 顺序 decoded index，零基 | OpenCV next index | pixel min / max | pixel std |
|---:|---:|---|---:|
| 748 | 749 | 74 / 78 | 0.433593414 |
| 749 | 750 | 74 / 76 | 0.293150985 |
| **750** | **751** | **0 / 167** | **10.890591870** |
| **751** | **752** | **10 / 170** | **10.742605870** |
| 752 | 753 | 7 / 197 | 11.045499976 |
| 753 | 754 | 7 / 199 | 11.090112981 |

从 0 开始的 754 帧顺序扫描中，首个 `max−min>20` 的 frame index 为 **750**。此阈值只区分实际近常数 blank 与明显图像，用于最小复现，不是新的 model metric 或 probe。frame 750 与 751 数组不同。

当前 loader 首个 live index 固定为 **751**。另调用 `_load_calibrated_lm_drive(movie, 2, existing_config)` 并用逐帧 `_pooled_lm_signal` 交叉核对，确认 `drive[0:2]` 对应 decoded 751、752。没有改变 loader 参数或 Canonical V1。

所以：**loader 跟随 README 的字面编号，但相对 decoded image-content onset 跳过了一帧；这是真实可复现的差异。** 仅凭图像内容不能最终证明 acquisition 已校正 spike 的 `t=0` 必须对应 decoded 750，因此此时不把“应改成 750”视为已证实修复。150 Hz 下候选差异为 **1 frame = 6.6666667 ms**。

最小复现，在项目根目录用现有 Python 运行；无需模型 / checkpoint：

```python
import cv2
from data.schottdorf_lee_2021 import _LIVE_START_FRAME

p = 'data/real/schottdorf_lee_2021_macaque/1x10_256.mpg'
cap = cv2.VideoCapture(p)
for i in range(754):
    ok, frame = cap.read()
    assert ok
    if i >= 748:
        print(i, int(frame.min()), int(frame.max()))
cap.release()
print('parser first live decoded index:', _LIVE_START_FRAME)
```

预期复现：749 仍为近常数 blank，750 已出现 image，parser 常量为 751。本轮只保存此证据，不尝试其他 decoder、audio alignment 或修改 offset。

## H. 最终逐项判定

| 检查项 | 判定 | 精确范围 |
|---|---|---|
| timestamp reference frame | **PASS** | README 明确 10 min time list 已 live-relative / clock-corrected；6 repeats 按列 local-relative 的解释与官方格式、负时间及逐 epoch counts 一致。具体 live zero 对应 decoded 750 还是 751，另列为未决。 |
| `Video Start(s)` offset handling | **PASS** | 当前 parser 只记录该 internal-control 标量，没有再次从已对齐 timestamps 减去它；无额外 spike offset。 |
| 6×1 min mapping | **PASS** | 六个独立 repeat 的第一分钟映射、`[0,60 s)` 截取、第 7 maintained column 排除均与原始说明一致；首帧绝对 anchor 的共同疑点未因此消失。 |
| 150 Hz binning | **PASS** | 150 Hz 是实际播放 rate；dt=1/150 s，tick→floor bin 与手算一致；不把 movie container 的 50 fps 当实验时间轴。此 PASS 仅指 rate / bin 算术，不认证 frame-zero anchor。 |
| overall timing contract | **UNVERIFIED** | 原始 movie 顺序解码内容起点 750，与 README / loader 起点 751 不一致；无法最终认证 spike `t=0`→movie decoded index。 |

未确认的额外 metadata 细节：Video Start(s) 的绝对 internal-clock epoch、第 7 maintained block 的绝对起点、10 min 辅助 histogram 不一致来源。均未自行补全，也未参与新的 offset 计算。

## I. 重训与停止边界

- 本轮不满足“全部 PASS”，因此**不能给出“不需要因 timing contract 重训”的最终确认**。
- 未确认应修哪个 offset，也未确认重训必要性；**不修改 parser、不重训**。
- 若后续证实 acquisition live zero 应对齐 decoded 750，则采用现行 `_LIVE_START_FRAME=751` 的 10 min 和 6×1 min data fitting 都有候选一帧 stimulus / spike 配对偏移；本轮实查直接涉及 `lSS01300` / `lSS01299`、cell `70#34`。同 loader 下的 22-cell 数据与由其训练的 artifacts 属于潜在影响范围，未逐 cell 扩展确认，也未估算 prediction 或 mechanism 指标变化。
- 该候选偏移还会使当前 background 计算包含 decoded frame 750；这仅是现行 `frame_index<751` 分支的条件性影响，不是本轮提出的新修复。
- **唯一下一步 blocker：取得足以区分 decoded frame 750 / 751 的官方 acquisition-trigger / corrected-timestamp→movie-frame-zero 对应证据。** 可用证据应来自原始 audio timing pulses 的官方解释或作者导出约定，而非现有训练结果。是否开展这一步，等待用户下一条指令。

执行记录：仅读取上述文件、执行 parser / binning 与最小 video decode；未加载模型、未训练、未改 loss / split / protocol。新增文件仅本 Markdown 报告。只读核对采用 programming 规则；异常复现按 debugging 规则限制为单次顺序读取，没有修复动作或临时调试文件。
