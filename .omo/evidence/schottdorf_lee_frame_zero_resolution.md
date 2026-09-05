# Schottdorf–Lee frame-zero resolution

日期：2026-08-31。最终判定：**UNRESOLVED_EXTERNAL_EVIDENCE_REQUIRED**。

## 1. 范围与停止条件

本报告只处理 acquisition-corrected spike `t=0` 对应顺序解码 zero-based frame **750** 还是 **751**。不重做上一报告已关闭的 timestamp reference、Video Start(s)、6×1 min mapping 或 150 Hz binning 检查。

未训练、未加载 checkpoint、未比较模型结果、未修改 `_LIVE_START_FRAME`、parser、loader、Canonical V1、loss、split 或 protocol。只新增本报告及原始视频/音频检查产物。旧报告保持不变：

[schottdorf_lee_timing_contract_final_check.md](D:/PythonProject/retina_rf_SNN/.omo/evidence/schottdorf_lee_timing_contract_final_check.md)。

**音频存在，但缺少把某个同步脉冲边沿、实际显示帧及导出 spike t=0 绑定在一起的官方定义。到此停止本地 audit；不选定 750 或 751，不把 README 判为 off-by-one error。**

## 2. 检查对象、官方来源与访问边界

### 2.1 原始文件与本地官方仓库

| 对象 | 路径 / identity | 本轮检查 |
| --- | --- | --- |
| 实际 MPEG payload | `D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_macaque/1x10_256.mpg` | streams、顺序 frame/packet、前 16.2 s 音频及帧 745–755 |
| 官方本地仓库 | `D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository` | README、Git tree、可用历史、Python/shell 源码及 notebook code cells |
| Movie annex pointer | `.../schottdorf_lee_2021_repository/stimuli/1x10_256.mpg` | Git 中是 `/annex/objects/MD5-s265338036--d64bdae05eb07895a8f30cda287c5a74`，不是另一个视频副本 |
| 官方 README | [README.md:94](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/README.md:94) | blank/live 编号及 audio timing 声明 |
| 作者分析库 | [library.py:145](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/retinatools/library.py:145)、[library.py:322](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/retinatools/library.py:322) | 顺序读帧和离线 model/data comparison 时间坐标 |
| 作者模型计算脚本 | [make_movie_MC.py:65](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/run_model_HPC/make_movie_MC.py:65)，同目录 `make_movie_PC.py`、`make_movie_BO.py`、`run_python.sh` | 搜索生成/播放/同步代码；这些文件处理已有 stimulus 的模型响应 |
| 作者展示 notebook | [make_movie_from_files.ipynb](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/run_model_HPC/make_movie_from_files.ipynb)，zero-based code-cell 4、6、8 | 读取已有 MPEG 和模型数组，输出展示帧；不是原始 stimulus/acquisition 生成程序 |
| 作者拟合 notebooks | `D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/run_model/` 中 `BlueOn.ipynb`、`MC_off.ipynb`、`MC_on.ipynb`、`PC_Gon.ipynb`、`PC_Ron.ipynb`、`PC_off.ipynb`、`Piecharts.ipynb` | 只检索 code cells，未使用内嵌模型输出推断 t=0 |
| 当前 loader | [schottdorf_lee_2021.py:14](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py:14)、[178](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py:178) | 只确认条件性影响位置，没有修改 |

Movie 大小 265,338,036 bytes；MD5 `d64bdae05eb07895a8f30cda287c5a74` 已由上一轮直接检查匹配官方 annex identity，本轮使用同一路径原文件。没有转换或重新编码输入 MPEG。

官方仓库 origin 为 `https://gin.g-node.org/Manuel/Macaque-ganglion-cells.git`。本地 HEAD 为 `cffefb08c760f04c9a951da46061b361d2288e9b`，作者 Manuel Schottdorf，时间 `2024-03-08T01:55:46Z`，提交标题 `Added missing files`。本地仓库 `git status --short` 为空，`git rev-parse --is-shallow-repository` 为 `true`；`git log --all` 只有该提交。因此**不能以这个 shallow boundary 的 README diff 推断文字最初何时写入，或称已经检查完整历史**。

### 2.2 网络一手来源

- [官方 DOI landing page](https://doi.gin.g-node.org/10.12751/g-node.xage77/) 可读；其中 Browse Repository、Browse Archive 和 ZIP archive 是官方链接。
- [作者 repository](https://gin.g-node.org/Manuel/Macaque-ganglion-cells)、[DOI archive](https://gin.g-node.org/doi/Macaque-ganglion-cells) 及 raw README 的本轮在线读取受 403/抓取错误限制。
- 官方 commits API `https://gin.g-node.org/api/v1/repos/Manuel/Macaque-ganglion-cells/commits?limit=10`：网页工具失败，原生 HTTPS 请求 SSL connection failed；未获取更早历史。
- DOI ZIP `https://doi.gin.g-node.org/10.12751/g-node.xage77/10.12751_g-node.xage77.zip`：网页工具不支持 ZIP；原生 HEAD 请求因本机 Schannel `SEC_E_NO_CREDENTIALS` 失败。**未将完整 DOI ZIP 内容称为已检查。**
- [论文正式页面](https://physoc.onlinelibrary.wiley.com/doi/abs/10.1113/JP281200) 可读，Data availability statement 将 raw data/model code 和同步信息指向该 DOI repository。Supporting Information 清单为 5 个模型展示视频和一个统计汇总 XLSX；该清单没有列出 acquisition-trigger/frame-number conversion 程序。未用展示视频判断 timing。
- [论文 PMC 正文入口](https://pmc.ncbi.nlm.nih.gov/articles/PMC8998785/) 直接读取遇 reCAPTCHA；一手正文索引中可见 Methods 关于 movie audio pulses 用于同步及 FFmpeg 提取图像的描述，但未得到帧 750/751 到 corrected t=0 的明确关系。不能把未完整取得的正文/历史宣称为穷尽排除。

仅以上官方/论文来源进入证据链。未使用第三方博客、第三方代理或模型 prediction/RF 结果。

## 3. 官方 frame-indexing / trigger 证据实际到哪一步

### 3.1 README

[官方本地 README:94](D:/PythonProject/retina_rf_SNN/data/real/schottdorf_lee_2021_repository/README.md:94) 的直接措辞包含：

> 751 blank frames
>
> live video (starting at frame 752, 90000 frames)

同一行说明 audio channel 提供 timing pulses，用于与 acquisition system 同步，模式按 5 s 重复。该句未定义：

- `frame 752` 是否等于某个具体 decoder 顺序输出数组的第 752 项，及是否应保留异常/重复 PTS 的图片；
- 哪个 pulse、哪种边沿用于定义 corrected spike t=0；
- pulse 与显示帧的开始/结束关系、播放设备 A/V offset；
- 空白帧的实际生成循环，以及原始 movie player 对边界图片的处理。

若把文字中的 752 理解为 one-based sequential decoded ordinal，会得到 zero-based 751；**这是当前 loader 的字面映射，不是新的 acquisition 同步证据**。本轮没有足够证据宣布 README 写错或正式认证上述映射。

### 3.2 作者代码

`retinatools/library.py:150–175` 的 MC 视频过滤函数从 `counter=0` 顺序读取已有 movie。`library.py:322–332` 的 `compare_to_file` 使用：

```text
time_cell = arange(...) * dt + 5000 - shift * 6.66
time      = arange(...) * dt
```

`run_model/MC_on.ipynb` code-cell 5、10、15 调用该函数时均传入 `shift=2`。这说明离线 comparison 还有一个显式 shift 输入；它不是原始 trigger-to-spike export 定义，不能将 `5000 ms` 或某个 shift 的取值直接当作 750/751 的裁决，更没有依据去用相关性选择 offset。

`run_model_HPC/make_movie_from_files.ipynb` code-cell 4 在循环外读过一帧，循环内再次 read 后存储；code-cell 8 注释 `startframe = 600 #1sec before video begins at 150fps`。这是配合模型数组绘制演示视频的代码：code-cell 6 读取 `Gon-19829503-frames.npy`，code-cell 8 保存展示 PNG。它既未生成原始 MPEG，也未记录实验 live trigger；循环外读取还意味着其内部数组 index 不能不加说明地当作原始 decoded ordinal。**未据该注释或数组下标选定 acquisition frame zero。**

在当前 Git tree 的 `.py/.md/.sh` 与上述 notebook code cells 中搜索 `trigger|synchron|audio|pulse|blank|frame.?number|751|752|VideoWriter|ffmpeg`，没有找到将导出 corrected t=0 绑定到某个 displayed frame 的原始 acquisition、movie-generation 或 playback 实现。这个结论仅限已取得的 tree，不含无法获取的更早历史/外部脚本。

## 4. 原始 MPEG streams

使用独立 FFmpeg CLI 顺序读取，不通过项目 loader，不 seek 到关键帧，不转码输入，不调用模型。

工具：`D:/ffmpeg-2025-04-14-git-3b2a9410ef-essentials_build/bin/ffprobe.exe` 和同目录 `ffmpeg.exe`；版本 `2025-04-14-git-3b2a9410ef`，libavcodec `62.0.101`，libavformat `62.0.100`。

完整输出：[streams.json](D:/PythonProject/retina_rf_SNN/.omo/evidence/schottdorf_lee_frame_zero/streams.json)。

| 字段 | video stream 0 | audio stream 1 |
| --- | --- | --- |
| Codec | MPEG-1 video (`mpeg1video`) | MPEG Layer 2 audio (`mp2`) |
| Stream ID | `0x1e0` | `0x1c0` |
| 格式 | 256×256、yuv420p、progressive、含 B frames | mono、32,000 samples/s、128 kbps |
| Time base | 1/90000 | 1/90000 |
| start_pts | 12363 | 5613 |
| start_time | 0.137366667 s | 0.062366667 s |
| 帧率字段 | avg_frame_rate=50/1；r_frame_rate=100/1 | 不适用 |

MPEG-PS 只有这两个 streams，没有第三个独立 data/timing-pulse stream。**audio stream 确实存在，并可解码出稀疏 pulse waveform。** 帧率字段、container 时间均只按原样记录；不拿它们替换已冻结实验 150 Hz 合同，也不将其等同 acquisition clock。

## 5. Frame 745–755：顺序输出、PTS/DTS、picture type

`n` 是从文件开头连续 decode 得到的 **zero-based frame-return / presentation-order ordinal**，不是压缩 packet 顺序、GOP timecode 或实验硬件 displayed-frame 编号。原始输出保存为 [first16s-frames.json](D:/PythonProject/retina_rf_SNN/.omo/evidence/schottdorf_lee_frame_zero/first16s-frames.json) 和 [video-packets-first16s.json](D:/PythonProject/retina_rf_SNN/.omo/evidence/schottdorf_lee_frame_zero/video-packets-first16s.json)。

所有时刻单位为 container 秒。Packet DTS 由 packet position 对应；n=752 的 position 缺失，使用唯一对应 PTS 的 packet。`frame.pkt_dts_time` 是 decoder 返回 frame 携带的字段，单独列出，**不把它误当压缩 packet 的原始 DTS**。

| n | PTS ticks | PTS s | packet DTS s | frame.pkt_dts_time s | Type | Keyframe | packet byte position | ffprobe GOP timecode 标签 |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 745 | 1353363 | 15.037367 | 15.037367 | 15.037367 | B | 0 | 2150430 | — |
| 746 | 1355163 | 15.057367 | 15.057367 | 15.057367 | B | 0 | 2152478 | — |
| 747 | 1356963 | 15.077367 | 15.017367 | 15.077367 | P | 0 | 2148382 | 00:00:24:28 |
| 748 | 1360563 | 15.117367 | 15.117367 | 15.117367 | B | 0 | 2160670 | — |
| 749 | 1362363 | 15.137367 | 15.077367 | 15.117367 | I | 1 | 2154526 | 00:00:00:00 |
| **750** | **1362363** | **15.137367** | **15.117367** | **15.137367** | **I** | **1** | **2162718** | — |
| **751** | **1364163** | **15.157367** | **15.157367** | **15.157367** | **B** | **0** | **2170910** | — |
| 752 | 1365963 | 15.177367 | 15.177367 | 15.177367 | B | 0 | 未给出 | — |
| 753 | 1367763 | 15.197367 | 15.137367 | 15.197367 | P | 0 | 2168862 | — |
| 754 | 1369563 | 15.217367 | 15.217367 | 15.217367 | B | 0 | 2177054 | — |
| 755 | 1371363 | 15.237367 | 15.237367 | 15.237367 | B | 0 | 2181150 | — |

该区间 frame duration 字段均为 1800 ticks=20 ms、repeat_pict=0、interlaced_frame=0。关键直接事实：

- n=749 和 n=750 是**不同图片、不同 packets，但 PTS 完全相同**。
- n=747→748 的 PTS 间隔为 40 ms；附近 GOP timecode 从 `00:00:24:28` 跳到 `00:00:00:00`。这里只报告标签，不解释为 acquisition clock。
- 因而不能在这里仅用 `round((PTS−start_PTS)×50)` 代替逐帧 ordinal，也不能由 PTS 等值推断实验播放时保留、跳过或合并了哪张图片。

### 5.1 与上一轮 OpenCV 顺序解码的辅助交叉检查

FFmpeg CLI 用 `select=between(n,745,755)`、`-fps_mode passthrough` 输出 11 张 bgr24 raw frames，共 2,162,688 bytes。输出 muxer 报告一次 `non monotonically increasing dts ... 750 >= 750`，exit code=0；核对实际字节长度为 11×256×256×3，未在输出中按 CFR 自动删/补帧。

| n | CLI pixel min/max | CLI 全部 BGR bytes 的 std | 上一报告 OpenCV std |
| ---: | --- | ---: | ---: |
| 748 | 74 / 78 | 0.43359341 | 0.43359341 |
| 749 | 74 / 76 | 0.29315098 | 0.293150985 |
| 750 | 0 / 167 | 10.89059187 | 10.890591870 |
| 751 | 10 / 170 | 10.74260587 | 10.742605870 |
| 752 | 7 / 197 | 11.04549998 | 11.04549998 |
| 753 | 7 / 199 | 11.09011298 | 11.09011298 |

这一独立 CLI 顺序解码复现了先前 OpenCV 4.11.0 的内容边界；**两者都使用 FFmpeg codec family，不称为独立 codec implementation，更不是对实验播放器的重放**。本轮未安装另一套 codec。内容边界仅为辅助证据，不能据 pixel threshold 定义 acquisition t=0。

## 6. 音频 pulse 的可观察时间，不等于 acquisition trigger 定义

顺序提取前 16.2 s mono PCM16：518,400 samples，peak absolute sample=32,768，非零 samples=4,178。前一个 decoded audio frame 的 PTS 是 `5613/90000 s`，每 audio frame 有 1152 samples。保存 [audio-frames-first16s.json](D:/PythonProject/retina_rf_SNN/.omo/evidence/schottdorf_lee_frame_zero/audio-frames-first16s.json) 与 [audio-first16p2s.s16le](D:/PythonProject/retina_rf_SNN/.omo/evidence/schottdorf_lee_frame_zero/audio-first16p2s.s16le)。

仅为可复算地描述解码 waveform，定义 pulse core 为 `abs(PCM16[s]) >= 16384` 的连续 samples。该阈值是半满量程的报告规则，**不是实验 trigger threshold**，没有用它判定 frame zero。MP2 解码中的微小非零尾部也不当作硬件边沿。

使用 decoded-audio sample clock：

```text
t_PCM(s) = s / 32000
t_container(s) = 5613 / 90000 + s / 32000
```

这只是给解码 samples 配上 container 时间标签；不减 codec/device latency、不乘实验播放比例，也不转换为 acquisition time。

| pulse core | 首/末 sample（零基、含末端） | PCM start s | container start s | container end-exclusive s |
| ---: | --- | ---: | ---: | ---: |
| 1 | 481439–481470 | 15.044968750 | 15.107335417 | 15.108335417 |
| 2 | 481631–481662 | 15.050968750 | 15.113335417 | 15.114335417 |
| 3 | 481823–481854 | 15.056968750 | 15.119335417 | 15.120335417 |
| 4 | 482015–482046 | 15.062968750 | 15.125335417 | 15.126335417 |
| 5 | 482207–482238 | 15.068968750 | 15.131335417 | 15.132335417 |
| 6 | 482399–482430 | 15.074968750 | 15.137335417 | 15.138335417 |
| 7 | 482591–482622 | 15.080968750 | 15.143335417 | 15.144335417 |
| 8 | 482783–482814 | 15.086968750 | 15.149335417 | 15.150335417 |
| 9 | 482975–483006 | 15.092968750 | 15.155335417 | 15.156335417 |
| 10 | 491039–491070 | 15.344968750 | 15.407335417 | 15.408335417 |
| 11 | 500639–500670 | 15.644968750 | 15.707335417 | 15.708335417 |
| 12 | 510239–510270 | 15.944968750 | 16.007335417 | 16.008335417 |

每个 core 为 32 samples=1 ms decoded-audio time。最初一组 9 个 cores 间距 6 ms。第 6 个 core start 与 n=749/750 的共同 PTS 相差约 31.25 µs（1 audio sample），第 9 个也靠近 n=751；**这些时间邻近关系没有官方 pulse-code/edge 定义，不能任选其中一个作 t=0**。没有取得独立 acquisition pulse trace 与导出 spike correction 的逐事件对应。

## 7. 四种对象必须分别判定

| 对象 | 本轮事实 | 是否已绑定到 corrected spike t=0 |
| --- | --- | --- |
| Decoded image-content onset | 顺序 CLI/OpenCV 输出在 750 处出现明显图像内容；749 近常数 | 否，像素不是 trigger |
| README frame numbering | 751 blank；live starting frame 752；未给具体 decoder ordinal/异常 PTS 处理规则 | 否，仅字面支持当前 751 映射 |
| Experimental displayed frame | 未取得原始 player 的 frame counter、实际显示日志或 pulse/display 同步程序 | **UNVERIFIED** |
| Acquisition-corrected spike t=0 | 这一 t=0 到 decoded 750/751 的具体边沿/帧对应没有直接证据 | **UNVERIFIED** |

判定所依赖的最直接证据不是“没有音频”，而是：**真实音轨存在多脉冲边界模式，视频边界存在重复 PTS，而官方可读说明和可用源码未定义哪一个已记录 pulse/display event 对应导出 t=0。** 目前无法完成最后一条映射。

## 8. 现有代码与 22-cell artifacts 的条件性影响

**没有确认一个应立即修复的 parser mismatch；也没有确认现行 751 映射正确。** 已复现的是 decoded content onset 与当前 loader 起点不同；未关闭的是 acquisition alignment。

- 现行 [常量:14](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py:14) 为 751；[178–192 的实现](D:/PythonProject/retina_rf_SNN/data/schottdorf_lee_2021.py:178) 把前 751 张输出用于 background，并从 index 751 填充 `live[0]`。本轮没有修改。
- **若外部证据最终确认 751**：现有 loader 不需因该 offset 修改；可以关闭这一 timing blocker，不需要因该 timing contract 重训。
- **若外部证据最终确认 750**：最小代码位置是上述常量及对应背景/live 分界合同；需在另一次获授权修改中用固定原始 movie、官方 t=0 映射做 regression test，核对 `live[0]`、背景只取正确 blank frames，以及已有 segment 边界抽样。这里仅列条件，不执行。
- 在后一条件下，同 loader 产生的 22-cell stimulus caches、fitting/checkpoints、prediction/RF/circuit/application artifacts 均属于潜在受影响 provenance 范围，包含 shared-BC 和此前采用相同 loader 的 22-cell 结果；不是说本轮已经量化或证明其数值无效。可能影响一张图像的配对位置以及 background normalization；若只是当前均匀模型轴的 ordinal 移动，差值为 1 bin=6.6666667 ms。**这不是已经测得的实际 CRT/acquisition offset**。
- 本轮未打开 checkpoint、未枚举重算各 artifact、未估计 NLL/RF/机制影响，不更改其通过/失败状态。当前不能出具“不需要 timing 重训”的无条件保证，也不执行重训。

## 9. 唯一外部 blocker 与作者询问信

需要一个可追溯的 **corrected t=0 → pulse edge → displayed frame → sequential decoded index** 对应证据。可由以下任一充分材料关闭，不要求全部同时提供：

1. 作者针对该 MD5 movie 明确确认：不 seek、不按 CFR drop/duplicate 的 sequential decode 中，第 750 或 751 个 zero-based image 对应 corrected t=0；同时说明 README frame counting 和 pulse edge 约定。
2. 原始 movie-generation / playback / spike-export correction 代码，能追踪实际 blank count、显示前后 trigger、音频 edge 和 t=0 的定义；包括边界重复 PTS 的播放策略或显示帧日志。
3. 同时记录的 acquisition trigger trace 与 display/frame marker（及其导出公式），可以直接锁定边界对应而无需依据图像内容或模型响应猜测。

未发现这些材料不等于证明它们不存在；本地历史是 shallow，远端完整 archive/history 未成功读取。**停止继续本地 audit，等待作者或原始同步材料。** 以下询问信仅供发送，未实际发送：

Subject: Frame-zero convention for the Schottdorf–Lee macaque dataset

Dear Dr. Schottdorf and Dr. Lee,

For `1x10_256.mpg` (MD5 `d64bdae05eb07895a8f30cda287c5a74`), does acquisition-corrected spike time zero correspond to zero-based frame 750 or 751 when decoded sequentially without frame dropping or duplication? The README specifies 751 blank frames and live frame 752. FFmpeg returns distinct frames 749/750 with the same PTS, and the audio contains a multi-pulse onset pattern. Could you identify which pulse edge and displayed frame define corrected t=0, including any playback A/V offset or duplicate-PTS handling? Original generation/playback or timestamp-correction code, or a synchronized trigger/frame trace, would resolve this. We have not changed the loader based on image content.

Thank you.

## 10. 命令与保存产物

以下列出实际使用工具/关键参数及输出位置。`MOVIE` 指第 2 节实际 payload，`OUT` 指 `D:/PythonProject/retina_rf_SNN/.omo/evidence/schottdorf_lee_frame_zero`，不是 training output。

```text
git -C OFFICIAL_REPO status --short
git -C OFFICIAL_REPO rev-parse --is-shallow-repository
git -C OFFICIAL_REPO log --all -20
git -C OFFICIAL_REPO log --all -G '751|752|timing|synchron|trigger|blank|frame' --format=fuller -- README.md
git -C OFFICIAL_REPO ls-tree -r --name-only HEAD
git -C OFFICIAL_REPO show HEAD:README.md
git -C OFFICIAL_REPO show HEAD:stimuli/1x10_256.mpg
git -C OFFICIAL_REPO grep -n -i -E 'trigger|synchron|audio|pulse|blank|frame.?number|751|752|VideoWriter' -- '*.py' '*.md' '*.sh'

ffprobe -v error -show_streams -show_format -of json -o OUT/streams.json MOVIE

ffprobe -v error -read_intervals "%+16" -select_streams v:0 -show_frames
  -show_entries frame=pts,pts_time,pkt_dts,pkt_dts_time,best_effort_timestamp,best_effort_timestamp_time,duration,duration_time,pkt_pos,pict_type,key_frame,coded_picture_number,display_picture_number,repeat_pict,interlaced_frame,top_field_first
  -of json -o OUT/first16s-frames.json MOVIE

ffprobe -v error -read_intervals "%+16" -select_streams v:0 -show_packets
  -show_entries packet=pts,pts_time,dts,dts_time,pos,flags,duration,duration_time
  -of json -o OUT/video-packets-first16s.json MOVIE

ffprobe -v error -read_intervals "%+16" -select_streams a:0 -show_frames
  -show_entries frame=pts,pts_time,pkt_dts,pkt_dts_time,best_effort_timestamp_time,pkt_pos,nb_samples
  -of json -o OUT/audio-frames-first16s.json MOVIE

ffmpeg -hide_banner -loglevel error -i MOVIE -map 0:a:0 -t 16.2
  -c:a pcm_s16le -f s16le OUT/audio-first16p2s.s16le

ffmpeg -hide_banner -loglevel error -i MOVIE -vf 'select=between(n\,745\,755)'
  -frames:v 11 -an -fps_mode passthrough -pix_fmt bgr24 -f rawvideo OUT/frames745-755-ffmpeg.bgr
```

还使用 PowerShell `ConvertFrom-Json` 读取 frame/packet 输出及 notebook code cells；`ReadAllBytes` + `Buffer.BlockCopy` 将 s16le 解码为 Int16，按第 6 节阈值逐 sample 列出连续区间；按每张 256×256×3 bytes 计算 rawvideo 的 min/max/mean/std。未写入或执行新的模型/训练脚本。

| 新 evidence 文件 | bytes | SHA256 |
| --- | ---: | --- |
| `streams.json` | 3921 | `017e986ddd3ce9a8b3ce8a9501415730f04efe79c40907d47f935549073dcb4e` |
| `first16s-frames.json` | 531155 | `2d4c627a224a6e9cc2ee6af0aff8ec2b6ec2f40409307dbcb7a358b7f982a0ff` |
| `video-packets-first16s.json` | 218613 | `3f25391b80cba1d6b335140b7663fff283a322a377f91b16f8e55c988bb27b59` |
| `audio-frames-first16s.json` | 111864 | `dc1067fa24d7a3be900186514945e47265d74b54e49ca5c7cf55a4bad8459d67` |
| `audio-first16p2s.s16le` | 1036800 | `cde0ad4fb8ba28347d1e435981586a9656930bc4fb018d7d057a1af1e81a95a4` |
| `frames745-755-ffmpeg.bgr` | 2162688 | `d9b815ddd46296931fefb7b2a652c2a78241883f90f224894e60e417bb83e38d` |

核对时官方 README SHA256=`aec20c21a4b00d58afaa4144f44c77b8b850f555871e20dc57f1562b4a831ef2`；当前 loader SHA256=`ae07cb57443ce95fcef2208060638c1cc12168ec9e1a1f3f9e58c82e24ddf764`。

**最终：UNRESOLVED_EXTERNAL_EVIDENCE_REQUIRED。Overall timing contract 保持 UNVERIFIED，仅这一 frame-zero 外部同步映射未关闭。**
