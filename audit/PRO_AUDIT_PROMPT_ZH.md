# ChatGPT Pro 独立审计提示词

请独立审计我的 Retina 研究项目： https://github.com/ResultYI/retina_rf_SNN 。请先确认实际读取的分支和 commit，阅读根目录 `AUDIT_INDEX.md`，再沿路径检查源码、每 cell 数值、训练轨迹、checkpoint metadata、数据合同与相关 artifacts。README、CURRENT_STATE 和旧审计报告只是导航，不是已验证结论。若你无法访问某文件、运行代码或读取二进制 checkpoint，请列出缺失证据，不要假称已经复算。

最高研究目标：从真实 visual stimulus 经与 species/experiment 一致的 cone front-end，建立有生理约束的 H1→BC→AC→RGC 动态模型，预测真实 RGC spikes，并判断能从 RGC response 中可靠恢复多少 RF、pathway contribution 与 temporal dynamics。Macaque 是 primary biological target。真实数据不必直接记录 cone response；可重建 stimulus 加真实 spike timing 可以经正确 front-end 适配。请区分 prediction、model-internal intervention、parameter/circuit identifiability 与 biological validity。

本轮只审计并给建议，不训练、不直接修改模型或实验合同。你可以质疑当前架构及研究论证；“冻结”表示当前审计对象固定，不表示未来不得修改架构。负结果正常接受，不要为了保住现有结果而降低标准，也不要为了提高某个 NLL、RF cosine 或 illusion signature 而异化研究目标。

审计对象是公开名称 Canonical V1、内部 causal identity `h1-shared-bc-direct-broad-ac`、spatial identity `bc-central-disk_ac-overlapping-full-disk` 的最终 22-cell macaque lineage。实际数据流为 Cone→H1→shared BC encoder，分为 narrow direct BC→RGC 和 broader BC→downstream AC→RGC。不能混入旧 independent-AC encoder 的结果。当前采用 native 150 Hz、17×17 L+M Weber drive、Bernoulli target 与 strictly-past observed spike history。单凭“revision 4”不能判断 artifact 是否属于当前合同。

请逐项给出有文件、函数、数值支持的独立判断：

1. **结果可信度与充分性。** 核验 final prediction package 中 Constant、center-surround LN、CNN、Canonical V1、SC-adapted 的 per-cell NLL、population 加权、paired bootstrap 和参数计数。分别判断是否足以证明真实数据可拟合、预测价值、circuit 解释力、论文主张。不能把跑通当成生物机制验证，也不能仅凭预测落后就否定生理模型价值。
2. **数据与评价公平性。** 检查 source/split/loss mask、inner-dev selection、fresh full-train refit、validation 信息边界、causal history 和 reset。比较优化预算与模型选择机会；长期反复查看同一 validation 并修改架构是否使它只能作为 development evidence？区分已证实 leakage 与适应性选择偏差风险。评估 cell bootstrap、共同动物/recording 相关性及多个 probe/参数点的统计边界。
3. **不可忽略的限制。** 阅读 frame 750/751 未解决报告、150 Hz counts→Bernoulli 损失、reset/pre-roll sensitivity、四代表 cell 的 independent-seed 结果。不要把4 cells×3 fits或固定18 fits的81组合当成完整22-cell多seed验证。原始同步证据不足时保持 UNKNOWN，不自行补假设。
4. **RF 和机制识别。** 检查 RF 条件、真实 forward/autograd 定义、时间截断及参数冗余。区分预置 spatial geometry、signed graded AC current 和真正从 spike learning 恢复的性质。同 family noise-free synthetic recovery 能证明什么、不能证明什么？相似 prediction 是否仍可能对应相反或不同机制？不得跨因果合同挪用结果。
5. **Illusion 与聚合。** 必须同时检查 parametric benchmark 和 `aggregation_check/`。核验 normal/AC-off、width=0 step-edge control、control-subtracted interaction、四类曲线及 ON/OFF/四类等权。14/8 sign 分组与 polarity 完全对应意味着什么？哪些现象存在于单细胞，哪些只在 cohort mean 出现？区分模型人工刺激输出、LN/CNN 比较及真实 perceptual/biological 证据。保留 null results、输入范围和 pointwise CI 的限制。
6. **架构是否应优化。** 从实际 forward 判断瓶颈在 core、front-end/data、likelihood/history、optimization、evaluation 还是 scientific interpretation。不要预设加大网络、强制 slow-H1/rapid-AC 或制造漂亮 surround 是答案。若建议架构改动，说明具体代码限制、生理/计算文献依据、保留的生理约束、自由度变化、与更简单解释的区别、预期收益及 identifiability 风险。可以提出有证据的 topology/representation 改动，但不要把迎合单个指标当成研究目标。
7. **优先级。** 最多给5项高价值建议，每项说明解决哪项不确定性、最小可证伪验证、应固定的合同、成功/失败怎样改变结论和相对成本。区分 correctness fix、数据/前端、评价、训练充分性和架构假设。如果当前证据足够支持一个较窄主张，请明确哪些额外工作可以不做。

输出要求：

- 开头列出实际读取/未读取的关键证据、commit/lineage，以及完成了哪些独立计算。
- 用 VERIFIED / DOWNGRADE / INVALID / UNKNOWN 分类核心结果并附具体路径。真实数字但解释过强归 DOWNGRADE；缺 checkpoint/raw data 不能独立确认归 UNKNOWN。
- 分别评价 prediction、RF、circuit intervention、biological interpretation 的证据充分性。
- 给出可保留主文结论、应放 supplement 的内容、应撤回或改写的说法。
- 最后回答：当前最大瓶颈是什么；是否真的需要改 core；若需要，优先哪一个最小且服务于研究目标的改动；若不需要，最有价值的下一步证据是什么。
- 文献建议引用原论文或官方代码。把文献事实、仓库事实、你的推断和建议分开。不得编造文献、访问记录或复算。

请独立判断，不要迎合我希望听到“已足够好”或“必须换架构”的任何预期。
