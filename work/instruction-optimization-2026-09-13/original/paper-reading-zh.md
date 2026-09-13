---
name: paper-reading-zh
description: 用于用户明确要求中文深度阅读、分析、批判或延展研究论文。仅出现论文链接、标题或学科关键词不触发；简单事实查询不触发。
---

# 论文深度阅读

## 准备工作

先获取用户提供的论文全文，再开始总结。

用户可能提供：

- URL，如arXiv、ACLAnthology、OpenReview、PapersWithCode、出版社页面或实验室主页
- PDF或文件附件
- 论文标题
- 论文标题加abstract

如果用户提供URL，先fetch全文。如果用户只提供标题，先web search找到论文的canonical版本，优先使用arXiv或官方会议版本。如果只能找到abstract，必须说明当前分析是临时的，并指出哪些部分无法验证。

不要只凭abstract作答。方法、实验细节、图表、limitations和appendix经常包含真正的贡献和真正的问题。

拿到全文后，先完整读一遍abstract、introduction、related work、method、experiments、conclusion，必要时阅读appendix。开始写之前，先确定三件事：

1. 论文的核心技术claim是什么。
2. 支撑这个claim的关键实验或关键论证是什么。
3. 最重要的baseline或prior work对比是什么。

如果novelty不清楚，先搜索2到3篇紧密相关的论文，用来校准这篇论文真正新在哪里。不要凭感觉宣布它novel，也不要为了生成follow-upidea而编造相关工作。

---

## 交付入口

开始撰写深度分析前，必须读取 [完整输出结构与风格](references/full-review.md)。沿用其完整 12 节顺序，只有第 6 节在没有形式化数学时可以跳过；不得因入口精简而减少交付内容。具体子问题按用户指定范围回答。

## 信息来源纪律

整个分析过程中，严格区分四类信息：

- **论文原文明确声称**：作者显式 claim 或展示了这一点
- **相关文献中的已有结论**：这是现有文献中的记录结果（尽可能引用）
- **基于证据的合理推断**：从论文展示的内容逻辑推出，但未被显式声明
- **仍然不确定的猜测**：你的假设，没有直接证据支撑

不要把推断写成事实，不要把猜测写成推断。如果不确定某个 claim 属于哪一类，直接说出来。

---

