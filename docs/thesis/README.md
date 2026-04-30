# 论文目录

> 完全继承胡修闻 2024 东南大学硕士论文骨架。每章一个目录，每节一个 Markdown 文件。

## 第一章 绪论 (`ch1_introduction/`)
- [1.1 研究背景和意义](ch1_introduction/1.1_research_background.md)
- [1.2 研究意义](ch1_introduction/1.2_significance.md)
- [1.3 研究现状](ch1_introduction/1.3_research_status.md)
- [1.4 研究内容](ch1_introduction/1.4_research_content.md)
- [1.5 本文组织架构](ch1_introduction/1.5_organization.md)

## 第二章 相关理论和关键技术概述 (`ch2_preliminaries/`)
- [2.1 SQL 注入攻击概述](ch2_preliminaries/2.1_sqli_overview.md)
- [2.2 代码表征概述](ch2_preliminaries/2.2_code_representation.md)
- [2.3 神经网络结构](ch2_preliminaries/2.3_neural_networks.md)
- [2.4 对抗机器学习](ch2_preliminaries/2.4_adversarial_ml.md)
- [2.5 本章小结](ch2_preliminaries/2.5_chapter_summary.md)

## 第三章 基于多视图融合的 SQL 注入攻击检测方法的实现 (`ch3_multi_view_detection/`)
- [3.1 问题分析](ch3_multi_view_detection/3.1_problem_analysis.md)
- [3.2 SQL 注入攻击检测框架](ch3_multi_view_detection/3.2_detection_framework.md)
- [3.3 多视图表征构建](ch3_multi_view_detection/3.3_multi_view_representation.md)
- [3.4 视图编码器构建](ch3_multi_view_detection/3.4_view_encoders.md)
- [3.5 分层注意力融合层构建](ch3_multi_view_detection/3.5_hierarchical_fusion.md)
- [3.6 算法具体流程](ch3_multi_view_detection/3.6_algorithm_flow.md)
- [3.7 实验结果与分析](ch3_multi_view_detection/3.7_experiments.md)
  - [3.7.1 实验数据集](ch3_multi_view_detection/3.7.1_dataset.md)
  - [3.7.2 实验评价指标](ch3_multi_view_detection/3.7.2_metrics.md)
  - [3.7.3 实验结果](ch3_multi_view_detection/3.7.3_results.md)
- [3.8 本章小结](ch3_multi_view_detection/3.8_chapter_summary.md)

## 第四章 基于变异攻击与嵌入扰动的协同进化对抗训练 (`ch4_adversarial_training/`)
- [4.1 问题分析](ch4_adversarial_training/4.1_problem_analysis.md)
- [4.2 对抗训练整体架构](ch4_adversarial_training/4.2_overall_framework.md)
- [4.3 基于变异的对抗样本生成](ch4_adversarial_training/4.3_mutation_based_generation.md)
- [4.4 FreeLB 嵌入扰动训练](ch4_adversarial_training/4.4_freelb.md)
- [4.5 协同进化迭代训练](ch4_adversarial_training/4.5_co_evolution.md)
- [4.6 算法具体流程](ch4_adversarial_training/4.6_algorithm_flow.md)
- [4.7 实验结果与分析](ch4_adversarial_training/4.7_experiments.md)
  - [4.7.1 实验数据集](ch4_adversarial_training/4.7.1_dataset.md)
  - [4.7.2 实验评价指标](ch4_adversarial_training/4.7.2_metrics.md)
  - [4.7.3 实验结果](ch4_adversarial_training/4.7.3_results.md)
- [4.8 本章小结](ch4_adversarial_training/4.8_chapter_summary.md)

## 第五章 SQL 注入入侵检测系统实现 (`ch5_system/`)
- [5.1 需求分析](ch5_system/5.1_requirements.md)
- [5.2 系统设计](ch5_system/5.2_design.md)
  - [5.2.1 系统设计思想](ch5_system/5.2.1_design_principle.md)
  - [5.2.2 系统架构](ch5_system/5.2.2_architecture.md)
- [5.3 系统模块介绍](ch5_system/5.3_modules.md)
  - [5.3.1 流量采集与存储模块](ch5_system/5.3.1_traffic_capture.md)
  - [5.3.2 检测模块](ch5_system/5.3.2_detection.md)
  - [5.3.3 结果可视化模块](ch5_system/5.3.3_visualization.md)
- [5.4 本章小结](ch5_system/5.4_chapter_summary.md)

## 第六章 总结与展望 (`ch6_conclusion/`)
- [6.1 总结](ch6_conclusion/6.1_summary.md)
- [6.2 展望](ch6_conclusion/6.2_future_work.md)

---

## 写作约定

- 每个 `.md` 文件顶部有 `# 标题`，正文按需要分子小节（H2/H3）
- 表格、公式、引用按 LaTeX 兼容 Markdown 写法
- 数据/图表占位用 `<!-- TODO: 填入实验数据 -->` 注明
- 引用 Hu 修闻 2024 仅放在 §2 相关工作；其他章节不与之做数字对比

## 状态追踪

每节文件顶部用以下标记显示进度：
```
status: [ ] empty / [-] outline / [#] drafted / [+] reviewed / [✓] final
```
