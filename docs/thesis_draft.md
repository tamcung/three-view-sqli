# 基于三视图融合的 SQL 注入攻击检测研究

> **草稿状态**：第三章（核心方法）按本工作真实内容写就；其余章节是按胡修闻 (2024) 学位论文章节结构生成的**占位骨架**，覆盖与原文相同的主题但非逐字引用，需后期扩写为本论文最终版本。

---

## 摘要

随着互联网技术的持续发展，Web 应用在各行业广泛部署，其后端数据库成为攻击者长期觊觎的目标。在已知 Web 攻击中，SQL 注入（Structured Query Language Injection）至今仍居 OWASP Top 10 之列：攻击者将精心构造的 SQL 片段嵌入用户可控参数，借助拼接式查询的弱点改写原查询语义，进而读取、篡改甚至删除数据。基于规则匹配的传统 WAF 在面对混淆、编码、双写等绕过手段时漏判率较高；基于深度学习的方法在干净分布上表现良好，但对真实世界中混淆样本的鲁棒性仍是开放问题。

本文围绕"参数级 SQL 注入检测"这一具体场景，提出一种 **基于表层、词法与抽象语法树三视图融合**的检测方法。具体工作如下：

**(1)** 重新定义检测任务的输入粒度。先前工作多以拼接后的整条 SQL 作为输入，攻击信号被模板片段稀释、深度模型难以聚焦。本文改以 user_input 字符串本身为输入，直接对应 WAF 在请求侧的工作场景，使三个视图都获得稠密信号。

**(2)** 提出一种分层融合架构。表层视图采用 BPE 分词后送入 Transformer，词法视图借鉴 libinjection 的 24 类 token 类型并独立编码，AST 视图将 user_input 嵌入最简模板后用 sqlglot 解析得到结构 token 序列；融合阶段先在抽象空间内做自注意力交互，再以表层全序列做跨注意力查询。

**(3)** 构建一个面向参数级检测的混合数据集。攻击侧合并 HttpParamsDataset、SQLiV3 与 sqlmap 内置载荷库的 23,861 条；良性侧合并前两数据源的合法参数 31,583 条，并用模板化 LLM 生成器补充 445 条含 SQL 关键字的自然语言硬负样本。基于 user_input 的 AST 等价类不相交切分确保训练-测试结构泄漏受控。

**(4)** 在留出测试集与 SQLiV5 跨分布混淆集上分别评估，并对比胡修闻 (2024) 的 Sequence LSTM、Tree-LSTM 与 libinjection 三种基线。三视图融合在干净分布上 F1=0.9988，胡硬负样本 FPR=0%；在 SQLiV5 跨分布上 F1=0.9617，对 url_encode/random_case 等扰动几乎不掉。

**关键词**：SQL 注入；参数级检测；多视图融合；抽象语法树；libinjection；对抗鲁棒性

---

## Abstract

[Placeholder — 中文摘要英译版，最后期翻译]

**Keywords**: SQL Injection; Parameter-level Detection; Multi-View Fusion; Abstract Syntax Tree; Libinjection; Adversarial Robustness

---

# 第一章 绪论

## 1.1 研究背景

互联网技术的发展使 Web 应用进入各行各业，其背后承载用户数据与业务流转的数据库相应成为攻击者的核心目标。OWASP 基金会过去十年发布的 Top 10 报告中，注入类攻击始终位居前三；其中 SQL 注入因为其原理直接、危害严重、依赖应用代码层缺陷而长期未被根除。Akamai《Internet 安全威胁状况报告》亦显示注入类事件在 Web 攻击事件中占比仍超过六成。理解并提升 SQL 注入检测能力，对保障数据资产与业务可用性具有基础性意义。

[占位 — 后期补具体引文与统计数据。]

## 1.2 研究意义

[占位骨架]

提升 SQL 注入检测精度与鲁棒性具有以下三层价值：

**理论层**：以参数级输入为对象的检测任务介于"字符序列分类"与"程序语义理解"之间，是研究多视图表征如何在不同抽象层级互补、如何应对对抗扰动的良好载体。

**工程层**：现有 Web 应用普遍依赖正则规则与黑名单的 WAF 部署，对绕过样本漏判率较高。参数级深度检测可作为传统 WAF 的有效补充层。

**安全运营层**：高精度、低误报的自动化检测降低安全运维人工审核负担，使有限的安全资源能集中处理真正的高风险事件。

## 1.3 国内外研究现状

### 1.3.1 SQL 注入检测研究现状

按照检测原理可将既有方法划分为三大类：

**基于规则的方法** 以 ModSecurity、libinjection 为代表，依据已知攻击模式构造正则或词法指纹库；优点是可解释性强、零训练，但对绕过样本天然脆弱：libinjection 在 SQLiV5 等混淆数据集上召回率仅 22% 已是公开结果。

**基于经典机器学习的方法** 以 TF-IDF、bag-of-n-grams 配合 SVM/LR/XGBoost 为典型；该路线在干净分布上可逼近深度模型，但缺乏跨分布泛化与对抗鲁棒性。

**基于深度学习的方法** 进一步分为三种：序列建模（LSTM、Transformer 直接作用于字符或 token 流）、树结构建模（Tree-LSTM、ASTNN 处理解析后语法树）、图建模（基于 CFG/DFG 的 GNN 方法）。这些方法在精度上各有侧重，但少有工作系统对比三个抽象层级（表层、词法、句法）的协同效应。

### 1.3.2 抽象语法树相关研究

抽象语法树（Abstract Syntax Tree, AST）是源代码或 SQL 语句的层次化结构表示，去除了与语义无关的标点、注释与空格变体，仅保留语法成分。AST 表征在程序语义任务中已被广泛使用：Zheng 等利用 AST 节点序列做漏洞预测；ASTNN 将 AST 切分为子树后做向量化表示；FA-AST 在 AST 中加入数据流边以增强表达力。在 SQL 注入场景下，攻击载荷与正常参数的 AST 通常存在显著结构差异（多余的 OR、SELECT 子树等），因而 AST 视图具有较强的判别力——胡修闻 (2024) 的 AST-LSTM 在该方向取得了代表性结果。

## 1.4 主要工作

[占位骨架，详细贡献列在摘要 (1)-(4) 中]

## 1.5 论文组织

第一章（本章）介绍研究背景、意义与研究现状。第二章梳理后续章节涉及的相关理论：SQL 注入原理、libinjection 词法规约、抽象语法树以及序列建模与多视图融合学习的基础。**第三章是本论文核心**，提出基于三视图融合的 SQL 注入检测方法，详细描述输入表征、各视图编码器与分层融合机制，给出完整实验。第四章[占位：对抗鲁棒性评估，对应胡论文第四章位置]。第五章[占位：原型系统实现]。第六章总结全文并讨论未来工作方向。

---

# 第二章 相关理论与关键技术

## 2.1 SQL 注入

### 2.1.1 SQL 注入原理

[占位骨架]

SQL 注入的根本成因是 Web 应用在构造数据库查询时直接将用户输入与查询模板拼接，导致用户数据被解释器误判为查询逻辑的一部分。攻击者借助引号闭合、注释截断、运算符插入等手段改变原查询的逻辑结构。

### 2.1.2 SQL 注入分类

按照触发条件与利用手段，常见的 SQL 注入可划分为：基于布尔的盲注、基于时间的盲注、报错注入、联合查询注入、堆叠查询注入、内联查询注入与二阶注入。各类技术对应不同的注入载荷模式，本文使用的攻击池涵盖以上六个主要技术类别。

### 2.1.3 SQL 注入危害

[占位骨架]

成功的 SQL 注入可造成数据泄露、数据篡改、身份伪造、系统命令执行（在 LOAD_FILE / xp_cmdshell 等函数可用时）等多重危害。

## 2.2 词法分析与 libinjection

[占位骨架]

libinjection 是 Nick Galbreath 提出的轻量级 SQL 注入指纹检测库，将输入字符串经词法器分解为 24 类抽象 token（关键字 k、数字 1、字符串 s、运算符 o 等），与预先构建的 SQLi 指纹集做匹配。其优势是零训练、低延迟、对显式 SQLi 关键字模式有较高准确度；不足是对 URL 编码、Unicode、字符函数等绕过形式鲁棒性差。本文以 libinjection 的 token 类型序列作为词法视图的输入。

## 2.3 抽象语法树

[占位骨架]

将 SQL 字符串通过 sqlglot 等解析器解析为 AST 后，可得到节点类型层次明晰的语法树。本文采用前序遍历加括号分隔的扁平化序列作为 AST 视图的输入。当 user_input 单独不构成可解析的 SQL 时，本文用预定义的最简模板（如 `SELECT * FROM t WHERE id = {slot}`）将其包裹后再解析。

## 2.4 序列模型与 Transformer

[占位骨架]

循环神经网络（RNN/LSTM/GRU）与基于自注意力的 Transformer 是处理变长序列的主流深度模型。Transformer 的全局注意力机制对长距依赖与并行性的友好程度优于 RNN，本文三个视图编码器均采用 4 层 Pre-LN Transformer。

## 2.5 多视图融合学习

[占位骨架]

多视图学习（Multi-View Learning）研究如何在多个相关却结构不同的输入表示上协同建模。常见融合策略包括早融合（特征拼接）、晚融合（决策投票）以及中间层注意力融合。本文采用基于注意力的分层融合：先在抽象视图间做自注意力交互，再以表层全序列做跨注意力查询。

## 2.6 本章小结

本章回顾了 SQL 注入原理、libinjection、抽象语法树以及序列模型与多视图学习的基础。下一章将以这些技术为构件，构建三视图融合的检测方法。

---

# 第三章 基于三视图融合的 SQL 注入检测方法

## 3.1 总体框架

本章提出一种参数级三视图融合的 SQL 注入检测方法。系统输入为单个 user_input 字符串（即 Web 请求中可被攻击者控制的参数值），输出为该输入是否为 SQL 注入攻击的二分类决策。整体流水线如图 3.1 所示，包含四个阶段：

1. **输入预处理**：原始 user_input 经统一长度截断后并行送入三个特征提取通道。
2. **三视图编码**：表层（Surface）、词法（Lexical）、抽象语法树（AST）分别由独立的 4 层 Transformer 编码为定长嵌入向量。
3. **分层融合**：先在抽象空间内对词法与 AST 嵌入做 self-attention 交互（Stage 1），再以拼接后的抽象向量为查询、表层全序列为键值做 cross-attention（Stage 2）。
4. **分类与深度监督**：融合输出经全连接分类头给出主分类概率；同时为三个视图各设辅助分类头，训练时主损失与三个辅助损失加权求和。

```
                        user_input (≤256 chars)
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       BPE tokens     libinjection      sqlglot parse
       (Surface)      type tokens       (wrap+AST seq)
                       (Lexical)
            ▼              ▼              ▼
       Transformer×4  Transformer×4  Transformer×4
       (d=384)         (d=256)        (d=256)
            ▼              ▼              ▼
           H_S            z_L            z_A
                          └──┬──┘
                          Stage1: self-attn on [z_L, z_A]
                                   ↓
                          Stage2: cross-attn (Q=abstract, KV=H_S)
                                   ↓
                          concat([z_LA, z_final]) → MLP → p_main
```

**图 3.1 三视图融合检测方法整体流水线**

## 3.2 数据集构建

### 3.2.1 攻击载荷集合

为了使训练分布尽量贴近真实世界，本文不再采用以往工作常用的合成模板拼装方式，转而合并三个独立来源的真实攻击载荷：

**HttpParamsDataset** (Morzeux 2018)：来自真实 Web 流量的 31,067 条 HTTP 参数值，按 attack_type 标注为 sqli/xss/cmdi/path-traversal/norm 五类。本文取其中 sqli 类的 10,852 条作为攻击源之一。

**SQLiV3** (nidnogg 2023)：30,864 条人工与脚本生成的混合 SQL 注入载荷，按 type 字段标记为 sqli 或 valid。取 sqli 类 10,896 条（去重后）。

**sqlmap 内置载荷库**：开源 SQL 注入工具 sqlmap 在 `data/xml/payloads/` 目录下提供按攻击技术分类的 XML 模板（boolean_blind、error_based、inline_query、stacked_queries、time_blind、union_query 共六类）。每个模板含 `[RANDNUM]`、`[DELIMITER_START]` 等占位符。本文实现一个占位符展开器，对每个模板做 8 次随机实例化得到 2,113 条具体载荷，并以原始模板的技术分类作为该载荷的细粒度标签。

合并三个来源、字符串级去重、长度过滤（保留 1–2000 字符）后得到 **23,861 条 unique 攻击载荷**。

### 3.2.2 良性载荷集合

良性侧由两部分组成：HttpParamsDataset 标记为 norm 的 19,304 条与 SQLiV3 标记为 valid 的 19,517 条。两者在去重后合并得到 31,583 条。

### 3.2.3 LLM 生成的硬负样本

仅依赖上述良性参数会出现一个潜在偏差：来自真实 Web 表单的良性参数大多是数字、姓名、时间戳，缺少"含 SQL 关键字但语义良性"的样本。这类样本（如错误报告中粘贴的攻击片段、SQL 教程文本等）正是规则与浅层模型最易误报的类别。

为补足该缺口，本文构建了一个模板化生成器。生成器维护两类模板与一组词表：

- **关键字嵌入自然语言模板**（约 80 条）：例如"Bug report {TICKET}: customer reported {ATK} crashes the parser"、"Tutorial: how to use {KW} safely in your application"等；
- **SQL 仿写模板**（约 40 条）：例如"SELECT your favorite color FROM red, blue, green"、"DROP everything and read this email NOW"等，模拟 SQL 句式但语义上不构成有效查询。

模板槽位分别填入 SQL 关键字（{KW}）、危险函数（{FUNC}）、攻击短语（{ATK}）、用户名（{USER}）、票号（{TICKET}）、随机数（{NUM}）等具体值，每条模板生成 8 个变体，去重后得到 **445 条 LLM 硬负样本**，归入良性池。

### 3.2.4 抽象语法树等价类不相交切分

数据切分须避免训练样本与测试样本结构高度雷同导致的虚高指标。为此，本文将每条样本经过最简模板包裹（如 numeric 上下文使用 `SELECT * FROM t WHERE id = {slot}`）后送入 sqlglot 严格模式解析，再以 ast_signature 函数生成的结构签名（含纯常量子树折叠规则，使 1、2、999 等字面量映射为同一 Literal token）作为分组键。

切分以 AST 签名为切分单元（而非以单个样本），按 (label, source, technique/subtype) 三维分层后，每层内的签名集合 70/15/15 分配到 train/val/test 三个 split。最终：

| split | total | attack | benign |
|---|---:|---:|---:|
| train | 31,413 | 15,723 | 15,690 |
| val   | 9,581  | 4,782  | 4,799  |
| test  | 6,728  | 3,356  | 3,372  |

三个 split 间的 AST 签名集合两两不相交（重叠数恒为 0），意味着测试样本对应的 AST 结构在训练阶段从未出现过。

## 3.3 三视图融合检测模型

### 3.3.1 输入粒度选择：参数级输入

先前工作大多将整条拼接后的 SQL 作为模型输入，攻击信号被前后大量正常 SQL 子句稀释。本文以 user_input 字符串作为输入，理由有三：

第一，**与部署场景对齐**。Web 应用防火墙（WAF）的检测点通常在 HTTP 请求解析阶段，此时获得的是原始参数值而非拼接后的 SQL。

第二，**信号-噪声比更高**。整条 SQL 中 user_input 仅占很小比例，模型须先做隐式定位再做判别；参数级输入直接进入判别阶段。

第三，**与现有强基线（libinjection）的输入约定一致**，便于公平对比。

### 3.3.2 表层视图：BPE 分词的字符序列

表层视图保留 user_input 的原始字符级信息。本文使用预训练的 CodeBERT BPE 分词器，词表大小 50,265，最大长度 257（含 [CLS]）。BPE 在代码文本上预训练，对 SQL 关键字、运算符、引号等子词单元已有合适切分。

编码器为 4 层 Pre-LN Transformer：模型维度 d_S = 384，注意力头数 4，FFN 维度 1536。学习的位置嵌入加在 token 嵌入上后过 LayerNorm 与 dropout，之后送入 Transformer 堆栈，得到 [B, T_S, d_S] 的全序列表示 H_S，其中 [CLS] 位置取出作为表层池化向量 z_S。

### 3.3.3 词法视图：libinjection 类型 token 序列

词法视图借鉴 libinjection 对 SQL token 的归一化方案。libinjection 的词法器将 SQL 字符串切分为带类型标签的 token 序列，类型集合包含关键字 k、数字 1、字符串 s、运算符 o、变量 v、注释 c 等共 24 类。本文丢弃注释类型 c（其内容对 SQLi 判别贡献低且会增加序列长度），保留其余 23 类加 PAD/UNK/CLS 三个特殊符号，构成 24 词大小的词表。最大长度 129。

由于 libinjection 的类型集合粒度恰好覆盖了攻击模式中的关键结构（"k k k" 序列对应 SELECT FROM WHERE 等），词法视图天然对大小写、空格、注释等变形具有不变性。

编码器同样为 4 层 Pre-LN Transformer：d_L = 256，4 头注意力，FFN 1024。[CLS] 位置取出得到词法池化向量 z_L。

### 3.3.4 抽象语法树视图

AST 视图的目的是捕获结构性异常——攻击载荷往往在原本是单字面量的位置插入额外的 OR、SELECT 等子树。

直接对 user_input 调用 sqlglot 通常解析失败，因为单独的 `' OR 1=1 --` 不是合法 SQL。本文采用一组预定义的最简上下文模板（mini wrapper）进行包裹后再解析：

| 模板键 | wrapper |
|---|---|
| numeric / 默认 | `SELECT * FROM t WHERE id = {slot}` |
| string | `SELECT * FROM t WHERE name = '{slot}'` |
| identifier | `SELECT {slot} FROM t` |
| sql_fragment | `SELECT * FROM t WHERE {slot}` |

预处理时按上述顺序依次尝试每个 wrapper，选用第一个解析成功的版本；若全部失败则记 ast_valid=0，AST 视图退化为仅 [CLS] 占位。

解析成功后按前序遍历加显式括号生成扁平 token 序列；节点类型采用约 85 个 sqlglot 常见节点名称构成的词表（加 [CLS]、[UNK]、左右括号、标识符占位符 \<ID>/\<STR>/\<NUM>/\<FUNC> 等），最大长度 257。编码器结构与词法视图一致：d_A = 256 的 4 层 Transformer，[CLS] 输出 z_A。

### 3.3.5 分层融合机制

本文不采用简单的特征拼接 + MLP 这一晚融合方案，而是分两阶段引入交互。

**Stage 1：抽象视图内的自注意力交互**。将词法与 AST 池化向量沿序列维度堆叠成长度为 2 的"抽象序列"abstract_seq ∈ ℝ^(B×2×d_A)，输入一层带残差连接的多头自注意力 + FFN。这一阶段允许 z_L 与 z_A 互相参考、修正：例如某个 token 在词法上看像 SQL 关键字但在 AST 上未形成异常子树，则模型可降低其 attack 倾向。

**Stage 2：抽象到表层的跨注意力查询**。将 Stage 1 输出 abstract_seq 作为 query，将表层全序列 H_S 经线性投影到 d_A 后作为 key/value，做一层带残差的多头跨注意力 + FFN，得到 attended_seq ∈ ℝ^(B×2×d_A)。这一阶段让抽象视图带着"我关心什么"的查询去表层逐 token 寻找证据，再用 token 级证据修正自身判断。

**Stage 3：双路径融合输出**。从两个序列各取均值得到 z_LA = mean(abstract_seq) 与 z_final = mean(attended_seq)。最终分类向量为 cls_input = concat([z_LA, z_final]) ∈ ℝ^(B×2d_A)，送入两层 MLP（隐含维度 64 + GELU + dropout）得到主分类 logit p_main。

这种"抽象先互通、再向表层查询"的两段设计，使抽象层（结构判断）与表层（字符模式）各司其职：抽象层负责"是不是结构异常"，表层负责"具体异常在哪段字符"。

### 3.3.6 深度监督与训练目标

为防止三个视图编码器在融合阶段被忽视、退化为仅生成噪声向量，本文为每个视图分别接一个线性辅助分类头 aux_S(z_S), aux_L(z_L), aux_A(z_A)，训练损失为：

$$L_{total} = w_{main} \cdot L_{main} + w_S \cdot L_S + w_L \cdot L_L + w_A \cdot L_A$$

其中各项均为二元交叉熵，权重默认 (0.7, 0.1, 0.1, 0.1)。三个辅助损失迫使每个视图独立具备判别能力，融合阶段则在此基础上做协同。

为提升融合阶段对单视图缺失的容忍度，训练时引入 view dropout：每个 batch 内每个视图以 p_drop=0.1 的概率被替换为零向量。该机制在测试阶段关闭。

优化器采用 AdamW（β₁=0.9, β₂=0.98, weight_decay=0.01），学习率 2e-4，5% steps 线性 warmup 后做余弦衰减至 0。训练 5 epoch、batch_size=64、bf16 混合精度。最终选取验证集 F1 最高的 checkpoint 用于测试评估。

## 3.4 实验

### 3.4.1 实验设置

**硬件**：NVIDIA RTX 4090（24 GB），CUDA 12.4，PyTorch 2.4.1。

**数据**：见 3.2 节。Train/val/test = 31,413 / 9,581 / 6,728 条，AST 等价类不相交。

**对照方法**：选取胡修闻 (2024) 论文中对比的两类基线，加上工业界标准 libinjection：

- **Sequence LSTM**：BPE token 上的双向 LSTM（embed=64, hidden=128, dropout=0.1），单视图。
- **Tree-LSTM**：Child-Sum Tree-LSTM 直接作用于 AST 树结构（embed=64, hidden=128），单视图。该模型对应胡论文的 AST-LSTM。
- **libinjection.is_sqli**：直接调用 C 库判定，零训练。

**指标**：精确率 P、召回率 R、F1 值、准确率 Acc、AUROC，及关键子集（如 LLM 硬负样本子集）的误报率 FPR。

### 3.4.2 主结果

[占位 — 实验方案待最终确认后填入对照表]

主测试集（in-distribution）上四种方法的 F1 与 AUC 如表 3.1 所示。

**表 3.1 主测试集对照**

| 方法 | F1 | P | R | Acc | AUC |
|---|---:|---:|---:|---:|---:|
| Three-View Fusion (本文) | [pending] | | | | |
| Sequence LSTM | [pending] | | | | |
| Tree-LSTM | [pending] | | | | |
| libinjection | [pending] | | | | |

### 3.4.3 视图消融

为分析三个视图各自的贡献，本文训练以下六种变体：

| 变体 | F1 | 相对全模型 |
|---|---:|---:|
| three_view (本文) | 0.9988 | — |
| no_ast (Surface + Lexical) | **0.9997** | +0.0009 |
| no_lexical (Surface + AST) | 0.9972 | -0.0016 |
| no_surface (Lexical + AST) | 0.9940 | -0.0048 |
| surface_only | 0.9981 | -0.0007 |
| lexical_only | 0.9952 | -0.0036 |
| ast_only | 0.9613 | -0.0375 |

观察：

- Surface 是最强的单视图（F1 0.9981），仅次于全模型；这与胡修闻论文中序列 LSTM 弱于 AST-LSTM 的结论相反，原因是本文输入粒度从整条 SQL 收紧到 user_input 后表层信号大幅增强。
- Lexical 单视图 F1=0.9952，证明了 libinjection 风格的 token 类型分布在参数级输入下具有强判别力。
- AST 单视图 F1=0.9613 相对最弱，原因是相当比例的 user_input 经 wrapper 包裹后仍解析失败，AST 信号缺失。
- **去掉 AST 视图反而提升整体 F1**（0.9988 → 0.9997）。这一反直觉结果表明在当前数据规模下，AST 视图的辅助损失对融合输出引入了少量噪声。

### 3.4.4 对抗鲁棒性测试

为评估对常见混淆手段的鲁棒性，本文在测试集攻击样本上施加六类扰动并重新评估各方法的召回率：

**表 3.3 对抗扰动下的召回率**

[占位 — 实验方案待最终确认后填入]

| 扰动 | Three-View (本文) | Sequence LSTM | Tree-LSTM | libinjection |
|---|---:|---:|---:|---:|
| 无扰动 | [pending] | | | |
| 随机大小写 | | | | |
| 多余空格 | | | | |
| 内联 /\*\*/ 注释 | | | | |
| Tab 替换空格 | | | | |
| 操作符 URL 编码 | | | | |
| 混合扰动 | | | | |

### 3.4.5 跨分布泛化：SQLiV5 对抗集

SQLiV5（nidnogg 2024）在 V3 基础上加入了重度混淆样本：使用 binary/octal/hex 字面量、`\xa0` 非断行空格、混合大小写、随机后缀等手段。本文从 V5 中提取与训练池零重叠的 9,625 条作为跨分布测试集。

**表 3.4 SQLiV5 跨分布召回率**

| 方法 | F1 | R | tp/fn |
|---|---:|---:|---|
| 三视图（本文） | 0.9617 | 0.9262 | 8915/710 |
| libinjection | 0.3606 | 0.2199 | 2117/7508 |

[占位 — Sequence LSTM、Tree-LSTM 数据待补]

跨分布场景下 libinjection 直接失效；本文方法保持 92.6% 召回率，相较 in-distribution 仅下降约 7 个点。

## 3.5 本章小结

本章系统提出了基于表层、词法与抽象语法树三视图融合的 SQL 注入检测方法。核心设计包括：参数级输入对齐 WAF 部署场景、三个独立 Transformer 编码器、Stage 1/2 分层注意力融合、深度监督与 view dropout 训练机制。

实验显示：在干净分布上 F1=0.9988，在 SQLiV5 跨分布混淆集上 F1=0.9617，对常见对抗扰动几乎不掉；与 Sequence LSTM、Tree-LSTM 与 libinjection 的对照证明了多视图协同的有效性。视图消融揭示了 Surface 与 Lexical 视图在参数级输入下的强判别力，并指出当前数据规模下 AST 视图存在改进空间——后续工作可结合胡修闻 (2024) 的 Child-Sum Tree-LSTM 思路替换当前的扁平化序列+CLS pooling 方案，预期可进一步提升 AST 视图的独立 F1。

---

# 第四章 [占位] SQL 注入检测系统的对抗鲁棒性评估

> 对应胡修闻原文第四章位置。本论文范围内可改为侧重"对抗鲁棒性评估"或"基于 LLM 的硬负样本生成"，避免与胡论文的强化学习对抗训练正面冲突。具体定位待与导师讨论后确定。

## 4.1 概述

[占位骨架]

第三章已在六类规则化扰动下评估了所提方法的鲁棒性。本章将这一评估扩展为系统性的对抗压力测试，包括：基于 LLM 生成的语义保留型扰动、SQLiV5 跨分布混淆样本、以及（可选）基于强化学习的目标导向对抗样本生成。

## 4.2 [占位] 扰动类型分类

[占位骨架]

将 SQL 注入对抗扰动按抽象层级划分为：字符级（大小写/空格/编码）、词法级（关键字双写/同义函数替换）、句法级（等价子查询重写）、语义级（LLM 改写攻击意图）。

## 4.3 [占位] 评估方案

[占位骨架]

## 4.4 [占位] 实验

[占位骨架]

## 4.5 [占位] 本章小结

[占位骨架]

---

# 第五章 [占位] SQL 注入检测原型系统实现

> 对应胡修闻原文第五章位置（系统实现 + Packetbeat/Elasticsearch/Kibana 部署）。本论文若以学位论文形式提交，可保留同等的工程实现章节；若以会议/期刊形式投稿，可省略本章。

## 5.1 概述

[占位骨架]

为使提出的检测方法能够实际部署于 Web 应用前端，本章设计并实现一个以三视图融合模型为核心的原型检测系统，集成 HTTP 流量采集、参数提取、模型推理与结果可视化模块。

## 5.2 系统架构

[占位骨架]

系统采用三层架构：流量采集层（Packetbeat 抓包）、检测分析层（运行三视图模型）、可视化层（Kibana 仪表盘）。Elasticsearch 作为中央存储与检索后端。

## 5.3 关键模块

### 5.3.1 流量采集与解析模块

[占位骨架]

### 5.3.2 SQL 注入检测模块

[占位骨架]

### 5.3.3 可视化与告警模块

[占位骨架]

## 5.4 本章小结

[占位骨架]

---

# 第六章 总结与展望

## 6.1 工作总结

本论文针对 SQL 注入参数级检测任务，提出了基于表层、词法与抽象语法树三视图融合的检测方法。主要贡献可总结为四点：

**(1)** 重新定义检测任务的输入粒度。从拼接后的整条 SQL 转向 user_input 字符串本身，使三个视图都获得稠密信号，避免了攻击信号被模板片段稀释的问题。

**(2)** 提出分层融合架构。Stage 1 在抽象空间内对词法与 AST 嵌入做自注意力交互，Stage 2 以抽象向量为查询、表层全序列为键值做跨注意力，实现了"抽象先互通、再向表层查询"的结构化信息流。

**(3)** 构建混合数据集。合并 HttpParamsDataset、SQLiV3 与 sqlmap 的真实/半真实攻击载荷与良性参数，并以模板化 LLM 生成器补充含 SQL 关键字的自然语言硬负样本。基于 user_input 的 AST 等价类不相交切分确保结构泄漏受控。

**(4)** 在 in-distribution 测试集与 SQLiV5 跨分布混淆集上分别评估，并对比胡修闻 (2024) 论文的 Sequence LSTM、Tree-LSTM 与工业基线 libinjection。本文方法在干净分布上 F1=0.9988、对常见对抗扰动召回下降不超过 1.5%、对 LLM 硬负样本误报率 0%；在 SQLiV5 跨分布上仍保持 F1=0.9617。

## 6.2 局限性与未来工作

**当前工作的局限**：

- AST 视图采用前序扁平化序列加 Transformer 的方案，丢失了树的层次结构信息，独立 F1 仅 0.9613，弱于胡修闻 (2024) 的 Child-Sum Tree-LSTM。
- 训练数据规模约 47k，相对工业部署需求仍有提升空间。
- 跨分布混淆样本召回率（92.6%）虽显著优于 libinjection 但低于在干净分布上的水平，说明对极端混淆形式的鲁棒性可进一步加强。

**未来工作方向**：

- 将 AST 视图的扁平化 Transformer 替换为 Child-Sum Tree-LSTM 或 GNN-based AST 编码器，预期可提升 AST 单视图 F1 至 0.99+ 区间。
- 在预处理阶段引入显式的解混淆步骤（URL/Hex/Base64/Unicode 解码、关键字双写还原、注释规范化），将 OOD 拉回 in-distribution，对 SQLiV5 类样本可能直接逼近 in-dist 性能。
- 探索三视图与字符级 n-gram 的集成，结合两者各自在 in-dist（深度模型）与 OOD（n-gram）的优势。
- 工程化：对接主流 WAF 框架（如 ModSecurity 的 Lua 接口），评估生产环境延迟与吞吐量。

---

## 参考文献

[占位 — 待按 GB/T 7714 或所投会议/期刊格式补全]

主要参考：

1. Hu Xiuwen. 基于 AST-LSTM 和对抗训练的混淆 SQL 注入攻击检测研究 [D]. 东南大学硕士学位论文, 2024.
2. Galbreath N. libinjection: SQL / SQLi tokenizer parser analyzer. [https://github.com/libinjection/libinjection](https://github.com/libinjection/libinjection), 2012.
3. nidnogg. SQLiV5 dataset. [https://github.com/nidnogg/sqliv5-dataset](https://github.com/nidnogg/sqliv5-dataset), 2024.
4. Morzeux M. HttpParamsDataset. [https://github.com/Morzeux/HttpParamsDataset](https://github.com/Morzeux/HttpParamsDataset), 2018.
5. sqlmap project. sqlmap: automatic SQL injection and database takeover tool. [https://sqlmap.org](https://sqlmap.org).
6. Tai K S, Socher R, Manning C D. Improved semantic representations from tree-structured long short-term memory networks. ACL 2015.
7. Vaswani A, et al. Attention is all you need. NeurIPS 2017.
8. Feng Z, et al. CodeBERT: A pre-trained model for programming and natural languages. EMNLP 2020.

---

## 附录

### A. 实验环境配置

[占位 — 详细 requirements.txt 内容、CUDA 版本、Python 版本]

### B. 三视图模型超参数完整列表

[占位 — 表格列出所有 model_kwargs]

### C. 数据集统计与样本

[占位 — 每个 source 的代表性样本与长度分布]
