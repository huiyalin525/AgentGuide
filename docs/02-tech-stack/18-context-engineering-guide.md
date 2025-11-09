# 上下文工程完全指南:设计控制信息流向LLM的系统

> **核心理念**: 上下文工程是设计架构的学科,在正确的时间向LLM提供正确的信息。这不是改变模型本身,而是构建连接模型与外部世界的桥梁。

## 目录

- [什么是上下文工程](#什么是上下文工程)
- [核心组件](#核心组件)
  - [1. Agents - 决策大脑](#1-agents---决策大脑)
  - [2. Query Augmentation - 查询增强](#2-query-augmentation---查询增强)
  - [3. Retrieval - 检索系统](#3-retrieval---检索系统)
  - [4. Prompting Techniques - 提示技巧](#4-prompting-techniques---提示技巧)
  - [5. Memory - 记忆系统](#5-memory---记忆系统)
  - [6. Tools - 工具集成](#6-tools---工具集成)
- [总结](#总结)

---

## 什么是上下文工程?

每个使用大语言模型(LLM)构建应用的开发者都会遇到同样的瓶颈。你从一个强大的模型开始,它能够写作、总结、推理,表现出惊人的能力。但当你尝试将其应用到现实世界的问题时,裂缝就开始出现:

- 无法回答关于你私有文档的问题
- 不知道昨天发生的事件
- 当不知道答案时会自信地编造

**问题的本质不在于模型的智能,而在于它从根本上是断开连接的。**

这种隔离是其核心架构限制的直接结果:**上下文窗口**。上下文窗口是模型的活动工作内存——保存当前任务指令和信息的有限空间。每个字、数字、标点符号都会消耗这个窗口中的空间。就像白板一样,一旦满了,旧信息就会被擦除以为新指令腾出空间,重要细节可能会丢失。
![](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/df7e1dd2d6fb4d12557ee907c2eec8075b638d925435d224a38843c688ba396f.jpg)

你无法仅通过编写更好的提示来修复这个根本限制。**你必须围绕模型构建一个系统。这就是上下文工程。**

---

## 核心组件

上下文工程由6个核心组件组成,每个组件解决LLM应用中的特定挑战:

### 1. Agents - 决策大脑

**定义**: 编排如何以及何时使用信息的决策系统。


![](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/c852f67e804d920084f34926939263c396b4cfd3be28be895ae1fefc4de7775e.jpg)
#### 什么是Agent?

在大语言模型的上下文中,AI Agent是一个能够:

1. **动态决策信息流**: 基于学到的内容决定下一步做什么,而不是遵循预定路径
2. **跨多次交互维护状态**: 记住已完成的事情并使用历史信息指导未来决策
3. **自适应使用工具**: 从可用工具中选择并以未明确编程的方式组合它们
4. **基于结果修改方法**: 当一种策略不起作用时,可以尝试不同的方法

![](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/51fc62d5607c390ad49ad16c1bda6cb23827ca873f78e1e25becc92d79389d45.jpg)
#### Agent架构类型

**单Agent架构**:
- 尝试自己处理所有任务
- 适用于中等复杂度的工作流

![单Agent架构](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/e88a4ddf4cf3e2635b738e63b411d19c49427fc79023e04fd6aeff076a6bbef7.jpg)

**多Agent架构**:
- 在专门的Agent之间分配工作
- 允许复杂的工作流但引入协调挑战

![多Agent架构](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/e527950b67920c530e603c4aff520750ef8a50d1389e2d35639ed915906732d2.jpg)

#### 上下文窗口的挑战

LLM具有有限的信息容量,因为上下文窗口一次只能容纳这么多信息。每次Agent处理信息时,它都需要做出关于:

- 哪些信息应该保持在上下文窗口中活跃
- 哪些应该外部存储并在需要时检索
- 哪些可以总结或压缩以节省空间
- 为推理和规划保留多少空间
![](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/8ebbf01083ef17c574545dde8a3b84828909a736816c5f09543f263e42d2145f.jpg)
#### 常见的上下文错误类型

**上下文污染(Context Poisoning)**:
- 错误或幻觉信息进入上下文
- 因为Agent重用和构建该上下文,这些错误会持续并复合
![](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/433b21a62cea8ffe30c2e93024aabd73306a43f40051ab7f925030a2d6a81566.jpg)

**上下文干扰(Context Distraction)**:
- Agent被过多的过去信息(历史、工具输出、摘要)负担
- 过度依赖重复过去的行为而不是新鲜推理
- ![](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/9bca936d5709ad019bea07c02c4c273bdab476eb26584348df6bee708836f594.jpg)

**上下文混乱(Context Confusion)**:
- 不相关的工具或文档挤满上下文
- 分散模型注意力并导致使用错误的工具或指令
![](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/44e16e8cfd49e763b2f147f7715d91496bc6a167b317d4ddb96e590d817a6b61.jpg)

**上下文冲突(Context Clash)**:
- 上下文中的矛盾信息误导Agent
- 使其陷入冲突假设之间
![](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/44e16e8cfd49e763b2f147f7715d91496bc6a167b317d4ddb96e590d817a6b61.jpg)
#### Agent的核心策略和任务

Agent能够有效编排上下文系统,因为它们能够以动态方式进行推理和决策:

1. **上下文总结**: 定期将累积的历史压缩成摘要以减少负担同时保留关键知识
2. **质量验证**: 检查检索的信息是否一致和有用
3. **上下文修剪**: 主动删除不相关或过时的上下文
4. **自适应检索策略**: 当初始尝试失败时重新制定查询、切换知识库或改变分块策略
5. **上下文卸载**: 将细节存储在外部并仅在需要时检索
6. **动态工具选择**: 只过滤和加载与任务相关的工具
7. **多源综合**: 组合来自多个源的信息,解决冲突并产生连贯的答案

![不同类型的Agent在上下文工程系统中的功能](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/085e32cc2ebf373a77de31e2fef9c232f3f8516311c28a3c2f6ec1098ee791d3.jpg)

---

### 2. Query Augmentation - 查询增强

**定义**: 将混乱、模糊的用户请求转换为精确、机器可读意图的艺术。


上下文工程中最重要的步骤之一是如何准备和呈现用户的查询。有两个主要问题需要考虑:

1. **用户通常不以理想方式与聊天机器人交互**
   - 现实世界中的用户交互可能不清楚、混乱且不完整
   - 需要实现处理所有类型交互的解决方案

2. **管道的不同部分需要以不同方式处理查询**
   - LLM理解良好的问题可能不是搜索向量数据库的最佳格式
   - 需要一种适合不同工具和步骤的查询增强方法

#### 2.1 查询重写(Query Rewriting)

将原始用户查询转换为更有效的检索版本。

![查询重写流程](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/d5d8f53972aacf6e7573915509c85b6d58b982f7b5e9680baca60d20034e0c5a.jpg)

**工作原理**:
- **重构不清楚的问题**: 将模糊或形式不佳的用户输入转换为精确、信息密集的术语
- **上下文移除**: 消除可能混淆检索过程的无关信息
- **关键词增强**: 引入常见术语以增加匹配相关文档的可能性

#### 2.2 查询扩展(Query Expansion)

从单个用户输入生成多个相关查询来增强检索。

![查询扩展流程](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/e9613e9d19173cae56f6e710dfa0e951564c88817865685253b15afddaea806a.jpg)

**需要注意的挑战**:
- **查询漂移**: 扩展的查询可能偏离用户的原始意图
- **过度扩展**: 添加过多术语可能降低精度
- **计算开销**: 处理多个查询会增加系统延迟

#### 2.3 查询分解(Query Decomposition)

将复杂、多方面的问题分解为更简单、集中的子查询。

![查询分解流程](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/ea13dd601826b0bcaf2be617dfd4d1f9dde5657bdbdfaae05859a800604afa01.jpg)

**过程**包括两个主要阶段:
1. **分解阶段**: LLM分析原始复杂查询并将其分解为更小、集中的子查询
2. **处理阶段**: 每个子查询独立通过检索管道处理

#### 2.4 查询Agent(Query Agents)

查询Agent是查询增强的最高级形式,使用AI Agent智能处理整个查询处理管道。

![查询Agent架构](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/413441a4f206ef6da7f19f9da5984ec72b08875319ccba4b003dae0449b932a1.jpg)

---

### 3. Retrieval - 检索系统

**定义**: 连接LLM到你的特定文档和知识库的桥梁。

LLM的能力取决于它能访问的信息。虽然LLM在海量数据集上训练,但它们缺乏对你特定私有文档和训练完成后创建的任何信息的了解。

![RAG架构](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/83d98474a0cf62bfe3394ed4793af5a693b2c18653859e2b61ce0af005035d4d.jpg)

**挑战**: 原始文档数据集几乎总是太大而无法放入LLM有限的上下文窗口。我们必须找到完美的片段——包含用户查询答案的单个段落或部分。

为了使我们庞大的知识库可搜索,我们必须首先将文档分解为更小、可管理的部分。这个基础过程称为**分块(Chunking)**。

#### 分块技术指南

分块是你为检索系统性能做出的最重要决定。

![分块策略矩阵](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/d047c89c6a8d7768ba13a34e485680912da7bfcef9c9f1deaf742c15fa6674f2.jpg)

设计分块策略时,必须平衡两个竞争优先级:

- **检索精度**: 块需要小而专注于单个想法
- **上下文丰富性**: 块必须足够大和自包含以便被理解

目标是找到"分块最佳点"——创建足够小以实现精确检索但足够完整以给LLM所需完整上下文的块。

#### 简单分块技术

**固定大小分块(Fixed-Size Chunking)**:
![固定大小分块](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/c792ee589ab789f6f8bfdd40fab414f7701474288a48793a938bd954915ce992.jpg)

**递归分块(Recursive Chunking)**:
- 使用优先级分隔符列表分割文本
- 尊重文档的自然结构

**基于文档的分块(Document-Based Chunking)**:
- 使用文档的固有结构

#### 高级分块技术

**语义分块(Semantic Chunking)**:
- 基于含义而不是分隔符分割文本

**基于LLM的分块(LLM-Based Chunking)**:
![LLM分块](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/38e46b06e6be9d998b7d54fe519b22f79c022fc1d8a6db6f5d3b64dcc42e5eb3.jpg)

**Agentic分块**:
![Agentic分块](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/cb580a2dd585687041549517a36bff67e83164fbceb141b3480b934a95b3f990.jpg)

**层次分块(Hierarchical Chunking)**:
![层次分块](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/c417065f4f166f7f6541832880c70bd4d5f50380b650c8a3c239c2250488e29b.jpg)

**延迟分块(Late Chunking)**:
![延迟分块](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/f999e4b4af937efe351cb78efd736fd8a08c9f650131f99ea77f77bf40ee4294.jpg)

#### 预分块 vs 后分块

**预分块(Pre-Chunking)**:
![预分块流程](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/ca9e669e815e5549e8b4807f8f8fa433224643d58e64add0df89c833eaa5c786.jpg)

**后分块(Post-Chunking)**:
![后分块流程](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/0640b36b605261a566de2082693e561c59ab9f12f786a1740809f2c1a886ac22.jpg)

---

### 4. Prompting Techniques - 提示技巧

**定义**: 给出清晰、有效指令以引导模型推理的技能。

提示工程是设计、细化和优化给予大语言模型的输入(提示)以获得期望输出的实践。

#### 经典提示技术

**思维链(Chain of Thought, CoT)**:
![CoT示例](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/746d32ba43a68f992b2d3afb73fede6052d4c6c8c1c627684f4c19463743d0f4.jpg)

**少样本提示(Few-Shot Prompting)**:
![Few-shot示例](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/faadb6a458899736c4801d4f0097afcb9da7601cedc96795b3fbeb5cd1860a68.jpg)

**结合CoT和Few-shot**:

结合CoT和Few-shot示例是一种强大的方式,可以同时指导模型的推理过程和输出格式,以获得最佳效率。


**专业技巧 #1**:
使思维链中的模型推理非常具体到你的用例。例如,你可以要求模型:
- 评估环境
- 重复任何相关信息
- 解释这些信息对当前请求的重要性


**专业技巧 #2**:
最大化效率并减少token数量,要求模型以"草稿"形式推理,每句话不超过5个单词。

这确保了模型的思考过程是可见的,同时减少了输出token数量。
#### 高级提示策略

**思维树(Tree of Thoughts, ToT)**:
![ToT示例](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/7cf13b2a073a3aa7dcfc344eb3e380aa32eb68188ca7d1c44ca14f78ff601e58.jpg)

**ReAct提示**:
![ReAct框架](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/bbb3c83e784ba4459ffc1f8328efd9eb4e96b27e0c1e9eac35b08a87288cdb0b.jpg)

---

### 5. Memory - 记忆系统

**定义**: 给你的应用程序历史感和从交互中学习能力的系统。


![Karpathy的类比](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/6734cb97bad47256b858e88f4d1107560a563b2bbb3caad1f782be80a0879bba.jpg)

#### Agent记忆的架构

在构建强大的Agent时,我们需要分层思考记忆,通常混合不同类型的记忆以获得最佳效果。

**短期记忆**:

短期记忆是Agent的即时工作空间。这是"现在",被塞入上下文窗口以推动即时决策和推理。这通过**上下文学习**实现,将最近的对话、操作或数据直接打包到提示中。

示例对话:
- 用户: "天气怎么样?"
- AI: "晴天,24°C"
- 用户: "我需要带夹克吗?"
- AI: "不需要,很暖和!"

因为受到模型token限制的约束,主要挑战是效率。诀窍是保持这个精简,以减少成本和延迟,同时不遗漏任何对下一步处理可能重要的细节。

**长期记忆**:

长期记忆超越了即时上下文窗口,将信息外部存储以便在需要时快速检索。这使Agent能够随着时间推移建立对其世界和用户的持久理解。它通常由**检索增强生成(RAG)**驱动,Agent查询外部知识库(如向量数据库)来提取相关信息。

这种记忆可以存储不同类型的信息,例如:
- **情节记忆**: 存储特定事件或过去的交互
- **语义记忆**: 保存一般知识和事实(可以是公司文档、产品手册或精选的领域知识库的信息,使Agent能够准确回答问题)

**混合记忆设置**:

实际上,大多数现代系统使用混合方法,将短期记忆的速度与长期记忆的深度相结合。一些高级架构甚至引入了额外的层:

- **工作记忆**: 与特定多步骤任务相关信息的临时存储区。例如,如果Agent正在预订旅行,其工作记忆可能会保存目的地、日期和预算直到任务完成,而不会使长期存储混乱。
- **程序记忆**: 这帮助Agent学习和掌握例程。通过观察成功的工作流程,Agent可以内化重复任务的步骤序列,使其随着时间推移变得更快、更可靠。


#### 有效记忆管理的关键原则

**修剪和细化**:
![记忆修剪](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/98dfe89bbe328ee350f68c167c0840aada50b46dd8e60f3dd2490df111fcc4ce.jpg)

**选择性存储**:
![选择性存储](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/488278520dd2dd398111698354d99ffdc3d9cb03ed414c08d8450b1145348eae.jpg)

**掌握检索的艺术**:
![记忆检索优化](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/74f1e86909ece36d7e9c1a0a41ba6aa0db8c6468cc63de8efb8c6faacf4e45cc.jpg)

---

### 6. Tools - 工具集成

如果记忆给Agent自我意识,那么工具就是给它超能力的东西。

![工具集成概念](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/f3f7f9dcb519e9682438c5fc538771fc41d99e377c9faed968f6fe9300c57a94.jpg)

#### 从提示到行动的演变

使LLM具有工具使用能力的旅程经历了快速演变。最初,开发者试图通过传统的提示工程从LLM获得行动,通过诱导模型生成看起来像命令的文本。这很聪明但容易出错。

真正的突破是**函数调用(Function Calling)**,也称为**工具调用(Tool Calling)**。这种能力现在已成为大多数模型的原生功能,允许LLM输出可以包含要调用的函数名称和要使用的参数的结构化JSON。

有了这个能力,就有很多可能性:

**简单工具**:
旅行Agent机器人可以使用`search_flights`工具,当用户询问"帮我找下周二去东京的航班"时,LLM不会猜测答案。它生成对你提供的函数的调用,进而查询真实的航空公司API。

**工具链**:
对于像"帮我计划一个周末去旧金山的旅行"这样的复杂请求,Agent可能需要将多个工具链在一起:`find_flights`、`search_hotels`和`get_local_events`。这需要Agent进行推理、规划并执行多步骤工作流程。

上下文工程在这里的工作是如何呈现这些工具。一个写得好的工具描述就像一个小型提示,指导模型,清楚地说明工具的作用、需要什么输入以及返回什么。

#### 编排挑战

给Agent一个工具是容易的(大部分情况下)。让它可靠、安全和有效地使用该工具才是真正工作开始的地方。上下文工程的核心任务是**编排**,即在Agent推理使用哪个工具时管理信息流和决策制定。

这涉及在上下文窗口中发生的几个关键步骤。让我们使用Glowe(一个由我们的Elysia编排框架支持的护肤领域知识应用)作为运行示例来分解这些关键编排步骤:

**1. 工具发现**:
Agent需要知道它拥有哪些工具。这通常通过在系统提示中提供可用工具及其描述的列表来完成。这些描述的质量非常关键。它们是Agent理解每个工具作用的唯一指南,使模型能够理解何时使用工具,更重要的是,何时避免使用它。

在Glowe中,我们在初始化每个新聊天树时配置一组专门的工具(步骤5)并提供精确的描述。

**2. 工具选择和规划(思考)**:
面对用户请求时,Agent必须推理是否需要工具。如果需要,是哪一个?对于复杂任务,它甚至可能需要将多个工具链在一起,形成计划(例如,"首先,在网上搜索天气;然后,使用电子邮件工具发送摘要")。

![工具选择](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/cff6fea22bbef7d903f37f1fb759dab5a1a1bab97cf33cf539137dead8a7b4b2.jpg)

在这里,决策Agent正确分析了传入的请求并选择了product_agent工具。

**3. 参数制定(行动)**:
一旦选择了工具,Agent必须弄清楚传递什么参数给它。如果工具是`get_weather(city, date)`,Agent需要从用户的查询中提取"旧金山"和"明天"并正确格式化它们。这也可以是带有使用工具所需信息的结构化请求或API调用。

![参数制定](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/fc58cb57ba51d99005e7df6a3d7756fb73f580872d77f9136ebe2513568d3558.jpg)

在这种情况下,product_agent需要一个文本查询来搜索产品集合。注意Agent如何在生成初始导致错误的格式错误参数后自我修正(自我修复)(编排的另一个关键部分)。

**4. 反思(观察)**:
执行工具后,输出("观察")被反馈到上下文窗口中。然后Agent反思这个输出以决定下一步。工具成功了吗?它产生了回答用户查询所需的信息吗?还是返回了需要不同方法的错误?


![反思观察2](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/7c809b370b0444a739bedfbee6f5c29161bc3579d3802ce11db010a30e41e86a.jpg)

如你所见,编排通过这个强大的反馈循环发生,通常称为**思考-行动-观察循环(Thought-Action-Observation Cycle)**。

#### 思考-行动-观察循环

这个循环构成了现代Agent框架(如Elysia)中的基本推理循环。Agent观察其行动的结果,并使用这些新信息来推动其下一个"思考",决定任务是否完成、是否需要使用另一个工具,或者是否应该向用户寻求澄清。


#### 工具使用的下一个前沿

**传统集成 vs MCP方法**:
![传统集成 vs MCP方法](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/c0a0b2c5db6b23dc230ebd263a271050df29c28d48e67307a99485c498d4e3e3.jpg)

---

## 总结

上下文工程不仅仅是提示大语言模型、构建检索系统或设计AI架构。它是关于构建在各种用途和用户中可靠工作的互联、动态系统。

![简单提示工程 vs 上下文工程](https://cdn-mineru.openxlab.org.cn/result/2025-11-06/670cedc5-9554-4d29-a213-d1c3ec0d969f/6bbd38a502ba3f349cd249c6fcf657f5b5e909a5f6b83388f4da0fa989430af0.jpg)

上下文工程由以下组件组成:

- **Agents** - 作为系统的决策大脑
- **Query Augmentation** - 将混乱的人类请求转换为可操作的意图
- **Retrieval** - 将模型连接到事实和知识库
- **Memory** - 给你的系统历史感和学习能力
- **Tools** - 给你的应用程序与实时数据和API交互的手

我们正在从与模型对话的提示者转变为构建模型生活世界的架构师。**最好的AI系统不是来自更大的模型,而是来自更好的工程。**

---

## 参考资源

- Weaviate分块策略博客: https://weaviate.io/blog/chunking-strategies-for-rag
- Elysia Agentic RAG框架: https://weaviate.io/blog/elysia-agentic-rag
- Anthropic有效上下文工程指南
- Andrej Karpathy: Software Is Changing (Again)
- Model Context Protocol介绍: https://humanloop.com/blog/mcp
