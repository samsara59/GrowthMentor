
# GrowthMentor-Agent 架构

## 1. 产品定义

GrowthMentor-Agent 是一个长期、可扩展的个人 AI 成长导师。

系统根据用户的成长目标，组合有来源的思想方法、
专业能力、历史案例、个人记忆和实践反馈，
通过持续对话帮助用户认识自己、作出选择、
采取行动并修正认识。

首个应用目标：
帮助用户发现喜欢、擅长和重视的事情，
并通过真实实践探索值得长期投入的方向。

长期目标不限制为优势评估、职业推荐或人格测评。

## 2. 系统分层

### Domain

app/domain/

定义不依赖特定理论的数据协议：

- GrowthTask
- UserEvidence
- KnowledgeEvidence
- GrowthHypothesis
- GrowthOutcome

### Programs

app/programs/

注册成长任务，描述：

- 任务目标
- 实施阶段
- 可选择的方法
- 当前实现状态

示例：
- self_discovery
- reflective_decision
- historical_learning

### Methods

app/methods/

管理可以独立替换的导师方法：

- 作者的自我探索方法
- 本地成长与优势课程
- 哲学与系统思维方法
- 历史人物成长路径研究方法

方法注册不代表来源已经核实。

### Capabilities

app/capabilities/

管理可执行能力及其调用：

- 知识检索
- 个人记忆读取
- 证据验证
- 未来其他工具

区分：
1. 已声明
2. 已绑定执行函数
3. 已实际执行

### Agent Runtime

app/agent/

管理 LangGraph、Supervisor 和工作流执行。

Supervisor 不直接写死某个学派或固定成长阶段。

旧版工作流作为兼容实现逐步迁移。

### Knowledge Infrastructure

app/ingestion/
app/knowledge/
app/rag/

负责：
- 教材解析
- 知识存储
- 向量检索
- 来源定位
- 引文核查

RAG 默认是共享工具，不必独立成为一个 Agent。

### Evidence and Memory

app/assessment/
app/memory/

分别处理：
- 证据与假设的核查
- 成长任务、长期记录和状态管理

### UI

app/ui/

展示：
- 自然对话
- 成长任务
- 导师方法
- 个人成长记录
- 理论依据
- 现实行动及反馈
- 工作流执行状态

## 3. 不可混淆的三类数据

1. 用户自述和实际行动记录
2. 教材、历史资料和其他外部知识
3. 模型提出的解释、建议与假设

不能把教材观点当作用户个人证据。
不能把模型猜测当作已验证事实。

## 4. 首个 Growth Program

self_discovery

任务：
探索喜欢、擅长、重视的事情。

参考方法：
八木仁平的自我理解方法和本地优势课程。

目标结果：
形成可被现实实践验证、修正的方向假设。

## 5. 其他 Growth Programs

reflective_decision

在具体条件下分析现实困境、制约与行动。

historical_learning

基于史料研究历史人物成长轨迹、
所处条件、失败、反馈和策略。

不得假装历史人物真实参与了对话。

## 6. 兼容与迁移策略

原有八个app模块暂时保留。

新增：
- app/domain
- app/programs
- app/methods
- app/capabilities

第一阶段先建立稳定接口和注册表。

第二阶段适配现有RAG、Memory及LangGraph。

第三阶段接入第一个完整成长任务。

旧版SQLite和用户经历不得因架构改造而直接删除。

## 7. 本地知识与隐私

books/和data/默认属于私有内容。

不得把教材原文、用户个人经历、数据库或密钥
上传到公开GitHub仓库。

教材原始文件解析成功，
不等于向量知识库已经完成索引。

方法目录的来源状态必须经过实际验证。

## 8. 架构验收

在相同的Domain协议下，
至少支持以下两类不同任务的表达：

A. 发现自己喜欢且擅长的事情
B. 借助不同方法分析现实困境

新增任务和方法不应要求修改基础设施层。

只有接入实际工作流并通过集成测试的能力，
才可以标记为正式可用。
