
# GrowthMentor-Agent

一个本地优先、可扩展的个人 AI 成长导师工程。

## 产品目标

根据用户的成长目标，组合思想方法、
个人经历、可靠知识与现实实践反馈，
帮助用户长期认识自己、决策、行动和成长。

第一个 Growth Program：

发现真正喜欢、擅长和重视的事情，
并通过实践探索值得投入的方向。

这不是一次性人格测评或自动职业判定工具。

## 核心架构

- app/domain/：跨领域成长任务协议
- app/programs/：成长任务目录
- app/methods/：导师方法目录
- app/capabilities/：能力注册与执行接口
- app/agent/：Supervisor与LangGraph
- app/assessment/：证据分析
- app/memory/：SQLite用户记忆
- app/ingestion/：教材解析
- app/knowledge/：知识管理
- app/rag/：本地知识检索
- app/core/：共享基础设施
- app/ui/：Streamlit界面

更多架构说明见 architecture.md。

## 当前开发状态

已有第一版：

- 教材导入和知识问答
- 用户经历收集
- 单经历与跨经历分析
- 动态追问验证
- SQLite持久化
- LangGraph调度
- Streamlit可视化

新增架构接口：

- GrowthTask和GrowthOutcome
- ProgramCatalog
- MethodCatalog
- CapabilityRegistry

旧版业务工作流尚未全面迁移到新协议。

新增的现实决策、历史人物成长研究等任务，
当前只是目录定义，不能视为完整功能。

## 本地教材

原始教材放在 books/。

解析及知识库文件放在 data/。

首次架构迁移不重新导入教材，
也不修改已有用户记忆。

需要单独验证教材是否已经实际进入
knowledge.db和Chroma，以及能否正确引用来源。

## 运行架构检查

Windows PowerShell：

python -m scripts.inspect_architecture

## 运行测试

python -m unittest discover -s tests -p "test_*.py" -v

## 启动旧版可视化原型

python -m streamlit run app/ui/streamlit_app.py

## 隐私与开源

不得公开上传：

- books/
- data/
- .env
- 私人对话和用户经历
- 私有教材内容
- 密钥或未脱敏的日志

当前是本地教学和开发原型。
尚未实现完善的账户认证、并发隔离或数据加密。

## 开发原则

先保持既有功能正常运行，再逐步迁移架构。

业务任务、导师理论和系统工具相互解耦。

教材知识、用户事实和模型推断分开保存。

所有成长建议应允许用户质疑、纠正和撤回。

长期以实际行动反馈修正成长方向。
