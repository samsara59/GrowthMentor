
class GrowthMentorError(Exception):
    """整个项目的业务异常基类。"""


class ConfigurationError(GrowthMentorError):
    """应用配置不合法。"""


class ModelServiceError(GrowthMentorError):
    """Ollama或其他模型服务发生错误。"""


class EvidenceValidationError(GrowthMentorError):
    """行为证据或原文引用未通过验证。"""


class MemoryPersistenceError(GrowthMentorError):
    """数据库读取或保存失败。"""


class WorkflowError(GrowthMentorError):
    """Agent工作流状态或执行发生错误。"""
