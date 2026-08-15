"""类型化异常：把配置 / 阶段检测 / 数据 / 策略的错误分类，便于上层捕获。"""


class StageVLAError(Exception):
    """项目异常基类。"""


class ConfigError(StageVLAError):
    """配置缺失、冲突或校验失败。"""


class StageDetectionError(StageVLAError):
    """阶段检测器输入非法或状态不满足判定前提。"""


class DemoNotFoundError(StageVLAError):
    """演示数据缺失或格式不合法（采集链断链时抛出）。"""


class PolicyNotFoundError(StageVLAError):
    """请求的策略后端未注册（policies.factory 查表失败）。"""


class TaskNotFoundError(StageVLAError):
    """请求的仿真任务/网关未注册。"""
