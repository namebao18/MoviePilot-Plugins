import inspect
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.event import eventmanager
from app.plugins import _PluginBase
from app.schemas.types import ChainEventType
from app.log import logger


class LLMProviderFix(_PluginBase):
    """
    LLM 供应商修复插件。

    确保 Agent（智能助手）的 LLM 请求始终使用系统 LLM 配置（DeepSeek 等），
    不占用 Agent Tokens 管理插件的供应商配额，避免 Agent 与 ChatGPT 等插件
    的 Token 使用互相干扰。

    原理：在 AgentLLMProvider 链式事件中，以优先级 40 抢先于 AgentTokens（优先级 50）
    处理。通过调用栈检测识别出是 Agent 模块发起的请求后，直接写入系统 LLM 配置并标记
    selected_provider_id，使 AgentTokens 管理插件跳过对该事件的处理。
    """

    plugin_name = "LLM 供应商修复"
    plugin_desc = (
        "确保 Agent 始终使用系统 LLM 配置，不受 Agent Tokens 管理插件影响。"
        "Agent 使用智能助手时自动走 DeepSeek 等系统配置，不再占用 ChatGPT 等插件的 Token 配额。"
    )
    plugin_icon = "llmfix.png"
    plugin_version = "2.0.0"
    plugin_label = "系统工具"
    plugin_author = "local"
    plugin_config_prefix = "llmproviderfix_"
    plugin_order = 100
    auth_level = 1

    _enabled = False

    def init_plugin(self, config: dict = None) -> None:
        """根据插件配置初始化运行状态。"""
        self._enabled = False
        if not config:
            return
        self._enabled = bool(config.get("enabled"))

    def get_state(self) -> bool:
        """获取插件启用状态。"""
        return self._enabled

    @staticmethod
    def get_command() -> List[Dict[str, Any]]:
        """返回插件远程命令列表。"""
        return []

    def get_api(self) -> List[Dict[str, Any]]:
        """返回插件 API 列表。"""
        return []

    def get_form(self) -> Tuple[Optional[List[dict]], Dict[str, Any]]:
        """返回插件配置表单与默认配置。"""
        return [
            {
                "component": "VForm",
                "content": [
                    {
                        "component": "VSwitch",
                        "props": {
                            "model": "enabled",
                            "label": "启用插件"
                        }
                    },
                    {
                        "component": "VAlert",
                        "props": {
                            "type": "info",
                            "text": (
                                "启用后，Agent 的 LLM 请求将始终使用系统配置，"
                                "不再经过 Agent Tokens 管理插件分配供应商。"
                                "这可以避免 Agent 占用 ChatGPT 等插件的 Token 配额，"
                                "让 Agent 和 ChatGPT 各自独立运行。"
                            )
                        }
                    }
                ]
            }
        ], {
            "enabled": False
        }

    def get_page(self) -> Optional[List[dict]]:
        """返回插件详情页面。"""
        if not self._enabled:
            return None
        return [
            {
                "component": "VAlert",
                "props": {
                    "type": "info",
                    "text": "插件已启用。Agent 的 LLM 请求将使用系统配置。"
                }
            }
        ]

    @eventmanager.register(ChainEventType.AgentLLMProvider, priority=40)
    def fix_llm_provider(self, event: Any) -> None:
        """
        处理 AgentLLMProvider 链式事件。

        在 AgentTokens 插件（优先级 50）之前运行。通过调用栈检测判断请求来源：
        - 若来自 Agent 模块，直接写入系统 LLM 配置并标记 selected_provider_id，
          使后续的 AgentTokens 插件跳过处理。
        - 若来自 ChatGPT 等其他插件，不做干涉，让 AgentTokens 正常处理。
        """
        if not self._enabled or not event or not event.event_data:
            return

        # 如果已经有其他插件设置了 selected_provider_id，跳过
        if self._event_get(event.event_data, "selected_provider_id"):
            return

        # 检测请求来源
        if not self._is_called_from_agent():
            return

        # 写入系统 LLM 配置
        self._event_set(event.event_data, "provider", settings.LLM_PROVIDER or "openai")
        self._event_set(event.event_data, "model", settings.LLM_MODEL)
        self._event_set(event.event_data, "api_key", settings.LLM_API_KEY)
        self._event_set(event.event_data, "base_url", settings.LLM_BASE_URL or "")
        self._event_set(event.event_data, "base_url_preset", settings.LLM_BASE_URL_PRESET or "")
        self._event_set(event.event_data, "user_agent", settings.LLM_USER_AGENT or "")
        self._event_set(
            event.event_data, "use_proxy",
            bool(settings.LLM_USE_PROXY) if settings.LLM_USE_PROXY is not None else True
        )

        # 标记为已处理，使 AgentTokens 插件跳过
        self._event_set(event.event_data, "selected_provider_id", "system")
        self._event_set(event.event_data, "selected_provider_name", "系统配置")
        self._event_set(event.event_data, "source", self.__class__.__name__)

        logger.info(
            "LLMProviderFix: Agent LLM 请求已固定为系统配置 (Provider=%s, Model=%s)",
            settings.LLM_PROVIDER,
            settings.LLM_MODEL,
        )

    @staticmethod
    def _is_called_from_agent() -> bool:
        """
        检测当前事件调用是否来自 Agent 模块。

        通过检查 Python 调用栈来确定事件是由 Agent._resolve_llm_runtime_config
        还是由 ChatGPT._resolve_agent_tokens_model_config 等方法发起的。
        Agent 使用 async_send_event 异步调用，ChatGPT 使用 send_event 同步调用，
        两者调用链不同。
        """
        try:
            for frame_info in inspect.stack():
                module = inspect.getmodule(frame_info.frame)
                if module and module.__file__ and "app/agent/__init__.py" in module.__file__:
                    return True
            return False
        except Exception:
            # 安全检查：无法检测时回退到 True（保守处理，走系统配置）
            return True

    @staticmethod
    def _event_get(event_data: Any, key: str, default: Any = None) -> Any:
        """兼容读取 Pydantic 事件模型或字典中的字段。"""
        if isinstance(event_data, dict):
            return event_data.get(key, default)
        return getattr(event_data, key, default)

    @staticmethod
    def _event_set(event_data: Any, key: str, value: Any) -> None:
        """兼容写入 Pydantic 事件模型或字典中的字段。"""
        if isinstance(event_data, dict):
            event_data[key] = value
        else:
            setattr(event_data, key, value)

    def stop_service(self) -> None:
        """停止插件后台服务并释放资源。"""
        pass
