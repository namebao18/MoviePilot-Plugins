# -*- coding: utf-8 -*-
"""
关键词搜索种子插件（KeywordSearch）

功能：
    为 MoviePilot 的 AI Agent 增加一个「按关键词直接搜索站点种子」的工具
    `search_torrents_by_keyword`，弥补内置 search_torrents 只能按媒体 ID
    （TMDB/豆瓣/Bangumi 等）精确搜索的局限。

背景：
    内置 search_torrents 依赖媒体识别后的 ID 匹配（async_search_by_id）。
    当资源标题与 TMDB 条目不匹配、或同一作品存在多个年份版本时，按 ID
    搜索会漏掉大量资源（例如搜「小鼠波波」只命中 1999 版 540p，漏掉
    2025 版全部 1080p）。本插件用关键词搜索（async_search_by_title）直接
    匹配站点种子标题，覆盖该类漏检。

实现要点：
    1. 插件实现 get_agent_tools() 返回工具类，被 MP 自动注册给 Agent；
    2. 工具内部调用 SearchChain().async_search_by_title(keyword, cache_local=True)；
    3. cache_local=True 会把结果写入「最近搜索结果」缓存，使 Agent 后续可
       继续用内置 get_search_results 筛选、add_download_tasks 下载，链路完整；
    4. 不改动 MoviePilot 任何源码，升级/重建不丢（走插件机制）。

版本：1.0.0
作者：local
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from app.agent.tools.base import MoviePilotTool
from app.agent.tools.impl._torrent_search_utils import (
    TORRENT_RESULT_LIMIT,
    build_filter_options,
    simplify_search_result,
)
from app.agent.tools.tags import ToolTag
from app.chain.search import SearchChain
from app.db.systemconfig_oper import SystemConfigOper
from app.helper.sites import SitesHelper
from app.log import logger
from app.plugins import _PluginBase
from app.schemas.types import SystemConfigKey


class SearchTorrentsByKeywordInput(BaseModel):
    """关键词搜索种子工具的输入参数模型"""

    keyword: str = Field(
        ...,
        description=(
            "Keyword to search torrent titles on indexer sites, matched literally "
            "against torrent titles (e.g., '小鼠波波', 'My Friend Maisy', 'Maisy'). "
            "Use the original Chinese/localized title as-is; do not convert to an ID."
        ),
    )
    page: Optional[int] = Field(
        1,
        description="Page number for pagination (default: 1, up to 50 results per page).",
    )
    sites: Optional[List[int]] = Field(
        None,
        description=(
            "Optional array of specific site IDs to search. If omitted, searches all "
            "configured indexer sites."
        ),
    )


class SearchTorrentsByKeywordTool(MoviePilotTool):
    """
    按关键词搜索站点种子的 Agent 工具。

    与内置 search_torrents（按媒体 ID 精确搜索）互补：
    - 当用户给出模糊/中文片名、或存在多个年份版本、或资源标题无法被识别为
      媒体 ID 时，用本工具按关键词直接匹配站点种子标题。
    """

    name: str = "search_torrents_by_keyword"
    tags: list[str] = [
        ToolTag.Read,
        ToolTag.Resource,
        ToolTag.Site,
        ToolTag.Media,
    ]
    description: str = (
        "Search torrent files by keyword across configured indexer sites, matching "
        "the keyword directly against torrent titles without requiring a media ID. "
        "Use this when searching by a fuzzy or localized title, when the media has "
        "multiple year versions, or when the built-in search_torrents (ID-based) "
        "returns too few or no results. Results are cached so get_search_results and "
        "add_download_tasks can be used afterwards. "
        "IMPORTANT: Results ≤ 2 do NOT mean the resource is unavailable; try other "
        "keywords, alternate titles, or site scope before concluding."
    )
    args_schema: Type[BaseModel] = SearchTorrentsByKeywordInput

    def get_tool_message(self, **kwargs) -> Optional[str]:
        """返回工具执行提示"""
        keyword = kwargs.get("keyword", "")
        sites = kwargs.get("sites")
        message = f"关键词搜索种子: {keyword}"
        if sites:
            message += f" [站点: {sites}]"
        return message

    async def run(
        self,
        keyword: str,
        page: Optional[int] = 1,
        sites: Optional[List[int]] = None,
        **kwargs,
    ) -> str:
        """
        按关键词搜索站点种子并写入最近搜索结果缓存。

        :param keyword: 搜索关键词（直接匹配种子标题）
        :param page: 页码（工具内部做分页展示，搜索本身返回全量）
        :param sites: 指定站点 ID 列表，为空则搜索全部已配置站点
        :param kwargs: 工具框架附加参数
        :return: JSON 格式搜索结果，或错误提示
        """
        keyword = (keyword or "").strip()
        page = max(1, page or 1)
        logger.info(
            f"执行工具: {self.name}, 参数: keyword={keyword}, page={page}, sites={sites}"
        )

        if not keyword:
            return "参数错误：keyword 不能为空，请提供要搜索的关键词（如 '小鼠波波'）。"

        try:
            search_chain = SearchChain()
            # 按关键词搜索，cache_local=True 会把结果写入最近搜索结果缓存，
            # 使后续 get_search_results / add_download_tasks 可以直接复用。
            contexts = await search_chain.async_search_by_title(
                title=keyword,
                page=0,
                sites=sites,
                cache_local=True,
            ) or []

            # 站点信息，便于 Agent 了解搜索范围
            all_indexers = await SitesHelper().async_get_indexers()
            all_sites = [
                {"id": indexer.get("id"), "name": indexer.get("name")}
                for indexer in (all_indexers or [])
            ]
            search_site_ids = sites or (
                SystemConfigOper().get(SystemConfigKey.IndexerSites) or []
            )

            if not contexts:
                payload = {
                    "total_count": 0,
                    "keyword": keyword,
                    "message": (
                        f"关键词“{keyword}”未搜索到种子资源。搜索结果为空不代表资源不存在："
                        "可尝试其他关键词/别名、切换站点范围，或改用 search_torrents 按媒体 ID 搜索。"
                    ),
                    "all_sites": all_sites,
                    "search_site_ids": search_site_ids,
                }
                return json.dumps(payload, ensure_ascii=False, indent=2)

            total_count = len(contexts)
            page_size = TORRENT_RESULT_LIMIT
            total_pages = (total_count + page_size - 1) // page_size
            start = (page - 1) * page_size
            end = start + page_size
            page_items = contexts[start:end]
            page_indices = list(range(start + 1, start + 1 + len(page_items)))

            results = [
                simplify_search_result(
                    context, index, include_description=False
                )
                for context, index in zip(page_items, page_indices)
            ]

            payload = {
                "total_count": total_count,
                "keyword": keyword,
                "page": page,
                "total_pages": total_pages,
                "results": results,
                "filter_options": build_filter_options(contexts),
                "all_sites": all_sites,
                "search_site_ids": search_site_ids,
                "message": (
                    "关键词搜索完成，结果已写入最近搜索结果缓存；"
                    "可继续使用 get_search_results 做筛选、add_download_tasks 下载。"
                ),
            }
            if page < total_pages:
                payload["message"] += (
                    f" 当前第 {page}/{total_pages} 页，可用 page={page + 1} 获取下一页。"
                )
            return json.dumps(payload, ensure_ascii=False, indent=2)

        except Exception as e:
            error_message = f"关键词搜索种子失败: {str(e)}"
            logger.error(f"关键词搜索种子失败: {e}", exc_info=True)
            return error_message


class KeywordSearch(_PluginBase):
    """关键词搜索种子插件（为 Agent 提供 search_torrents_by_keyword 工具）"""

    # 插件元信息
    plugin_name = "关键词搜索种子"
    plugin_desc = (
        "为 AI Agent 提供按关键词直接搜索站点种子的工具，弥补内置按媒体 ID 搜索的漏检问题。"
    )
    plugin_icon = "search.png"
    plugin_version = "1.0.0"
    plugin_author = "local"
    plugin_config_prefix = "keywordsearch_"
    plugin_order = 50
    auth_level = 1

    _enabled = False

    def init_plugin(self, config: dict = None) -> None:
        """根据插件配置初始化运行状态"""
        self._enabled = False
        if not config:
            return
        self._enabled = bool(config.get("enabled"))

    def get_state(self) -> bool:
        """获取插件启用状态"""
        return self._enabled

    def get_agent_tools(self) -> List[Type]:
        """
        向 Agent 注册工具。

        只有插件启用时才提供工具，避免对 Agent 造成无谓的工具占用。
        """
        if not self._enabled:
            return []
        return [SearchTorrentsByKeywordTool]

    @staticmethod
    def get_command() -> List[Dict[str, Any]]:
        """返回插件远程命令列表"""
        return []

    def get_api(self) -> List[Dict[str, Any]]:
        """返回插件 API 列表"""
        return []

    def get_form(self) -> tuple:
        """返回插件配置表单与默认配置"""
        return [
            {
                "component": "VForm",
                "content": [
                    {
                        "component": "VSwitch",
                        "props": {
                            "model": "enabled",
                            "label": "启用插件",
                        },
                    },
                    {
                        "component": "VAlert",
                        "props": {
                            "type": "info",
                            "text": (
                                "启用后，AI Agent 将获得 search_torrents_by_keyword 工具，"
                                "可按关键词（中文片名等）直接搜索站点种子，"
                                "用于弥补内置 search_torrents 只能按媒体 ID 精确搜索导致的漏检。"
                            ),
                        },
                    },
                ],
            }
        ], {
            "enabled": False,
        }

    def get_page(self) -> Optional[List[dict]]:
        """返回插件详情页面"""
        if not self._enabled:
            return None
        return [
            {
                "component": "VAlert",
                "props": {
                    "type": "info",
                    "text": "插件已启用。Agent 可按关键词搜索站点种子。",
                },
            }
        ]

    def stop_service(self) -> None:
        """停止插件后台服务并释放资源"""
        pass
