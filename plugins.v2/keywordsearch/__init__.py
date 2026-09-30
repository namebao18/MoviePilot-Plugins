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

多版本/多类型处理（v1.2.0）：
    关键词搜索会同时带回"电视剧多季 / 电影 / 动漫 / 同名无关内容"混合结果。
    ⚠️ 关键事实：**按标题搜索不做媒体识别**，所以结果的 meta_info.type 基本
    都是 UNKNOWN，无法直接区分电影/剧集。因此本工具：
    - 默认（recognize=false）：不做识别，用标题启发式粗略标注 media_kind
      （movie/tv/anime/unknown），并置 type_uncertain=true 提示"类型不确定"。
    - recognize=true：对**去重后的标题**并发调用 MP 媒体识别
      （async_recognize_by_meta），补全正式类型/年份/季；**并发 + 单条超时**，
      避免逐条串行导致的长时间阻塞。
    - media_type：按 media_kind 过滤结果（movie/tv/anime/auto）。

性能教训（v1.1.0 → v1.2.0）：
    v1.1.0 对每条结果**串行**识别，26 条结果导致工具执行超过 300 秒被强制
    停止。v1.2.0 改为：① 按解析标题去重后只识别唯一标题；② 并发识别；
    ③ 单条超时；④ 默认不识别。

刮削入库说明：
    下载阶段的媒体识别由 add_download_tasks 独立完成（recognize_by_meta），
    最终入库刮削由下载完成后的 transfer 流程按真实文件做。本搜索工具不影响
    入库刮削，只影响"搜到什么、怎么展示"。

版本：1.2.0
作者：local
"""

import asyncio
import json
import re
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
from app.schemas.types import MediaType, SystemConfigKey


# 动漫相关标签/关键词（MP 无独立"动漫"类型，靠标签或标题判断）
_ANIME_KEYWORDS = (
    "动漫", "动画", "アニメ", "anime", "番剧", "国漫", "日漫", "漫改", "里番",
)
# 单条媒体识别的超时（秒），避免个别条目卡住整体
_RECOGNIZE_TIMEOUT = 12


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
    media_type: Optional[str] = Field(
        "auto",
        description=(
            "Optional result filter by media kind. Allowed values: 'movie', 'tv', "
            "'anime', 'auto' (default, no filter). Filtering is most reliable when "
            "recognize=true; otherwise it uses a title heuristic and may be inaccurate. "
            "Use this when the same name has mixed versions (e.g. TV + movie + anime)."
        ),
    )
    recognize: Optional[bool] = Field(
        False,
        description=(
            "When true, run MoviePilot media recognition on unique result titles "
            "(concurrently, with per-item timeout) to fill the exact type/year and "
            "mark whether it can be recognized for library import. More accurate for "
            "disambiguating same-name multi-version results, but slower. Default false."
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
        "IMPORTANT: keyword search does NOT identify media, so the type field is "
        "usually unknown; each result is annotated with media_kind plus type_uncertain. "
        "When the same name returns mixed versions (TV seasons / movie / anime), set "
        "recognize=true to label each unique title with its exact type/year/season "
        "(slower), or set media_type to 'movie'/'tv'/'anime' to filter. "
        "IMPORTANT: Results ≤ 2 do NOT mean the resource is unavailable; try other "
        "keywords, alternate titles, or site scope before concluding."
    )
    args_schema: Type[BaseModel] = SearchTorrentsByKeywordInput

    def get_tool_message(self, **kwargs) -> Optional[str]:
        """返回工具执行提示"""
        keyword = kwargs.get("keyword", "")
        media_type = kwargs.get("media_type")
        sites = kwargs.get("sites")
        message = f"关键词搜索种子: {keyword}"
        if media_type and media_type != "auto":
            message += f" [{media_type}]"
        if sites:
            message += f" [站点: {sites}]"
        return message

    @staticmethod
    def _heuristic_kind(title: str, category: Optional[str], labels: Optional[List[str]]) -> tuple:
        """
        标题 + 站点分类/标签的启发式类型判断（不访问网络）。

        :return: (kind, uncertain) kind ∈ movie/tv/anime/unknown；uncertain 表示不确定
        """
        text = (title or "").lower()
        label_text = " ".join(labels or [])
        if any(kw.lower() in text for kw in _ANIME_KEYWORDS) or any(
            kw in label_text for kw in ("动画", "动漫", "anime")
        ):
            return "anime", False
        # 有明确的季集标记 -> 剧集
        if re.search(r"[sS]\d{1,2}(\b|E\d+)|第\d+季|EP?\d{1,3}\b", title or ""):
            return "tv", False
        # 站点分类字段（中文）作为弱线索
        cat = category or ""
        if cat in ("电影", "Movie"):
            return "movie", True
        if cat in ("电视剧", "TV", "剧集"):
            return "tv", True
        return "unknown", True

    async def _recognize_unique_titles(
        self, contexts: List[Any]
    ) -> Dict[str, Any]:
        """
        对去重后的标题并发做媒体识别，返回 {识别键: MediaInfo}。

        去重键 = meta_info.name/标题，避免对同部片的大量重复条目重复识别。
        并发 + 单条超时，避免整体阻塞。
        """
        from app.chain.media import MediaChain

        # 收集唯一标题（用解析出的 name 优先，否则用 torrent title）
        targets: Dict[str, Any] = {}
        for context in contexts:
            mi = getattr(context, "meta_info", None)
            if not mi:
                continue
            key = (getattr(mi, "name", None) or getattr(mi, "title", None) or "").strip()
            if key and key not in targets:
                targets[key] = mi
        if not targets:
            return {}

        media_chain = MediaChain()

        async def recognize_one(key: str, meta_info: Any) -> tuple:
            try:
                result = await asyncio.wait_for(
                    media_chain.async_recognize_by_meta(meta_info, obtain_images=False),
                    timeout=_RECOGNIZE_TIMEOUT,
                )
                return key, result
            except asyncio.TimeoutError:
                logger.warning(f"关键词搜索深度识别超时（{_RECOGNIZE_TIMEOUT}s）：{key}")
                return key, None
            except Exception as e:
                logger.warning(f"关键词搜索深度识别失败：{key} - {e}")
                return key, None

        tasks = [recognize_one(k, v) for k, v in targets.items()]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return {key: media for key, media in results if media}

    @staticmethod
    def _build_entry(
        context: Any,
        index: int,
        media_kind: str,
        uncertain: bool,
        media_info: Any,
    ) -> Dict[str, Any]:
        """
        构造单条结果输出：内置精简字段 + 分类标注。

        :param context: 搜索结果上下文
        :param index: 在原始缓存中的序号
        :param media_kind: 类型（movie/tv/anime/unknown）
        :param uncertain: 类型是否不确定
        :param media_info: 深度识别得到的 MediaInfo（可能为 None）
        :return: 输出字典
        """
        simplified = simplify_search_result(context, index, include_description=False)
        simplified["media_kind"] = media_kind
        simplified["type_uncertain"] = uncertain
        recognized = bool(media_info) or bool(getattr(context, "media_info", None))
        simplified["recognizable"] = recognized
        if media_info:
            simplified["recognized_media"] = {
                "title": media_info.title,
                "year": media_info.year,
                "type": media_info.type.value if media_info.type else None,
                "season": media_info.season,
                "tmdb_id": media_info.tmdb_id,
            }
        return simplified

    async def run(
        self,
        keyword: str,
        page: Optional[int] = 1,
        sites: Optional[List[int]] = None,
        media_type: Optional[str] = "auto",
        recognize: Optional[bool] = False,
        **kwargs,
    ) -> str:
        """
        按关键词搜索站点种子并写入最近搜索结果缓存。

        :param keyword: 搜索关键词（直接匹配种子标题）
        :param page: 页码（工具内部做分页展示，搜索本身返回全量）
        :param sites: 指定站点 ID 列表，为空则搜索全部已配置站点
        :param media_type: 结果类型过滤（movie/tv/anime/auto）
        :param recognize: 是否对去重后的标题做并发深度媒体识别
        :param kwargs: 工具框架附加参数
        :return: JSON 格式搜索结果，或错误提示
        """
        keyword = (keyword or "").strip()
        page = max(1, page or 1)
        media_type = (media_type or "auto").strip().lower()
        if media_type not in ("movie", "tv", "anime", "auto"):
            media_type = "auto"
        logger.info(
            f"执行工具: {self.name}, 参数: keyword={keyword}, page={page}, "
            f"sites={sites}, media_type={media_type}, recognize={recognize}"
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

            # 深度识别（可选）：仅对去重后的标题并发识别
            recognized_map: Dict[str, Any] = {}
            if recognize:
                recognized_map = await self._recognize_unique_titles(contexts)

            # 为每条结果标注 media_kind
            kind_map: Dict[int, str] = {}
            uncertain_map: Dict[int, bool] = {}
            for idx, context in enumerate(contexts):
                mi = getattr(context, "meta_info", None)
                ti = getattr(context, "torrent_info", None)
                title = (ti.title if ti else "") or ""
                category = getattr(ti, "category", None) if ti else None
                labels = getattr(ti, "labels", None) if ti else None
                # 优先用深度识别结果
                media_info = None
                if mi:
                    key = (getattr(mi, "name", None) or getattr(mi, "title", None) or "").strip()
                    media_info = recognized_map.get(key)
                    # 回填到 context，保持与缓存一致（便于后续工具复用）
                    if media_info and not getattr(context, "media_info", None):
                        context.media_info = media_info
                if media_info and getattr(media_info, "type", None):
                    kind = "movie" if media_info.type == MediaType.MOVIE else "tv"
                    # 动漫若无独立类型（归 TV），用标题/标签再判一次
                    if kind == "tv":
                        hk, _ = self._heuristic_kind(title, category, labels)
                        if hk == "anime":
                            kind = "anime"
                    kind_map[idx] = kind
                    uncertain_map[idx] = False
                else:
                    kind, uncertain = self._heuristic_kind(title, category, labels)
                    kind_map[idx] = kind
                    uncertain_map[idx] = uncertain

            # 按类型过滤（保留原始索引，保证 hash:id 与缓存一致）
            def keep(idx: int) -> bool:
                if media_type == "auto":
                    return True
                return kind_map.get(idx) == media_type

            selected = [(idx, ctx) for idx, ctx in enumerate(contexts) if keep(idx)]
            total_count = len(selected)

            # 类型分布
            dist: Dict[str, int] = {}
            for idx in kind_map:
                dist[kind_map[idx]] = dist.get(kind_map[idx], 0) + 1

            if total_count == 0:
                payload = {
                    "total_count": 0,
                    "keyword": keyword,
                    "filtered_by": media_type,
                    "media_kind_distribution": dist,
                    "message": (
                        f"关键词“{keyword}”共搜到 {len(contexts)} 条，但均不属于类型 "
                        f"“{media_type}”。类型分布见 media_kind_distribution，"
                        "可改用 media_type=auto 查看全部；若类型不准，可加 recognize=true 提高准确度。"
                    ),
                    "all_sites": all_sites,
                    "search_site_ids": search_site_ids,
                }
                return json.dumps(payload, ensure_ascii=False, indent=2)

            # 分页（针对过滤后的结果）
            page_size = TORRENT_RESULT_LIMIT
            total_pages = (total_count + page_size - 1) // page_size
            start = (page - 1) * page_size
            end = start + page_size
            page_selected = selected[start:end]

            results = []
            for idx, ctx in page_selected:
                key = None
                mi = getattr(ctx, "meta_info", None)
                if mi:
                    key = (getattr(mi, "name", None) or getattr(mi, "title", None) or "").strip()
                media_info = recognized_map.get(key) if key else None
                results.append(
                    self._build_entry(
                        ctx, idx + 1, kind_map.get(idx, "unknown"),
                        uncertain_map.get(idx, True), media_info,
                    )
                )

            payload = {
                "total_count": total_count,
                "keyword": keyword,
                "filtered_by": media_type,
                "media_kind_distribution": dist,
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
            if not recognize:
                payload["message"] += (
                    " 注意：本次未做媒体识别，media_kind 为启发式结果（type_uncertain=true 表示不确定）；"
                    "如需精确类型/年份，请加 recognize=true。"
                )
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
        "支持按 movie/tv/anime 过滤与并发深度识别，便于处理同名多版本（多季/电影/动漫）结果。"
    )
    plugin_icon = "search.png"
    plugin_version = "1.2.0"
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
                                "支持 media_type 过滤与 recognize 并发深度识别。"
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
