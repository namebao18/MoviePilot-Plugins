# AI字幕自动生成(成人特调) — AutoSubv2AV

基于 [AutoSubv2](https://github.com/jxxghp/MoviePilot-Plugins) v3.0.0 魔改，针对**成人影片库**场景特调。

## 与上游的差异

| 特性 | 说明 |
|---|---|
| **NFO 语言判断** | 读同目录 .nfo 的 countrycode/studio/originaltitle 判断语言（ja/ko/zh/en），比音频探测更准更快 |
| **中文广告跳过** | 非中文片用 clip_timestamps 跳过开头 N 秒（默认 300s），避开成人片开头的中文广告导致的语言误判 |
| **NFO 中文字幕跳过** | NFO 已标注「中文字幕」的片直接跳过，不浪费算力 |
| **板块过滤** | 可限定只处理指定板块（如 霓虹/欧美） |
| **翻译外置** | 默认不做翻译（	ranslate_zh=false），字幕交青龙脚本按语言分流翻译 |
| **默认扫描 7 个正式库** | 霓虹/欧美/动漫/传媒/三级/韩国/探花 |

## 核心逻辑

### 语言判定（3 层）
1. **NFO**：countrycode=JP→ja、KR→ko、TW/CN/HK→zh、US/GB→en；无国家代码则看 studio/originaltitle 是否含假名
2. **板块兜底**：霓虹/动漫→ja，欧美→en，韩国→ko，探花/传媒/三级→zh
3. **whisper 自动**（前两层都失败时）

### 广告处理
`
非中文片（ja/en/ko）→ clip_timestamps=300（跳过开头5分钟，避开中文广告）
中文片（zh）        → clip_timestamps=0（从头，不跳）
`
> clip_timestamps 是 faster-whisper 原生参数，时间戳自动对齐，无需手动偏移。

## 配置项

| 配置 | 默认 | 说明 |
|---|---|---|
| 启用插件 | - | |
| 媒体入库自动执行 | true | 监听 MP 入库事件 |
| 手动执行一次 | false | 全量扫描 |
| 媒体路径 | 7 个正式库 | |
| 板块过滤 | 空（全部） | 逗号分隔 |
| 语言判定来源 | NFO优先 | nfo_first/board/auto |
| 广告跳过秒数 | 300 | 仅非中文片 |
| NFO有中文字幕则跳过 | true | |
| 文件大小 | 5 MB | |
| whisper模型 | large-v3 | |
| CPU线程数 | 2 | |
| 计算类型 | float32 | |
| 允许从音轨生成字幕 | true | |
| 翻译成中文 | **false** | 交青龙脚本 |

## 依赖
`
iso639~=0.1.4
srt~=3.5.3
faster-whisper~=1.0.1
cacheout~=0.16.0
`

## 作者
原插件作者 TimoYoung；成人场景特调 by namebao18
