# my-deepswe

DeepSWE 数据浏览器（heatmap / tasks / trials）的**零依赖干净重写版**。
纯原生 JS + HTML + CSS，没有后端，所有数据来自 `artifacts/` 下的静态 JSON。

## 快速开始

```bash
cd my-deepswe
python3 -m http.server 8731
# 打开 http://localhost:8731/
```

> 必须用 HTTP 服务，不能直接双击 `index.html`（`file://` 会被 CORS 拦）。

仓库自带一份真实的 DeepSWE 数据作为示例（113 tasks / 14k trials），开箱即可浏览。

## 三个视图

- **Heatmap** — tasks × models 透视表。可切换着色指标（通过率 / 成本 / 步数 / token / 时长）、
  Split-effort vs Base-model、排除报错、数字 vs 纯色；点单元格跳到对应筛选的 trials。
- **Tasks** — 任务目录，搜索 + 语言筛选；点进去看 instruction / patch / Dockerfile / 各模型成绩。
- **Trials** — 全部 rollout 的可筛选/可排序表格；点进去看完整 agent 轨迹。

## 换成你自己的数据

**只需替换 `artifacts/` 下的 JSON，不用改任何代码。** 完整字段说明见
[DATA_FORMAT.md](DATA_FORMAT.md)。

想先看一个最小可用示例的长相：

```bash
python3 make_sample_data.py          # 生成 sample_artifacts/
cp -r sample_artifacts/* artifacts/  # 覆盖示例数据
python3 -m http.server 8731
```

## 文件结构

| 路径 | 作用 |
|---|---|
| `index.html` / `styles.css` | 壳 + 样式（亮/暗主题） |
| `js/data.js` | 数据加载 + 所有 helper + heatmap 聚合引擎 |
| `js/view_*.js` | 三个视图 + 详情页 |
| `js/app.js` `js/dom.js` | 路由 / 主题 / DOM 工具 |
| `artifacts/` | **你的数据** |
| `DATA_FORMAT.md` | 数据格式文档 |
| `make_sample_data.py` | 示例数据生成器 |

聚合逻辑是对原站的忠实复刻，已用原始数据逐格验证（排序、汇总、格式化、配色一致）。

