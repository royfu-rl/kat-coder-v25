# Environment Module

## Contents

1. [Core Pipeline: From Submit to Trained Sample](#1-core-pipeline-from-submit-to-trained-sample)
   - 1.1 [Slime Side: Submit and Write Back](#11-slime-side-submit-and-write-back)
   - 1.2 [`_build_request`: The Only Contract](#12-_build_request-the-only-contract)
   - 1.3 [Environment Side: The `collect_one` Pipeline](#13-environment-side-the-collect_one-pipeline)
   - 1.4 [`_write_back`: One Segment, One Sample](#14-_write_back-one-segment-one-sample)
2. [Base Classes and Data Models](#2-base-classes-and-data-models)
   - 2.1 [The Three Base Classes](#21-the-three-base-classes)
   - 2.2 [The Two-Layer Trajectory Model](#22-the-two-layer-trajectory-model)
   - 2.3 [Config, Result, Task](#23-config-result-task)
3. [Agent Classes: Run a Real CLI, Capture Its Turns](#3-agent-classes-run-a-real-cli-capture-its-turns)
4. [Verifier Classes: Apply Patch, Run Tests, Score](#4-verifier-classes-apply-patch-run-tests-score)
5. [Sandbox: One Interface, Swappable Backend](#5-sandbox-one-interface-swappable-backend)
6. [Gateway Server: Interception Without Modification](#6-gateway-server-interception-without-modification)
   - 6.1 [Proxy Server: Intercept, Forward, Capture](#61-proxy-server-intercept-forward-capture)
   - 6.2 [Harness Adapter: One Gateway, Many Protocols](#62-harness-adapter-one-gateway-many-protocols)
7. [Registered Environments](#7-registered-environments)

---

## 1. Core Pipeline: From Submit to Trained Sample

一条请求的完整生命周期分两侧：slime 提交并写回，environment 采集。

### 1.1 Slime Side: Submit and Write Back

slime 的自定义 rollout 契约是 `generate(args, sample)`，唯一入口，内部拆成三段职责单一的私有方法：

```
generate(args, sample) -> [Sample, ...]
  ├─ _setup_logging(sample)              # 按 rollout step 建 log_dir / logger
  ├─ result = _run_rollout(sample)       # 一次远程采集的完整往返
  │    ├─ req  = _build_request(args, sample)      # metadata + args → (CollectConfig, extra_args)
  │    ├─ resp = _post_collect(req)                # HTTP POST /collect，同步等响应
  │    └─ segs = _parse_result(resp)               # CollectResult → [Segment, ...]
  └─ _write_back(segs, sample)           # 每个 Segment 映射成一个 Sample
```

`generate` 外层套全局超时（`asyncio.wait(timeout=agent_timeout)`）。远程采集卡死时 slime 侧兜底中止，产出 `status=ABORTED` 的样本，不阻塞整个 batch。

旧设计一次 rollout 回传一条轨迹、写回一个 Sample。新设计回传一组 Segment，写回多个 Sample——上下文压缩、子 Agent 分叉都会把一次 rollout 切成多段连续序列，每段独立训练。

### 1.2 `_build_request`: The Only Contract

slime 与 environment 之间只有这一个契约。请求体三段：`collect_config` / `output` / `extra_args`。每个字段的来源分三类：读 `metadata`、读 `args`、固定值。

`ModelConfig`（连接推理服务，塞进 `collect_config.model`）：`base_url` 来自 `args.sglang_router_ip:port`，`tokenizer_path` 来自 `args.hf_checkpoint`，`model`/`api_key`/`support_stream` 是固定值。

`CollectConfig`（一次采集的参数）：`env`/`instance_id`/`sample`/`agent` 来自 `metadata`，选哪个环境、哪条样本、哪个脚手架；`sampling_args` 来自 `args` 的采样参数；`timeout` 是 `args.agent_timeout × 1.2`，保证 environment 侧超时晚于 slime 侧；`return_tokenid=True` 是关键开关，为 True 才回传 token id 与 mask。

`extra_args`（透传 + 训练态标志）：`metadata["extra_args"]` 原样透传环境特定参数；`train_mode=True` 触发落盘与 token/mask/logprob 计算；`proxy_mode`/`sglang_tool_call_parser`/`sglang_chat_template` 仅在非空时放入，否则 environment 侧回退默认。`metadata["data_source"]` 不发给 environment，只在 slime 本地保留用于回写。

### 1.3 Environment Side: The `collect_one` Pipeline

environment 收请求，按 `collect_config.env` 查注册表路由到对应环境类，跑单样本采集流水线。这是系统的心脏：

```
HTTP POST /collect
  └─ collect_endpoint(request)
       └─ run_collection                          # 多进程 spawn + 超时隔离
            └─ env = Env()  →  collect_async
                 └─ collect_one(state)             # 单样本采集流水线
                      │
                      ├─ turns   = rollout()                     # Agent 在沙箱跑，Gateway 拦每次模型调用
                      │                                          # → sglang 推理 → [Turn, ...]
                      ├─ reward  = score()                       # Verifier 应用 patch 跑测试 → outcome reward
                      │
                      ├─ trajectory_manager.process_turn(turns)  # 算 step 级指标；turn-level mask / filter /
                      │                                          # reward shaping / advantage shaping
                      │                                          # → 回填每个 Step 的张量字段
                      │
                      ├─ episode, segments = segment_builder(turns)   # Turn 序列 → 连续可训练 Segment
                      ├─ trajectory_manager.process_session(episode)  # 算 session 级指标
                      │
                      └─ return CollectResult(segments)          # 只回传 Segment 维度
```

每条样本有独立 `output_dir`，落原始请求日志、轨迹文件、patch、各阶段耗时。`train_mode` 下异常直接 raise 让 slime 重试，且只有此模式才算 token/mask/logprob。多进程 spawn 隔离，子进程超时或崩溃不拖垮 server。

一个设计决策：mask、filter、reward shaping、advantage reshaping 全在 environment 侧做，不把 step 级和 session 级指标透传回 slime 再算。这些是 CPU 密集型计算，放近数据处更快，slime 只消费成品 Segment。

environment 自身不直接调推理 API。模型推理由沙箱内 Agent 经 Gateway 间接发起，链路是 Agent 容器 → Gateway → sglang。

### 1.4 `_write_back`: One Segment, One Sample

每个 Segment 已是塑形完成的训练序列。`_write_back` 按 chat template 的 assistant 边界切出 prompt / response 两段，连同 mask、reward、status 写入一个 Sample。N 个 Segment 产出 N 个 Sample。

---

## 2. Base Classes and Data Models

系统的骨架是三个抽象基类加一套数据模型。三个基类各配一个全局注册表，用同一套模式：装饰器注册，按名实例化。加一个新环境、新脚手架、新验证器，都是新增一个子类加一行装饰器。

### 2.1 The Three Base Classes

```python
class BaseEnv(ABC):
    def info() -> EnvInfo                              # 环境元信息
    def peek(number) -> [dict]                         # 预览数据集样本
    def export(type, number) -> Dataset                # 导出训练数据
    async def collect_async(config, extra_args) -> CollectResult   # 采集入口

class BaseAgent(ABC):
    async def rollout(task, model_config, extra_args) -> AgentResult   # 跑一条轨迹

class BaseVerifier(ABC):
    async def verify(task, extra_args) -> VerifierResult               # 给一条预测打分
```

`BaseEnv` 的子类只填四个钩子，把 `collect_one` 流水线的四步补全：`get_dataset()` 加载数据集、`build_task(sample)` 把样本转成 Task、`rollout(task)` 选 Agent 跑、`score(task)` 选 Verifier 打分。

### 2.2 The Two-Layer Trajectory Model

数据模型分两层。raw 层是 Proxy 如实记录的，训练层是 environment 算出来的。这个分层是 token 采集能成立的前提：Proxy 只记录，塑形逻辑全在训练层。

Raw 层：

- **Turn** `{prompt, response}`：Proxy 拦到的一次原始 OpenAI 格式请求/响应，轨迹的原子单位。
- **Session** `[Turn, ...]`：一次 rollout 的原始轨迹。

训练层：

- **Step(Turn)**：Turn 上补训练张量——`prompt_token` / `response_token` / `attention_mask` / `step_metrics` / `reward_shaping_tensor` / `advantage_shaping_tensor`。
- **Segment**：一条连续可训练序列——`prompt_token` / `response_token` / `attention_mask` / `segment_metrics`。
- **Episode(Session)**：一次 rollout 的训练态产物——`turns` / `segments` / `session_metrics`。

Segment 不是 Turn。Turn 是通信单位，一问一答；Segment 是训练单位，一条连续 token 序列。一个 Session 切成多个 Segment，一个 Segment 对应 slime 的一个 Sample。

### 2.3 Config, Result, Task

- **CollectConfig**：一次采集的全部输入。选环境、选样本、选脚手架、采样参数、`return_tokenid`。
- **ModelConfig**：连接推理服务的信息。`base_url` / `tokenizer_path` / `model`。
- **CollectResult**：一次采集的产物。`segments` / outcome `reward` / `status` / `fix_status`（真实 rollout 结局：completed / failed / timeout / error）。
- **Task**：样本经 `build_task` 转换后的执行单元。`prompt` / `working_dir` / `timeout` / `return_tokenid` / 镜像信息。SWE 任务额外带 `instance_id` / `base_commit` / `image_name`。

`status` 与 `fix_status` 分开：`status` 恒为 completed 保持接口稳定，真实结局走 `fix_status`，供 slime 判断样本是完成、截断还是中止。

---

## 3. Agent Classes: Run a Real CLI, Capture Its Turns

Agent 层跑真实 CLI 脚手架产出 Turn。它不感知 turn 结构——脚手架自己实现 agent loop，Gateway 拦截还原出 Turn。

```
BaseAgent
  └─ CommonAgent          # 带 idle 监控的 rollout 循环
       └─ DockerAgent     # 沙箱脚手架基类，绝大多数脚手架继承它
```

一次 rollout 的核心调用链，两层职责分明：`_supervise` 守边界，`_run_scaffold` 干活。

```
CommonAgent.rollout(task, model_config, extra_args)
  │
  ├─ with borrow_gateway(model_config, return_token_id=task.return_tokenid) as gw:
  │    gw.set_log_dir / set_sampling_args / set_timer
  │    model_config.base_url = gw.get_url()          # 脚手架 endpoint 改写成 Gateway 地址
  │
  │    └─ _supervise(task, model_config, extra_args, gw)      # 监控层: 守 idle / max_turns / timeout
  │         │  每隔固定间隔检查, 任一命中 → gw.pause() + 取消任务
  │         │
  │         └─ _run_scaffold(task, model_config, extra_args)  # 执行层 (DockerAgent 实现)
  │              │
  │              ├─ image = resolve_image(task)               # 命中缓存 or 远程构建服务现装脚手架
  │              ├─ env   = self._container_env(gw_url, model)  # 钩子①: 注入模型环境变量
  │              ├─ box   = Sandbox(image, env).start()        # 起容器 (下一节)
  │              │
  │              ├─ box.exec_run(prepare_commands)            # 准备环境
  │              ├─ cmd = "cd <workdir> && git reset --hard && " + self._task_command(task)  # 钩子②
  │              ├─ result = box.exec_run(cmd, timeout=task.timeout)
  │              │            # 容器内脚手架 agent loop 自循环调 LLM → 命中 Gateway
  │              │
  │              ├─ if result.exit_code: return {success: False, error_type: EXECUTION_ERROR}
  │              ├─ patch = box.exec_run("git add -A && git diff --cached")  # 抽 patch
  │              └─ finally: box.remove(force=True)            # 幂等清理
  │
  └─ if ret.success:  return AgentResult(status="completed", turns=gw.get_turns(), data=ret)
     else:            按 error_type 抛 Runtime/Timeout 或 返回 status="failed"
```

`CommonAgent` 管监控与收尾：借一个 Gateway，把脚手架的模型 endpoint 改写成 Gateway 地址，`_supervise` 套一层守护——每隔固定间隔查三个终止条件（idle 过久、超 max_turns、超 timeout），任一命中就 pause Gateway 并取消任务。跑完取走全部 Turn 组装 `AgentResult`。`ErrorType` 区分 PREPARE_FAILED / SANDBOX_ERROR / EXECUTION_ERROR / IDLE_TIMEOUT / TIMEOUT / TURN_LIMIT，决定是抛异常让 slime 重试，还是返回 `status="failed"`。

`_run_scaffold` 是执行层，管沙箱生命周期一条直线：拿镜像（命中缓存或调远程构建服务现装脚手架）→ 起容器（注入指向 Gateway 的模型环境变量）→ 准备环境 → 跑任务命令（容器内脚手架自循环调 LLM，全程命中 Gateway）→ 容器内 `git diff` 抽 patch → 幂等清理容器。patch 不靠脚手架落盘，任务结束后直接在容器里 `git add -A && git diff --cached` 取。

加一个新脚手架，只需子类实现三个钩子：`_container_env` 注入模型环境变量（各脚手架变量名不同）、`_format_model` 模型名格式、`_task_command` 调真实 CLI 的那一行命令——这是唯一的核心差异点。统一模式是「真实 CLI + 沙箱执行 + 指向 Gateway 的 model endpoint」。

已注册的脚手架一览。八个继承 DockerAgent，一个例外走宿主机 subprocess：

| 脚手架 | 类型 | 任务命令核心 | 关键差异 |
|---|---|---|---|
| claude_code | DockerAgent | `claude -p <prompt> --permission-mode bypassPermissions --allowedTools ... --max-turns 150` | 现场写 `~/.claude/settings.json` 指向 Gateway；走 Anthropic 协议 |
| mini_agent | DockerAgent | `mini -t <prompt> -y --exit-immediately ...` | mini-swe-agent；走 litellm，环境变量 inline 进命令 |
| opencode | DockerAgent | `opencode -m openai/<model> run <prompt>` | 现场写 provider 配置文件 |
| goose | DockerAgent | `goose run -t <prompt>` | |
| kilocode | DockerAgent | `kilocode --auto <prompt>` | 无自然退出，靠 idle 超时打断 |
| cline | DockerAgent | `cline -F plain -o <prompt>` | `is_task_completed` 检测完成标记 |
| roocode | DockerAgent | 启动脚本 | 异步任务，轮询完成状态 |
| grok_cli | DockerAgent | 启动脚本 | |
| sweagent | CommonAgent（非 Docker） | 宿主机 `sweagent run-batch --config ... --agent.model.api_base <gateway>` | 用远程 sandbox；解析预测文件取 patch |

sweagent 是唯一例外。它不在本地容器跑，而是宿主机 subprocess 调 CLI，容器由远程 sandbox 起。除它之外，加一个脚手架就是新建一个 DockerAgent 子类实现那三个钩子——命令表里每一行差异，本质都收敛在 `_task_command` 那一行。

---

## 4. Verifier Classes: Apply Patch, Run Tests, Score

所有 Verifier 都做同一件事：起沙箱，应用 patch，跑评测脚本，解析报告，返回 `VerifierResult(score, status)`。分数通常是 0/1：pass_to_pass 与 fail_to_pass 全过则 1。

```
BaseVerifier
  └─ SandboxVerifier      # 起沙箱 → apply patch → 跑 eval 脚本 → 解析报告
```

差异只在最后跑的是谁的脚本，分两类：

- **调官方 harness**：patch 交给数据集自带的官方评测脚本，读它产出的报告算准确率。省事，但绑定官方 harness 的行为。
- **手搓评测**：自己上传 patch、测试补丁、评测脚本到沙箱，依次跑，再解析测试日志。灵活，适配官方 harness 覆盖不到的训练集。

两类的骨架完全一致，只是「跑评测脚本」这一步换了实现。已注册的 Verifier：

| Verifier | 打分对象 | 评测机制 |
|---|---|---|
| swe_bench | SWE-bench Verified / Multilingual / rebench | 调官方 harness，读 resolved / completed 算准确率 |
| swe_rebench_v2 | SWE-rebench-V2 | 调官方脚本（远程沙箱），读报告 |
| multi_swe_bench | Multi-SWE-bench（多语言） | 调多语言 harness，读最终报告 |
| swe_compass | swe_compass 训练集 | 手搓：起容器跑 test_cmd，判 pass_to_pass ∧ fail_to_pass |
| auto_builder | autobuilder 训练/bench | 手搓：上传 patch/test/脚本，依次跑，解析测试日志 |

超时或异常时统一兜底：读沙箱里已落盘的报告文件抢救结果，不让一次超时直接判 0。

---

## 5. Sandbox: One Interface, Swappable Backend

Sandbox 是执行隔离的统一抽象，Agent 与 Verifier 共用。接口只有五个动作，`ExecResult(exit_code, output)` 是 docker 风格返回：

```python
class Sandbox(ABC):
    def start(max_retries=2) -> Result       # 起容器 → 轮询 ready → 建 session
    def exec_run(cmd, timeout) -> ExecResult  # 在 session 里执行，带重试分类
    def upload_file(src, dst) -> bool         # 推文件进容器
    def remove(force=True)                    # 幂等销毁，轮询直到停止
    def is_alive() -> bool                    # 探活
```

接口简单，难点全在远程后端的韧性上。本地 docker daemon 实现只是 SDK 直包，几乎无状态；远程容器 HTTP 服务实现要扛住镜像拉取慢、网关抖动、session 回收三类真实故障。这一层是整个系统唯一需要按部署环境替换的适配点，把这些坑一并封在接口后面，上层 Agent 与 Verifier 完全无感。

`start` 不是一次请求就完事。远程节点镜像缓存饱和时，拉镜像极限能到十几分钟。所以 `start` 用**渐进式探活**：POST 建容器拿到 session，然后先密后疏地轮询 `is_alive`——前 30 秒密探（缓存命中时几秒就 ready），中段 5 秒，后段 10 秒，总预算约 20 分钟。ready 后再建 bash session、逐条 export 环境变量、跑启动脚本。整轮失败就销毁重建，最多重试若干次，把超出预算的部分让给上层重试整条任务，而不是无限等下去。

`exec_run` 的核心是**按 HTTP 状态分类决定重试还是放弃**——这是远程沙箱能不能稳的关键，也是本地实现不需要的部分：

| 状态 | 处理 |
|---|---|
| 200 | 正常，返回 `ExecResult` |
| 429 限流 | 带 jitter 重试 |
| 500 / 502 / 503 | 服务端 transient 故障，指数退避重试（上限 30s + jitter） |
| 504 网关超时 | 不重试。命令很可能还在容器里跑，重试只会在同 session 排队叠加耗时 |
| 511 | runtime 退出码非 0，返回 `exit_code=1` 不重试 |
| 401 / 403 | 认证/权限错，不可重试直接返回 |

最后是**防泄漏**。远程容器有默认 idle-TTL，不显式设长会被后台回收，之后 `exec_run` 拿到的错误会被误判成任务失败。创建时显式传一个长于最长 rollout 的 TTL 盖过默认值，同时保留有限值做兜底：正常收尾靠 `remove(force=True)`，异常退出靠 TTL 到期自动回收，两条路都堵上才不漏容器。

---

## 6. Gateway Server: Interception Without Modification

不改任何脚手架源码，把脚手架的模型 endpoint 指向本地 Gateway。脚手架以为在直连模型，实际每次调用都过 Gateway：转发给真实模型，同时把 (request, response) 存成 Turn。跑完取走全部 Turn 就是完整轨迹。

Gateway 分两个组件。

### 6.1 Proxy Server: Intercept, Forward, Capture

收脚手架的模型请求，改写后转发到 sglang，把响应存成 Turn 再原样返回。追踪 active request 数与 idle 时间，供上层打断卡死的 Agent。三种转发模式：

- **chat**：流式透传，不碰 token。
- **chat_simple**（CICO）：非流式直传。
- **generate**（TITO，token-in-token-out）：维护 append-only 的 token 历史，直接调 sglang 原生 `/generate` 传 `input_ids`，拿回 `output_ids` 与 logprobs。它消除多轮 re-tokenization，是 RL 训练拿到干净 token 序列的关键路径。

### 6.2 Harness Adapter: One Gateway, Many Protocols

不同脚手架说不同协议。Claude Code 走 Anthropic 的 `/v1/messages`，其余多走 OpenAI 的 `/v1/chat/completions`。Harness Adapter 把这些协议归一到同一条 Gateway：请求转成 OpenAI 格式转发，响应转回脚手架原生格式，含流式事件重建、非流式模型的 mock streaming。这让每种脚手架的原生协议都走同一条 Proxy，不为任何一家改协议。

---

## 7. Registered Environments

全部继承核心采集流水线，当前工作树是 SWE 家族：

| 环境 | 样本类型 / 子集 | Verifier |
|---|---|---|
| swebench_verified | SWE-bench Verified (300) | swe_bench |
| swebench_multilingual | SWE-bench Multilingual (300) | swe_bench |
| swe_rebench_v2 | SWE-rebench-V2 main / train | swe_rebench_v2 |
| swe_rebench_v2_train | SWE-rebench-V2 train | swe_rebench_v2 |
| multi_swe_bench | Multi-SWE-bench，多语言 lite / test / train / full | multi_swe_bench |
| swe_compass | swe_compass 训练集 | swe_compass |
| code_comp | 代码理解 | 无（不打分） |
| auto_builder | autobuilder 训练 / bench（easy / medium / hard 等 9 个变体） | auto_builder |

加一个新环境的成本恒定：新增一个流水线子类，声明数据集、Task 转换、选哪个 Agent、选哪个 Verifier。math、搜索、工具、游戏类环境按同一套流水线补齐即可，不改任何上层。
]
