// Illustrative task cards for the benchmarks discussed in ../code-bench.md.
// Each demo reconstructs one benchmark's task *shape* — instruction, verifier,
// judgement — in the same layout deepswe/ uses for real task detail pages.
// Content is representative, not a verbatim copy of any published instance.
const DEMOS = {};

DEMOS['swe-bench'] = {
  bench: 'SWE-bench', lang: 'python',
  title: 'QuerySet.only() 与 select_related() 组合时返回错误字段',
  repo: 'django/django', commit: '0c3b5bc0', id: 'django__django-14855',
  tabs: [
    { label: 'Issue', hint: '输入', kind: 'html', body:
      '<p>仓库里一条真实 issue，是模型唯一的任务描述——没有文件清单，没有函数名，' +
      '要自己在几十万行代码里定位。</p>' +
      '<p><strong>Description.</strong> 在一个反向关联上先调 <code>select_related()</code> ' +
      '再调 <code>only()</code> 限定字段时，返回的对象里被限定掉的字段仍然触发了额外查询，' +
      '且部分字段取到了错误的值。预期是 <code>only()</code> 的字段集合能正确下推到 ' +
      'JOIN 之后的列选择。</p>' +
      '<ul><li>Python 3.9 / Django main 分支可复现</li>' +
      '<li>去掉 <code>select_related()</code> 后行为正常</li></ul>' },
    { label: 'Verifier', hint: 'F2P 1 · P2P 412', kind: 'diff', file: 'tests/test.patch', body: `
--- a/tests/defer_regress/tests.py
+++ b/tests/defer_regress/tests.py
@@ class DeferRegressionTest(TestCase):
+    def test_only_with_select_related(self):
+        obj = Item.objects.select_related("child").only("name", "child__value").get()
+        with self.assertNumQueries(0):
+            self.assertEqual(obj.child.value, 42)
+        self.assertEqual(obj.name, "root")
` },
    { label: 'Judgement', hint: '判据', kind: 'html', body:
      '<dl class="specs">' +
      '<dt>评测方式</dt><dd>exec-based — 容器内真实运行测试</dd>' +
      '<dt>FAIL_TO_PASS</dt><dd>打上 test patch 后必须失败，补丁后必须通过</dd>' +
      '<dt>PASS_TO_PASS</dt><dd>412 条既有测试不得回归</dd>' +
      '<dt>提交物</dt><dd>一个 <code>git diff</code>，不含测试文件改动</dd>' +
      '</dl>' +
      '<p>这套双判据是 SWE-bench 立下的范式，后来被几乎所有 SWE 系 benchmark 继承：' +
      '只让目标测试转绿不算过，还得证明没弄坏别的东西。</p>' },
  ],
  verdict: [ { cls:'fail', text:'pre-patch FAIL' }, { cls:'pass', text:'post-patch PASS' },
             { cls:'plain', text:'2294 tasks' } ],
  note: '示意重构 · 形态取自 issue→patch 主线',
};

DEMOS['swe-bench-verified'] = {
  bench: 'SWE-bench Verified', lang: 'python',
  title: '人工验证过的同一形态：判据无歧义的 500 题切片',
  repo: 'sympy/sympy', commit: 'e4a58bd6', id: 'sympy__sympy-24152',
  tabs: [
    { label: 'Annotation', hint: '人工验证', kind: 'html', body:
      '<p>题目形态与 SWE-bench 完全一致，区别全在这一层——每道题经人工确认过' +
      '「issue 描述足以定位问题、测试足以判定正确」。</p>' +
      '<dl class="specs">' +
      '<dt>描述充分性</dt><dd>✓ issue 明确给出复现代码与预期输出</dd>' +
      '<dt>测试有效性</dt><dd>✓ 测试恰好覆盖 issue 所述行为，无超纲断言</dd>' +
      '<dt>方案唯一性</dt><dd>✓ 不存在「改对了但测试仍挂」的合法实现</dd>' +
      '<dt>环境可复现</dt><dd>✓ 依赖锁定，无网络依赖</dd>' +
      '</dl>' +
      '<p>原集 2294 题里筛出 500 道，OpenAI 联合原作者标注。剔除的正是那些' +
      '模型改对了却因判据噪声被判失败的题。</p>' },
    { label: 'Issue', hint: '输入', kind: 'html', body:
      '<p><strong>Description.</strong> <code>TensorProduct</code> 在展开含标量因子的' +
      '表达式时提前返回，导致嵌套的张量积没有被完全展开。</p>' +
      '<pre>>>> from sympy import *\n' +
      '>>> U, V = symbols("U V", commutative=False)\n' +
      '>>> P = TensorProduct(2*U - V, U + V)\n' +
      '>>> print(P.expand(tensorproduct=True))\n' +
      '2*U x U + 2*U x V - V x U - V x V   # 预期\n' +
      '2*U x U + 2*U x V - V x (U + V)     # 实际</pre>' },
    { label: 'Why saturated', hint: '现状', kind: 'html', body:
      '<p>正因判据干净，Verified 的分数最能反映真实的缺陷定位能力——也正因如此它饱和了。</p>' +
      '<dl class="specs">' +
      '<dt>前沿模型区间</dt><dd>顶级模型集中在 80 分上下，彼此差距落在噪声里</dd>' +
      '<dt>剩余未解题</dt><dd>多是真难，或仍带歧义</dd>' +
      '<dt>训练增量</dt><dd>有限：可学的模式已被充分覆盖</dd>' +
      '<dt>当前定位</dt><dd>能力下限的确认，而非前沿模型的区分</dd>' +
      '</dl>' +
      '<p>它留下的方法论比分数更有用：评测集的信噪比比规模更重要，' +
      '因为噪声会同时污染评测与训练奖励。</p>' },
  ],
  verdict: [ { cls:'pass', text:'500 verified' }, { cls:'warn', text:'~80% saturated' } ],
  note: '示意重构 · 人工去噪后的干净信号',
};

DEMOS['multi-swe-bench'] = {
  bench: 'Multi-SWE-bench', lang: 'rust',
  title: 'Trait 实现漏了边界条件：容量为 0 时 reserve 溢出',
  repo: 'tokio-rs/bytes', commit: '7f1a2c9d', id: 'multi__rust-bytes-512',
  tabs: [
    { label: 'Issue', hint: '输入', kind: 'html', body:
      '<p>同一种「定位加修复」能力，换到 Rust 上就要求模型掌握完全不同的语言机制——' +
      '所有权、生命周期、trait 约束。</p>' +
      '<p><strong>Description.</strong> 为零容量的 <code>BytesMut</code> 调用 ' +
      '<code>reserve(additional)</code> 时，内部的容量计算在 <code>additional</code> ' +
      '接近 <code>usize::MAX</code> 时发生溢出，debug 构建 panic、release 构建静默分配错误大小。</p>' +
      '<ul><li>需要在 <code>BufMut</code> 的实现里补上 checked 运算</li>' +
      '<li>不得改变既有的 amortized growth 行为</li></ul>' },
    { label: 'Verifier', hint: 'cargo test', kind: 'diff', file: 'tests/test_bytes_mut.rs', body: `
--- a/tests/test_bytes_mut.rs
+++ b/tests/test_bytes_mut.rs
@@ mod reserve {
+    #[test]
+    #[should_panic(expected = "capacity overflow")]
+    fn reserve_overflow_from_empty() {
+        let mut b = BytesMut::new();
+        b.reserve(usize::MAX);
+    }
+
+    #[test]
+    fn reserve_keeps_amortized_growth() {
+        let mut b = BytesMut::with_capacity(8);
+        b.reserve(9);
+        assert!(b.capacity() >= 17);
+    }
` },
    { label: 'Environment', hint: '判分基础设施', kind: 'html', body:
      '<p>多语言环境的真正难点不在题，在判分基础设施——每种语言的构建工具链、' +
      '测试框架、日志格式都不同。做法是把语言差异吸收在环境层，而非判分逻辑里。</p>' +
      '<dl class="specs">' +
      '<dt>Base image</dt><dd class="mono">rust:1.74-slim</dd>' +
      '<dt>装依赖</dt><dd class="mono">cargo fetch --locked</dd>' +
      '<dt>跑测试</dt><dd class="mono">cargo test --no-fail-fast -- --format json</dd>' +
      '<dt>解析结果</dt><dd>语言专属 log_parser → 统一的结构化通过 / 失败信号</dd>' +
      '</dl>' +
      '<p>每种语言固化一条「装依赖 → 编译 → 跑测试 → 解析结果」的链，' +
      '判分层就只看统一后的信号。想扩语言，先补这条链。</p>' },
  ],
  verdict: [ { cls:'pass', text:'exec-based' }, { cls:'plain', text:'7 languages' },
             { cls:'plain', text:'1632 tasks' } ],
  note: '示意重构 · 语言作为受控变量',
};

DEMOS['swe-bench-live'] = {
  bench: 'SWE-bench Live', lang: 'python',
  title: '本月新抓仓库里一条上周提交的 issue',
  repo: 'newly-crawled/pkg', commit: 'a91f4d20', id: 'live-2026-05__pkg-118',
  tabs: [
    { label: 'Freshness', hint: '抗污染', kind: 'html', body:
      '<p>任何静态 benchmark 一旦发布，题目就可能进入下一代模型的训练语料。' +
      '对策是让题目比模型更新——用「比模型新」保证「模型没见过」。</p>' +
      '<dl class="specs">' +
      '<dt>issue 提交</dt><dd>2026-05-11（评测集切片前 5 天）</dd>' +
      '<dt>仓库首次纳入</dt><dd>2026-05 月度批次</dd>' +
      '<dt>模型知识截止</dt><dd>早于 issue 提交日 → 不可能背过答案</dd>' +
      '<dt>更新节奏</dt><dd>每月滚动抓取新仓库与新 issue</dd>' +
      '</dl>' +
      '<p>测的仍是 issue→patch，但测的是真实泛化而非记忆。</p>' },
    { label: 'Issue', hint: '输入', kind: 'html', body:
      '<p><strong>Description.</strong> 配置加载器在 Windows 上把 ' +
      '<code>~/.config/pkg/config.toml</code> 解析成了字面量目录 <code>~</code>，' +
      '导致用户配置被静默忽略、全部回落到默认值。Linux / macOS 正常。</p>' +
      '<ul><li>应使用平台无关的 home 目录解析</li>' +
      '<li>路径已存在但不可读时要报错，而不是静默回落</li></ul>' },
    { label: 'Cross-platform', hint: 'Windows 切片', kind: 'shell', body: `
# 同一道题，两个平台各跑一次判分
$ pytest tests/test_config.py -q          # ubuntu-22.04
2 passed

$ pytest tests/test_config.py -q          # windows-2022
1 failed, 1 passed
  test_expands_home_dir - AssertionError: '~\\\\.config\\\\pkg' != 'C:\\\\Users\\\\u\\\\.config\\\\pkg'

# 很多环境依赖在 Windows 上的行为与 Linux 不同，
# 只在 Linux 判分会漏掉整整一类缺陷。
` },
  ],
  verdict: [ { cls:'pass', text:'monthly rolling' }, { cls:'plain', text:'1319 tasks' },
             { cls:'plain', text:'+ Windows slice' } ],
  note: '示意重构 · 用时间新鲜度抗数据泄漏',
};

DEMOS['swe-bench-pro'] = {
  bench: 'SWE-Bench Pro', lang: 'go',
  title: '跨三个模块的并发缺陷：连接池、任务队列、错误处理',
  repo: '<private enterprise repo>', commit: 'redacted', id: 'pro__svc-0473',
  tabs: [
    { label: 'Issue', hint: '输入', kind: 'html', body:
      '<p>企业级仓库上的 issue→patch。模型面对的是一个从未公开的大型代码库，' +
      '要做长链条定位与多文件协同修改，而不是在见过的开源仓里补一个已知小 patch。</p>' +
      '<p><strong>Description.</strong> 高并发下 worker 偶发卡死。观察到连接池在' +
      '任务被取消时未归还连接，队列侧的重试逻辑又把同一任务重新入队，' +
      '错误处理层把 context 取消误判为可重试错误，三者叠加后连接被逐步耗尽。</p>' +
      '<ul><li>取消路径必须归还连接</li>' +
      '<li>context 取消不得计入可重试错误</li>' +
      '<li>重复入队需要幂等保护</li></ul>' +
      '<p>任何一处漏改，集成测试都会挂。</p>' },
    { label: 'Patch scope', hint: '~160 LOC · 3 files', kind: 'diff', file: 'solution.patch（结构示意）', body: `
--- a/internal/pool/conn.go          |  +38  -9
--- a/internal/queue/dispatch.go     |  +52  -14
--- a/internal/errs/classify.go      |  +21  -6
@@ internal/pool/conn.go
+func (p *Pool) release(c *Conn, err error) {
+    if errors.Is(err, context.Canceled) {
+        p.put(c)          // 取消不是坏连接，必须归还
+        return
+    }
@@ internal/errs/classify.go
-    return true  // 一切非 nil error 都重试
+    if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
+        return false
+    }
+    return isTransient(err)
` },
    { label: 'Why it separates', hint: '区分度', kind: 'html', body:
      '<p>难度靠仓库选择而非题目堆砌：选企业级、强耦合、多模块的仓库，' +
      '题目自然长程；选私有 / 强 copyleft 仓库，自然抗泄漏。</p>' +
      '<dl class="specs">' +
      '<dt>平均改动</dt><dd>约 160 行，跨多个文件</dd>' +
      '<dt>抗泄漏</dt><dd>私有 / copyleft 仓库，未进过训练语料</dd>' +
      '<dt>Claude Mythos 5</dt><dd>80.3%</dd>' +
      '<dt>Fable 5</dt><dd>80%</dd>' +
      '<dt>GPT-5.6 Sol</dt><dd>64.6%</dd>' +
      '</dl>' +
      '<p>约 16 个百分点的差距，在已饱和的 Verified 上完全看不到。' +
      '造高区分度环境时，仓库的选择比题目的数量更决定天花板。</p>' },
  ],
  verdict: [ { cls:'pass', text:'exec-based' }, { cls:'warn', text:'not saturated' },
             { cls:'plain', text:'本文对照系' } ],
  note: '示意重构 · 企业级长程 issue→patch',
};

DEMOS['deepswe'] = {
  bench: 'DeepSWE v1.1', lang: 'typescript',
  title: 'Add iterable collection combinators to true-myth',
  repo: 'true-myth-ts/true-myth', commit: 'live HEAD', id: 'add-iterable-collection-combinators',
  tabs: [
    { label: 'Trials', hint: '隔离判分', kind: 'html', body:
      '<p>从活跃维护的开源库里取一条真实 issue，模型在<strong>隔离容器</strong>里定位、修复、跑测试。' +
      'v1.1 相比 v1 用同一批题目但更干净的隔离式判分。</p>' +
      '<dl class="specs">' +
      '<dt>题目 / 仓库</dt><dd>113 题 · 91 个仓库</dd>' +
      '<dt>判分方式</dt><dd>程序化验证，每题一个独立可重置沙盒</dd>' +
      '<dt>v1 → v1.1</dt><dd>题目不变，改的全是判分隔离性</dd>' +
      '<dt>GPT-5.6 Sol</dt><dd>72.7%</dd>' +
      '<dt>Claude Fable 5</dt><dd>69.7%</dd>' +
      '</dl>' +
      '<p>与 Pro 形成互补：同样测长程能力，但仓库来源不同——' +
      '活跃开源仓 vs 封闭企业仓，模型的相对排名随之翻转。</p>' },
    { label: 'Instruction', hint: '输入', kind: 'html', body:
      '<p><strong>Task.</strong> 为 <code>Maybe</code> 与 <code>Result</code> 补上作用于可迭代集合的' +
      '组合子，使一组容器值能被折叠成单个容器值。</p>' +
      '<ul>' +
      '<li><code>all(iterable)</code>：全部 <code>Just</code> / <code>Ok</code> 时返回容器包裹的数组，' +
      '否则短路返回第一个 <code>Nothing</code> / <code>Err</code></li>' +
      '<li><code>any(iterable)</code>：返回第一个成功值，全失败时聚合错误</li>' +
      '<li>类型签名必须保持完整推断，不得退化为 <code>any</code></li>' +
      '<li>惰性可迭代对象只允许消费一次</li>' +
      '</ul>' },
    { label: 'Isolation', hint: '为何重要', kind: 'shell', body: `
# v1：多题共享判分环境 → 别的题污染了这道题
$ npm test              # 上一道题遗留的 node_modules 缓存仍在
  ✓ all() short-circuits          # 假阳：其实没实现，命中了缓存产物
  47 passing

# v1.1：每题独立沙盒，判分前重置
$ docker run --rm -v $PWD:/w task-add-iterable npm test
  ✗ all() short-circuits          # 真实结果
  46 passing, 1 failing

# 判分环境若与其它题共享状态，就会出现假阳。
# 判分的隔离性和判据本身一样重要。
` },
  ],
  verdict: [ { cls:'pass', text:'isolated judge' }, { cls:'plain', text:'113 tasks' },
             { cls:'plain', text:'live OSS repos' } ],
  note: '取自 DeepSWE 公开任务目录 · 判分说明为示意',
};

DEMOS['terminal-bench'] = {
  bench: 'Terminal-Bench 2.1', lang: 'shell',
  title: '只给一个终端和一个目标，边界要自己划',
  repo: 'laude-institute/terminal-bench', commit: 'harbor', id: 'tb2.1__mixed-domain',
  tabs: [
    { label: 'Task', hint: '只给目标', kind: 'html', body:
      '<p>把模型放进真实终端，测它在命令行环境下端到端解决任务的成功率，' +
      '覆盖软件工程、机器学习、安全、数据科学。</p>' +
      '<p><strong>典型题目：</strong></p>' +
      '<ul>' +
      '<li>从源码编译一个能在 QEMU 里成功启动的 Linux 内核</li>' +
      '<li>配置一台可正常 push / pull 的 git 服务器</li>' +
      '<li>训练一个 fastText 模型，精度达到给定阈值</li>' +
      '<li>在给定压缩包里定位目标文件、把结果写进 <code>/app/solution.txt</code></li>' +
      '</ul>' +
      '<p><strong>没有分步指令。</strong>这是与 Pro / DeepSWE 最大的区别——' +
      '那两个给定 issue 描述、边界清晰；这里模型要自己探索文件系统、装依赖、' +
      '跑命令、验证结果。</p>' },
    { label: 'Trajectory', hint: 'agent 自主探索', kind: 'shell', body: `
# 目标：编译一个能在 QEMU 里启动的内核。没人告诉你要装什么。
$ ls /app
linux-6.6.tar.xz

$ make -C linux-6.6 defconfig
# make: command not found —— 工具链得自己判断、自己装
$ apt-get install -y build-essential flex bison libelf-dev qemu-system-x86

$ make -C linux-6.6 defconfig && make -C linux-6.6 -j$(nproc)
  ...
  Kernel: arch/x86/boot/bzImage is ready

# 自己验证，而不是等判分告诉你
$ qemu-system-x86_64 -kernel linux-6.6/arch/x86/boot/bzImage -append console=ttyS0 -nographic
  [    0.000000] Linux version 6.6.0 ...
  [    1.284431] Run /init as init process
` },
    { label: 'Judgement', hint: '只验结果', kind: 'html', body:
      '<p>判据是<strong>结果状态</strong>而非<strong>过程正确</strong>。' +
      '内核能在 QEMU 里启动，就算过，不管你怎么编译的。</p>' +
      '<dl class="specs">' +
      '<dt>判分对象</dt><dd>环境的终局状态（能不能启动 / 精度达没达标 / 文件内容对不对）</dd>' +
      '<dt>不判</dt><dd>用了什么命令、装了什么包、走了几步</dd>' +
      '<dt>题量</dt><dd>初代 80 → 2.0 扩到 89 → 2.1 为当前活跃版</dd>' +
      '<dt>维护</dt><dd>Stanford × Laude，构建在 Harbor 框架上</dd>' +
      '<dt>GPT-5.6 Sol Ultra</dt><dd>91.9%（对 Claude Mythos 5 的 88%）</dd>' +
      '</dl>' +
      '<p>这种「给目标、验结果、不管过程」的设计天然容纳多样解法，' +
      '也最贴近真实的 agent 训练——奖励来自环境的真实反馈，' +
      '而非对某条标准路径的匹配。</p>' },
  ],
  verdict: [ { cls:'pass', text:'outcome-based' }, { cls:'plain', text:'89 tasks' },
             { cls:'plain', text:'language-agnostic' } ],
  note: '示意重构 · 终端代理的端到端能力',
};

DEMOS['programbench'] = {
  bench: 'ProgramBench', lang: 'c/c++',
  title: '给一个编译好的 grep，把整份能通过测试的源码写出来',
  repo: 'facebookresearch/ProgramBench', commit: '—', id: 'pb__grep-c',
  tabs: [
    { label: 'Task', hint: '输入 = 二进制', kind: 'html', body:
      '<p>模型拿到的是<strong>黑盒二进制的可观测行为</strong>加一份文档，没有源码。' +
      '任务是反推出一整套 C 源码树，让它在功能上与原程序等价、通过全部测试。</p>' +
      '<dl class="specs">' +
      '<dt>给什么</dt><dd><code>/bin/grep</code> 可执行文件 + 行为文档（man page 级）</dd>' +
      '<dt>要什么</dt><dd>完整可编译的源码树 + Makefile</dd>' +
      '<dt>判什么</dt><dd>约 248k 行测试全跑，纯 exec-based</dd>' +
      '<dt>覆盖语言</dt><dd>Go / Rust / C / C++ / Java 等系统级语言，200 题</dd>' +
      '</dl>' +
      '<p>它测的不是补丁，是<strong>从规格到完整实现的重建</strong>——' +
      '覆盖范围比「改几个文件」大得多。</p>' },
    { label: 'Probing', hint: 'agent 反推行为', kind: 'shell', body: `
# 没有源码，只能拿二进制当规格来问
$ ./grep --help | head -5
Usage: grep [OPTION]... PATTERNS [FILE]...

$ printf 'aXb\\n' | ./grep -o 'X'
X
$ printf 'aXb\\n' | ./grep -oP 'a(?=X)'      # 试探 PCRE 支持
a
$ printf 'a\\nb\\n' | ./grep -c ''            # 空模式的边界行为
2
$ echo abc | ./grep -E 'a{1,'; echo "exit=$?"
grep: Unmatched { or \\{
exit=2                                       # 错误码也要对上

# 每一次探测都是在补全规格，再照着规格写实现
` },
    { label: 'Anti-leak', hint: '为何抗污染', kind: 'html', body:
      '<p>「给二进制反推源码」这个任务方向<strong>本身</strong>抗数据泄漏。' +
      'grep 的源码可能早进过训练语料，但参考答案是<strong>可执行的行为</strong>' +
      '而非某段文本，模型背不下来。</p>' +
      '<dl class="specs">' +
      '<dt>输入</dt><dd>公开可得（编译产物到处都有）</dd>' +
      '<dt>答案</dt><dd>不可记忆（判据是行为等价，不是文本匹配）</dd>' +
      '<dt>判据</dt><dd>完全可执行，248k 行测试</dd>' +
      '</dl>' +
      '<p>最值得抄的就是这个手法：找一个<strong>输入公开、答案却无法记忆的转换方向</strong>。' +
      '想造抗污染的题，先找这样一个方向，再谈规模。</p>' },
  ],
  verdict: [ { cls:'pass', text:'248k test lines' }, { cls:'plain', text:'200 tasks' },
             { cls:'plain', text:'binary → source' } ],
  note: '示意重构 · 从产物重建完整程序',
};

DEMOS['frontierswe'] = {
  bench: 'FrontierSWE', lang: 'zig',
  title: '实现一个无锁并发队列，并把吞吐压过性能线',
  repo: 'Proximal-Labs/frontier-swe', commit: '—', id: 'fswe__lockfree-mpmc-zig',
  tabs: [
    { label: 'Task', hint: '双重门槛', kind: 'html', body:
      '<p>用 Zig 实现一个前沿的无锁 MPMC 队列，再把它的吞吐压到给定的性能线以上。' +
      '<strong>通不过性能门槛，正确也不算过。</strong></p>' +
      '<ul>' +
      '<li>正确性：并发下不能出数据竞争、不能丢消息、不能重复投递</li>' +
      '<li>性能：8 生产者 × 8 消费者下吞吐 ≥ 基线的 0.9×</li>' +
      '<li>不得使用互斥锁，不得依赖外部并发库</li>' +
      '</ul>' +
      '<p>只有 17 题，全落在 Zig / Rust / Haskell / Mojo / 汇编这些' +
      '<strong>训练语料稀薄</strong>的语言上，分实现、研究、性能优化三类。</p>' },
    { label: 'Scoring', hint: '复合评分', kind: 'shell', body: `
# 第一关：正确性（含竞态检测）
$ zig build test -Doptimize=ReleaseSafe
  All 24 tests passed.
$ zig build race-check
  no data race detected over 10^8 ops

# 第二关：性能门槛 —— 过不了这关，正确也不算过
$ zig build bench -Doptimize=ReleaseFast
  producers=8 consumers=8
  throughput   =  41.2 M ops/s
  baseline     =  38.0 M ops/s
  ratio        =   1.08   (threshold >= 0.90)   PASS

# 复合评分同时看正确性与速度，两个门槛都得跨
` },
    { label: 'Why 17 tasks', hint: '换维度', kind: 'html', body:
      '<p>造 benchmark 常默认「区分度靠加长任务、加大题量」。FrontierSWE 给了另一条路：' +
      '<strong>换一个低资源语言或冷门领域，17 道题也能把前沿模型筛出层次。</strong></p>' +
      '<dl class="specs">' +
      '<dt>主流语言现状</dt><dd>顶级模型早已挤在饱和区，同一道算法题拉不开差距</dd>' +
      '<dt>换成 Zig / Mojo</dt><dd>模型没见够，同一道题立刻分层</dd>' +
      '<dt>题量</dt><dd>17（反常地小），但每题信息量极大</dd>' +
      '<dt>三类任务</dt><dd>实现 / 研究 / 性能优化</dd>' +
      '</dl>' +
      '<p>当主流方向饱和时，换维度比加规模更有效。</p>' },
  ],
  verdict: [ { cls:'pass', text:'correctness' }, { cls:'pass', text:'perf threshold' },
             { cls:'plain', text:'17 tasks' } ],
  note: '示意重构 · 人类能力边缘的工程',
};

DEMOS['swe-marathon'] = {
  bench: 'SWE-Marathon', lang: 'multi',
  title: '从零克隆一个功能完整的全栈待办应用',
  repo: 'abundant-ai/swe-marathon', commit: '—', id: 'marathon__fullstack-todo',
  tabs: [
    { label: 'Task', hint: '数小时级', kind: 'html', body:
      '<p>20 道题，每一道都相当于人类工程师<strong>数小时到一天</strong>的连续工作。' +
      '早已不是几百行的单次 issue→patch。</p>' +
      '<p><strong>本题要求：</strong>前端、后端、数据库都要跑通。</p>' +
      '<ul>' +
      '<li>REST API：待办的增删改查 + 鉴权</li>' +
      '<li>持久层：迁移脚本 + 索引</li>' +
      '<li>前端：列表 / 详情 / 筛选，能对接真实 API</li>' +
      '<li>端到端：起服务后浏览器流程可完整走通</li>' +
      '</ul>' +
      '<p>其它题型包括库复现、实现一个能编译目标语言子集的编译器、内核优化。</p>' },
    { label: 'Verification', hint: '多层独立', kind: 'html', body:
      '<p>任务一长，作弊空间就来了——模型会想方设法只让某一个测试变绿、绕开真实目标。' +
      '对策是<strong>彼此独立</strong>的多层验证，任何一层挂掉都不算过。</p>' +
      '<dl class="specs">' +
      '<dt>功能层</dt><dd>单元测试：每个 API 行为独立验证</dd>' +
      '<dt>集成层</dt><dd>真实起服务 + 数据库，跨组件跑通完整流程</dd>' +
      '<dt>边界层</dt><dd>并发写、超长输入、鉴权失败、数据库断连</dd>' +
      '<dt>反投机</dt><dd>硬编码返回值能过功能层，但过不了集成与边界层</dd>' +
      '</dl>' +
      '<p>它测的不是能不能做对一步，而是<strong>能不能连续做对几十步</strong>。</p>' },
    { label: 'Anti-shortcut', hint: '奖励设计', kind: 'shell', body: `
# 模型的一条常见捷径：让目标测试通过，但没真正实现
+ def list_todos(user_id):
+     return [{"id": 1, "title": "test", "done": False}]   # 硬编码

$ pytest tests/unit/test_api.py::test_list          # 功能层
  1 passed                                          # ← 捷径在这里能过

$ pytest tests/integration/                          # 集成层
  ✗ test_created_todo_appears_in_list
    AssertionError: 新建的 todo 不在返回结果里

$ pytest tests/boundary/                             # 边界层
  ✗ test_list_respects_user_scope
    AssertionError: 返回了其他用户的数据

# 长程任务的奖励设计，本质是反投机设计：
# 先假设模型会抄近路，再用彼此独立的检查把每条捷径堵死。
` },
  ],
  verdict: [ { cls:'pass', text:'3-layer verify' }, { cls:'plain', text:'20 tasks' },
             { cls:'warn', text:'hours per task' } ],
  note: '示意重构 · 超长时程自主开发',
};

DEMOS['posttrainbench'] = {
  bench: 'PostTrainBench', lang: 'ml/python',
  title: '给你一个基座模型、一块 GPU、一个时限：把分数提上去',
  repo: 'aisa-group/PostTrainBench', commit: '—', id: 'ptb-lite__gsm8k-uplift',
  tabs: [
    { label: 'Task', hint: '一句话目标', kind: 'html', body:
      '<p>目标只有一句话：<strong>把它在某个基准上的分数提上去。</strong>' +
      '至于用 SFT、GRPO 还是 LoRA，脚本怎么写、超参怎么调，全由 agent 自己决定。</p>' +
      '<dl class="specs">' +
      '<dt>给什么</dt><dd>基座模型权重 + 目标基准 + 一块 GPU + 时限</dd>' +
      '<dt>要什么</dt><dd>后训练之后的模型权重</dd>' +
      '<dt>判什么</dt><dd>只看最终基准分涨了多少</dd>' +
      '<dt>规模</dt><dd>4 个模型 × 7 个基准（K3 blog 用 Lite 变体）</dd>' +
      '</dl>' +
      '<p>它测的不是写业务代码，而是<strong>模型能不能改进模型</strong>——' +
      '把评测对象从「能不能写代码」抬到「能不能自动化 AI 研发」。</p>' },
    { label: 'Agent run', hint: '自主实验闭环', kind: 'shell', body: `
# agent 自己决定路线，自己写脚本，自己调超参
$ nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NVIDIA H100 80GB HBM3, 81559 MiB

$ python prep_data.py --source train.jsonl --format chatml   # 自己准备数据
  wrote 8412 samples

$ python train.py --method grpo --lr 1e-6 --kl 0.02 --steps 400
  step 100  reward 0.412  kl 0.008
  step 400  reward 0.601  kl 0.021
  saved -> ckpt/grpo-400

$ python eval.py --ckpt ckpt/grpo-400 --bench gsm8k
  baseline 41.2  ->  after 52.8   (+11.6)

# 时限内可以反复试：换方法、换超参、换数据配比
` },
    { label: 'Judgement', hint: '结果增量', kind: 'html', body:
      '<p>判<strong>结果增量</strong>、不判过程。后训练没有唯一正确的脚本，' +
      'PostTrainBench 干脆不看过程，只测「跑完后基准分涨了多少」。</p>' +
      '<dl class="specs">' +
      '<dt>判据</dt><dd>Δ score（后训练分 − 基线分）</dd>' +
      '<dt>不判</dt><dd>用了 SFT / GRPO / LoRA 哪条路线、脚本长什么样</dd>' +
      '<dt>约束</dt><dd>限定 GPU 与时间预算，逼出实验取舍</dd>' +
      '</dl>' +
      '<p>当任务<strong>没有标准解、但结果可量化</strong>时，把判据锚在终局指标的增量上——' +
      '天然容纳各种解法，也省掉了标注参考答案这件几乎做不动的事。' +
      '这是「结果奖励」优于「过程奖励」的一个干净案例。</p>' },
  ],
  verdict: [ { cls:'pass', text:'Δ score only' }, { cls:'plain', text:'4 models × 7 benches' },
             { cls:'warn', text:'GPU / time capped' } ],
  note: '示意重构 · agent 做后训练',
};

DEMOS['mls-bench'] = {
  bench: 'MLS-Bench-Lite', lang: 'ml/python',
  title: '受限编辑预算内改一段训练代码，要求换规模、换分布仍成立',
  repo: 'Imbernoulli/MLS-Bench', commit: '—', id: 'mls-lite__generalize-014',
  tabs: [
    { label: 'Task', hint: '受限编辑', kind: 'html', body:
      '<p>给一段在小数据集上跑得不错的 ML 训练代码，<strong>只允许改动有限几行</strong>，' +
      '要求改完之后换到更大规模、换一个数据分布仍然成立。</p>' +
      '<dl class="specs">' +
      '<dt>编辑预算</dt><dd>有限行数——逼出方法层面的改动，而非堆工程量</dd>' +
      '<dt>不看什么</dt><dd>在原任务上的精度</dd>' +
      '<dt>只看什么</dt><dd>留出分布上泛化得如何</dd>' +
      '<dt>规模</dt><dd>完整集 140 题，Lite 变体 30 题</dd>' +
      '</dl>' +
      '<p>它要的不是调包跑通，而是<strong>方法层面真正可泛化的改动</strong>。</p>' },
    { label: 'Held-out', hint: '判分分布', kind: 'shell', body: `
# 优化分布上的表现 —— 两个改法看起来一样好
$ python eval.py --split train-dist
  baseline        0.812
  edit-A (调参)    0.849
  edit-B (改方法)  0.844

# 留出分布上判分 —— 差别在这里才显出来
$ python eval.py --split heldout --scale 8x
  baseline        0.604
  edit-A (调参)    0.598      # 过拟合到原分布，没迁移
  edit-B (改方法)  0.731      # 真泛化

# 只在训练分布上测精度，模型会记住捷径。
# 判分集必须和优化目标不在同一个分布上。
` },
    { label: 'vs PostTrain', hint: '区别', kind: 'html', body:
      '<p>两者都让 agent 动 ML 代码，但要的东西不同。</p>' +
      '<dl class="specs">' +
      '<dt>PostTrainBench</dt><dd>优化<strong>既有</strong>训练流程，判同分布上的分数增量</dd>' +
      '<dt>MLS-Bench</dt><dd>要求提出<strong>方法层面的新做法</strong>，判留出分布上的泛化</dd>' +
      '<dt>共同点</dt><dd>都不判过程，只判可量化的结果</dd>' +
      '</dl>' +
      '<p>它提升的是模型发明可泛化 ML 方法的能力——' +
      '不是记住某个 trick，而是找到能迁移的原理。' +
      '想分清「背答案」和「会方法」，判分集就必须换分布。</p>' },
  ],
  verdict: [ { cls:'pass', text:'held-out judged' }, { cls:'plain', text:'140 (Lite 30)' },
             { cls:'warn', text:'edit budget' } ],
  note: '示意重构 · 逼出真方法而非过拟合',
};

DEMOS['long-horizon'] = {
  bench: 'SWE-EVO / Chain / Cycle / ABC', lang: 'multi',
  title: '长程演化、链式升级、全周期、后端部署：四种判据升级',
  repo: '2026 长程与全流程一线', commit: '—', id: 'long-horizon-frontier',
  tabs: [
    { label: 'SWE-EVO', hint: '21 files', kind: 'html', body:
      '<p><strong>版本化演化。</strong>给定某个库连续几个版本的演进目标，' +
      '模型要跨大量文件做协调一致的重构。</p>' +
      '<dl class="specs">' +
      '<dt>平均改动</dt><dd>21 个文件 / 题</dd>' +
      '<dt>GPT-5.4</dt><dd>25%</dd>' +
      '<dt>判据升级</dt><dd>查<strong>跨文件一致性</strong>——单点测试兜不住协调性</dd>' +
      '</dl>' +
      '<p>一次演化里，接口改了、调用方没跟上，单个测试仍可能是绿的；' +
      '演化任务的判据必须查整体一致性。</p>' },
    { label: 'SWE-Chain', hint: '链式', kind: 'shell', body: `
# 上一步的输出作下一步的输入，1660 题串成版本升级链
step 1  升级 core 到 v2 API
        -> patch_1                       [PASS]
step 2  在 patch_1 之上适配 plugin 层
        -> patch_2   (输入 = 仓库 + patch_1)   [PASS]
step 3  在 patch_2 之上迁移配置格式
        -> patch_3   (输入 = 仓库 + patch_1 + patch_2)   [FAIL]
        AssertionError: plugin 层沿用了 step 2 的旧字段名

# 单步都对，衔接错了整链就断。
# 链式任务的判据要查前后步骤的衔接。
` },
    { label: 'SWE-Cycle', hint: '四模式', kind: 'html', body:
      '<p><strong>全周期。</strong>489 题拆成四种模式，覆盖从零搭起到验证的完整闭环。</p>' +
      '<dl class="specs">' +
      '<dt>环境重建</dt><dd>从仓库状态复原出可跑的开发环境</dd>' +
      '<dt>实现</dt><dd>按需求写出功能代码</dd>' +
      '<dt>测试生成</dt><dd>为自己写的实现补出有效测试</dd>' +
      '<dt>验证</dt><dd>跑通闭环，确认实现与测试互相印证</dd>' +
      '</dl>' +
      '<p>它把「会改代码」和「能独立走完一轮开发」区分开——' +
      '后者要求模型自己造出验证手段，而不是等别人给判据。</p>' },
    { label: 'ABC-Bench', hint: '服务级', kind: 'shell', body: `
# 后端垂直：224 题，要把服务真正部署起来再测
$ docker compose up -d
  db      ... healthy
  api     ... healthy

# 判据不是「代码看起来对」，是 HTTP 端到端行为
$ curl -s -XPOST localhost:8080/orders -d '{"sku":"A1","qty":2}'
  201  {"id":"o_18","status":"pending"}
$ curl -s localhost:8080/orders/o_18
  200  {"id":"o_18","sku":"A1","qty":2,"status":"pending"}
$ curl -s -XPOST localhost:8080/orders -d '{"sku":"A1","qty":-1}'
  400  {"error":"qty must be positive"}

# 服务任务要查部署后的真实行为。
# 判据的粒度必须匹配任务的粒度，否则再长的任务也只是测了一个点。
` },
  ],
  verdict: [ { cls:'pass', text:'exec-based' }, { cls:'plain', text:'判据随形态升级' },
             { cls:'warn', text:'单次 patch 已测不出差距' } ],
  note: '示意重构 · 2026 长程与全流程一线',
};

DEMOS['dual-use'] = {
  bench: 'SWE-rebench / R2E-Gym', lang: 'multi',
  title: 'setup agent 把一个陌生仓库自动变成可判分的新题',
  repo: 'nebius/SWE-rebench', commit: '—', id: 'rebench__auto-setup',
  tabs: [
    { label: 'Pipeline', hint: '无人工介入', kind: 'shell', body: `
# setup agent 拿到一个此前没被处理过的仓库，全程自动
[1/5] clone + 探测语言与构建系统
      detected: python 3.11 · poetry · pytest
[2/5] 装依赖
      $ poetry install --with dev            OK (retry 1/5)
[3/5] 跑通基线测试
      $ poetry run pytest -q                 412 passed
[4/5] 从合并 PR 切分 test patch / code patch
      candidate PR #2841 -> F2P 1 · P2P 412
[5/5] 校验可验证性
      只打 test patch      -> FAIL   ✓ 符合预期
      test + code patch    -> PASS   ✓ 符合预期
      => 产出一道可执行、可判分的新题

# 不靠人工挑题，靠 agent 批量把真实仓库变成可执行环境。
` },
    { label: 'Dual use', hint: '训练 + 评测', kind: 'html', body:
      '<p>有一类环境天生同时服务评测与训练：动态切片既能当评测池，也能当 RL 训练池。</p>' +
      '<dl class="specs">' +
      '<dt>SWE-rebench V1</dt><dd>数千题起步，自动 setup agent 合成可执行任务</dd>' +
      '<dt>SWE-rebench V2</dt><dd>32000+ 题 · 3600+ 仓 · 20 种语言</dd>' +
      '<dt>R2E-Gym</dt><dd>8700+ 题，从 commit diff <strong>反向</strong>合成问题描述，双验证器</dd>' +
      '<dt>提升什么</dt><dd>不是某种模型能力，而是<strong>训练数据的供给</strong></dd>' +
      '</dl>' +
      '<p>为 RL 提供源源不断的可验证任务，这是它们与纯评测集的根本区别。</p>' },
    { label: 'Bottleneck', hint: '成功率', kind: 'html', body:
      '<p>环境合成的瓶颈只有一句话：<strong>能不能自动把一个陌生仓库跑通并切出可验证任务。</strong></p>' +
      '<dl class="specs">' +
      '<dt>失败在哪</dt><dd>装依赖失败、测试跑不起来、日志解析不出结构化信号</dd>' +
      '<dt>为何关键</dt><dd>成功率直接决定单位成本，进而决定训练池的规模与新鲜度</dd>' +
      '<dt>与本系列的关系</dt><dd>与 AutoBuilder 的目标完全一致（12% → 48%）</dd>' +
      '</dl>' +
      '<p>谁的自动 setup 成功率高，谁就能以更低成本产出更大、更新鲜的训练池。</p>' },
  ],
  verdict: [ { cls:'pass', text:'auto-synthesized' }, { cls:'plain', text:'32000+ tasks' },
             { cls:'plain', text:'train + eval' } ],
  note: '示意重构 · 自动化合成的双用池',
};


