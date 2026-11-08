Black Box RL Gateway 建设：业界多数工作已将 rl infra 框架中的 Rollout Server 和 Harness Environment 基于 Gateway 这一中间件进行 bridge，以实现纯粹的 TITO（token-in token-out），并兼容所有真实生产场景的 Harness。我们完成了对此前基于 Chat 接口的 Proxyserver 升级，进而保障 RL 训练过程 Token 的绝对一致性，优化训练稳定性，大幅提升了训练过程的 Token efficiency。
	○ [2605 Nvidia] Polar: Agentic RL on Any Harness at Scale & [2602 Minimax-Forge] 
 
