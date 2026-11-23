# PrimeWordle

一个基于“哥德尔质数乘积编码”抽象出的泛化 Wordle / Mastermind 逻辑解密小游戏，以及配套的规则求解器、合成数据生成器和从零训练的 25M 迷你 Transformer（无思维链，纯序列决策）。

## 游戏规则与编码思路

玩法灵感来自《植物大战僵尸2》改版里的“植物解码”，用 3 个参数 $(P, K, L)$ 统一描述一类多重集猜谜问题：

- **$P \in [2, 32]$**：基础符号数量，底层按顺序映射到前 $P$ 个质数（`2, 3, 5, ..., 131`）。
- **$K \in [1, 8]$**：单个格子最多相乘的符号数量。$K=1$ 退化为标准 Wordle / Mastermind，$K=2$ 对应两两融合，$K \ge 3$ 为高阶多重集组合。
- **$L \in [1, 32]$**：密码格子数。
- 另有 `allow_underfill`（是否允许每格填 $1 \sim K$ 个符号）和 `allow_duplicate_slots`（是否允许不同格子答案相同）两个开关。

利用算术基本定理，每个格子把所选符号对应的质数相乘，得到唯一的哥德尔整数（自动支持重复幂次且与顺序无关）。单轮校验按严格优先级返回 4 种状态：

`EXACT (完全相同) > MISPLACED (乘积出现在其他未匹配格) > PARTIAL (与当前格目标 gcd > 1) > WRONG (互质)`

对模型来说不需要处理大整数，直接操作 80 个离散符号 Token 即可：
- 16 个结构/配置/反馈 Token（含 `<PAD>/<BOS>/<EOS>/<TURN>/<SLOT>/<FB>/<CFG>/<UNDERFILL_*>/<DUP_*>/<UNK_CFG>` 与 4 种反馈）
- 32 个符号 Token（`S_0 .. S_31`）
- 32 个配置数字 Token（`N_1 .. N_32`）

外部规则约束（例如某些组合禁止融合、26 字母英文词典 Trie 等）可在解码层通过动态 Logits Mask 实现，不需要修改底层词表。

## 项目结构

```text
engine/       核心引擎：质数编码、4 态校验、单格记牌器、80 词表 Tokenizer
solver/       基线求解器（精确候选池 + 大空间分组探测）、合成数据生成器、基准测试
model/        纯 PyTorch 16 层 Decoder-only Transformer（25.2M 参数，RoPE 32K，GQA，QK-Norm，SwiGLU）
train/        多 shard 流式 SFT 训练、GRPO / GRPO v2 强化学习、压测与 310 局盲考脚本
web/          零依赖本地 HTTP 服务与前端小游戏界面（含记牌器、笔记本、本地存档）
weights/      最终导出的纯推理 bf16 权重（primegpt_25m_grpo_v2_bf16.pt，约 48 MB）
tests/        引擎、求解器、模型、训练与 GRPO 单元测试
play_demo.py  命令行快速推理演示脚本
run_web.py    本地网页小游戏启动入口
```

## 快速开始

### 1. 环境依赖

- Python 3.11+
- 仅玩网页小游戏或运行基线求解器：零第三方依赖（仅标准库）。
- 运行模型推理或训练：安装 PyTorch 与 NumPy（推荐支持 `bf16` 的 CUDA GPU，也可在 CPU 上跑单局推理）：

```bash
pip install -r requirements.txt
```

### 2. 直接玩本地网页小游戏

```bash
python run_web.py
```

### 3. 用附带权重跑一局模型推理

仓库 `weights/primegpt_25m_grpo_v2_bf16.pt` 附带了最终训练好的 25M 推理权重（`bf16`，约 48 MB）：

```bash
# 默认跑一局 P=5, K=2, L=5（允许不满填）
python play_demo.py

# 自定义难度，例如 P=10, K=2, L=10 强制满填 + 开启外层防同格复读护栏
python play_demo.py --p 10 --k 2 --l 10 --no-underfill --anti-rep
```

### 4. 运行单元测试

```bash
python -m unittest discover -s tests -v
```

## 训练与数据生成复现

模型参数量约 **25.21M**（16 层、`d_model=384`、6 Q heads / 2 KV heads、`d_ff=1024`、RoPE `theta=500000`），全部训练在单张笔记本 RTX 4060（8GB VRAM）上完成：

1. **生成一阶段全域 SFT 分片**（例如 8 个 5,000 万 Token 的 `uint8` `.pt` 分片，共 4 亿不重复 Token）：
   ```bash
   python -m solver.generate_dataset --shard-dir data/sft/run_400m --num-shards 8 --target-tokens 50000000 --workers 12
   ```
2. **一阶段纯 1-Epoch SFT**（只对模型猜测段和 `<EOS>` 计算 Loss，屏蔽配置头和环境反馈）：
   ```bash
   python -m train.run_overnight_400m
   ```
3. **二阶段抗干扰 SFT（1 亿 Token）**：混入 40% 脏残局接盘（脏前缀 `loss_mask=0`）+ 35% 随机稀疏禁乘图（含 `UNDERFILL_OFF` 与禁止自叠）+ 25% 全域防遗忘，低学习率续训：
   ```bash
   python -m train.run_round2_adv_100m
   ```
4. **三阶段全科 GRPO 强化学习**：关闭外层防复读进行温度采样，冻结参考模型加 KL 惩罚（`kl_coef=0.05`）与踩雷/复读惩罚：
   ```bash
   python -m train.grpo_v2
   ```

## 实测结果（310 局独立盲考参考）

在固定随机种子、未见过的参数组合上（`train/blind_exam.py`，关闭外层防复读 `AntiRep=OFF`），最终权重 `primegpt_25m_grpo_v2_bf16.pt` 的实测表现如下：

| 测试项 | 局数 | 胜率 | 平均补充轮数 | 排除雷 (`wrong_mine`) | 丢分雷 (`exact_drop`) | 复读雷 (`repeat_mine`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 陌生小盘（`K=1..2`，含随机脏残局） | 30 | 100.0% | 5.50 | 2.7% | 0.0% | 2.0% |
| 陌生中盘（`K=3..4`，`P=5..24`） | 30 | 80.0% | 14.23 | 1.4% | 0.0% | 4.1% |
| 陌生稀疏禁乘图（`K=2`，30%~70% 禁乘+脏残局） | 30 | 93.3% | 9.17 | 10.7% | 0.0% | 10.7% |
| 陌生高阶（`K=5..8`，`P=10..32`） | 30 | 63.3% | 20.07 | 2.3% | 0.0% | 1.9% |
| **120 局陌生 OOD 汇总** | **120** | **84.2%** | **12.24** | **3.8%** | **0.0%** | **4.4%** |
| 隐藏规则 `<UNK_CFG>`（带 2~3 轮前缀） | 20 | 70.0% | 8.45 | 0.7% | 0.0% | 4.9% |
| 零样本英文 Wordle（外层 2,587~2,977 词典 Trie） | 150 | 99% (5词10轮) / 65% (6轮内) | 5.91 (5字母) | — | — | — |

说明：若在实际部署时开启外层防同格复读护栏（`--anti-rep`），上述 `K=2` 50% 禁乘 10×10 盘面（含 5 轮随机残局）实测胜率可达 100%，平均 7.4 轮左右完成。

## License

[MIT](LICENSE)
