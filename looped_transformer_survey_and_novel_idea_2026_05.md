# Looped Transformer 전수 조사 + 신규 논문 아이디어 / 실험 방향

작성일: 2026-05-28
대상: 자체 4B hybrid-reasoning LLM(Gemma4 토크나이저, pre/mid-training) 구축자
조사 규모: 5개 클러스터, 85+ 논문(2016 ACT ~ 2026-05 arXiv), 병렬 리서치 에이전트 + 표적 신규성 검증

---

## 0. TL;DR (먼저 읽으세요)

1. **이 분야는 2026년 5월 현재 극도로 포화 상태다.** 2025 하반기~2026 상반기에만 looped/recurrent-depth 관련 논문이 수십 편 쏟아졌다. "looped transformer로 적응적 추론" 류의 **명백한 아이디어는 거의 전부 선점**됐다(아래 §3 점유 리스트 참고). 표절/중복 위험이 매우 높으므로, 아무 아이디어나 던지면 99% 이미 존재한다.

2. **이미 점유되어 절대 새 논문거리가 안 되는 것들**: pretraining부터 looped LM(Huginn, Ouro), 학습된 per-token 적응 halting(Ouro entropy-gate, AdaPonderLM, PonderLM-3), elastic depth(LoopFormer), fixed-point/equilibrium(Attractor), looped+MoE 스케일링(Sparse-Layers), latent↔explicit 하이브리드 스위칭(SwiReasoning, SLT, HybridCoT, LT-Tuning), CoT-step↔loop 정렬 supervision(RELAY), value-of-computation halting(Rational Metareasoning), monotonic improvement supervision(2511.16886), overthinking 페널티 halting(Ouro adaptive-loss), conformal early-exit(LYNX).

3. **그럼에도 일관되게 비어 있는 단 하나의 과학적 공백**: 기존의 모든 halting 신호(KL 수렴, entropy, confidence, fixed-point, loss-improvement)는 **"정답률–깊이 곡선이 단조(monotone)"라고 암묵적으로 가정**한다. 그러나 overthinking 증거(Ouro는 T=4–5에서 정점 후 성능 저하, "Loop, Think & Generalize"는 over-recurrence가 예측을 망친다고 명시)는 이 곡선이 **봉우리형(interior optimum) — 정점을 지나면 정답률이 능동적으로 하락**할 수 있음을 시사한다. **이 비단조성을 정면으로 측정·예측·보정(특히 양방향 conformal)하고, 정점을 지난 looping을 능동적으로 차단하는 연구는 없다.**

4. **제안(아래 §4)**: 이 공백을 겨냥한 1개 메인 + 2개 대안.
   - **메인 — "Overthinking은 잠재 깊이의 내부 최적점이다" + PeakDepth halting**: 비단조 정답률–깊이 곡선을 4B 규모에서 실증하고, per-instance argmax-정답률 깊이를 예측하는 halting head를 정답으로 supervise하고, **under-shoot와 over-shoot를 동시에 bound하는 양방향 conformal 보정**으로 calibrate. 사용자의 overthinking 문제를 직접 겨냥.
   - **대안 A — Loop–Token 교환비(exchange rate) 연구 + 공동 예산 컨트롤러**: 잠재 loop와 명시 토큰을 두 개의 "연산 화폐"로 보고 둘 사이의 정량적 교환비 r(난이도)를 측정, from-scratch로 둘을 모두 유창하게 쓰고 trade하는 모델 학습.
   - **대안 B — 자기참조가 아닌 robustness halting**: "수렴했지만 틀린" 실패를 막기 위해 잠재 궤적을 섭동시켜 답이 불변일 때 정지(섭동-일관성 halting).

5. **사용자 자산과의 정렬**: 사용자는 답이 달린 reasoning 데이터(정답률–깊이 곡선 측정 가능)와 explicit CoT 길이(난이도 proxy)를 이미 보유 → 메인 아이디어를 mid-training에서 바로 실험 가능. closed-source distill 0% 정책과도 충돌 없음(자체 모델 forward만 사용).

> 정직한 경고: 이 분야의 속도상, 아래 제안도 6~12개월 내 누군가 부분적으로 건드릴 수 있다. 그래서 각 제안마다 "최근접 이웃 논문 + 차별화 포인트"를 명시했다(§4, §5). 실제 착수 전 §5 체크리스트로 최종 arXiv 재검색을 반드시 하라.

---

## 1. 조사 범위 및 방법

5개 클러스터로 분할해 병렬 조사 후 표적 신규성 검증 검색을 추가했다.

- **C1. 이론·기초**: Universal Transformer, ALBERT/weight-sharing, Looped-as-Programmable-Computer, DEQ, Turing-completeness, circuit complexity, CoT 표현력.
- **C2. Latent reasoning / recurrent depth (2024–2026)**: Huginn, Coconut, Saunshi, Ouro, HRM/TRM, 최신 follow-up.
- **C3. 적응적 연산·halting·early-exit·MoD**: ACT, PonderNet, MoD, CALM, MoR, AdaPonderLM, PonderLM-3, 명시 CoT 길이 제어(L1/AdaptThink/CODA 등).
- **C4. 효율 recursive LLM·KV공유·시스템·최신 2026 아키텍처**: RecurrentGemma/Griffin, YOCO/KV-sharing, Relaxed Recursive, Sparse-Layers(looped+MoE), PLT, Hyperloop, SpiralFormer, Attractor 등.
- **C5. 인접 패러다임·하이브리드 추론**: diffusion-as-depth, Energy-Based Transformer, System-1/2, latent↔explicit 스위칭, TTT, pause/thinking token, 명시 난이도 모델, latent overthinking 비판.

용어 정리(혼동 방지):
- **Depth-recurrence(=looping)**: 같은 블록을 깊이 방향으로 반복 적용. test-time compute가 메모리/컨텍스트를 늘리지 않음. (본 보고서의 주제)
- **Sequence-recurrence**: RNN/SSM처럼 시간(토큰) 축으로 상태를 굴림(Griffin, Mamba). 주제와 구별됨.
- **Latent reasoning**: 토큰을 내뱉지 않고 hidden state에서 추론. looping은 그 한 구현.

---

## 2. 문헌 지도 (클러스터별, 원문 인용 포함)

각 항목: **제목**(저자, venue/연도) — arXiv. 핵심 / "원문 인용" / 메커니즘 / 한계.

### 2.1 이론·기초 (C1)

**Adaptive Computation Time (ACT)** (Graves, 2016) — arXiv:1603.08983.
- 학습된 halting의 원조. RNN이 입력당 ponder step 수를 학습.
- "an algorithm that allows recurrent neural networks to learn how many computational steps to take between receiving an input and emitting an output."
- 시그모이드 halting unit + 누적 halting prob ≥ 1−ε에서 정지 + ponder cost 정규화. 한계: gradient 편향, τ 튜닝 민감.

**Universal Transformers (UT)** (Dehghani et al., ICLR 2019) — arXiv:1807.03819.
- 가중치 공유 블록을 깊이 방향 반복 + per-position ACT halting. 조건부 Turing-complete.
- "a parallel-in-time self-attentive recurrent sequence model ... combine the parallelizability ... of the Transformer with the recurrent inductive bias of RNNs." / "a dynamic per-position halting mechanism."
- 모든 후속 연구의 조상. 한계: 소규모 task, ACT 편향 상속.

**ALBERT** (Lan et al., ICLR 2020) — arXiv:1909.11942.
- cross-layer 가중치 전면 공유 = 단일 블록을 L번 looping(단, halting/timestep encoding 없음). 효율 동기.
- 한계: encoder-only, 표현력/iterative 동기 아님. 공유는 파라미터만 줄이고 FLOPs는 동일.

**Looped Transformers as Programmable Computers** (Giannou et al., ICML 2023) — arXiv:2301.13196.
- 가중치를 손으로 짜고 loop에 넣으면 13층 트랜스포머가 소형 명령셋 컴퓨터(SUBLEQ)를 에뮬레이트.
- "using transformer networks as universal computers by programming them with specific weights and placing them in a loop. Our input sequence acts as a punchcard."
- 한계: 구성적 존재 증명(학습 아님), 고정밀/하드맥스 이상화.

**Deep Equilibrium Models (DEQ)** (Bai et al., NeurIPS 2019) — arXiv:1909.01377.
- 단일 가중치공유 층의 고정점 = 무한 깊이. root-finding + implicit diff, 메모리 상수.
- "equivalent to running an infinite depth (weight-tied) feedforward network ... only constant memory, regardless of the effective 'depth'."
- 한계: 고정점 존재/안정성 비보장, solver 비용.

**Circuit-complexity 라인** (looping이 왜 깊이를 사는가):
- Merrill & Sabharwal, *Saturated Transformers are Constant-Depth Threshold Circuits* (TACL 2022, arXiv:2106.16213): 고정깊이는 TC⁰ 상한.
- Merrill & Sabharwal, *The Parallelism Tradeoff* (TACL 2023, arXiv:2207.00729): log-precision 트랜스포머 ⊆ uniform TC⁰. "any model architecture as parallelizable as the transformer will obey limitations similar to it."
- Li et al., *CoT Empowers Transformers to Solve Inherently Serial Problems* (ICLR 2024, arXiv:2402.12875): "With T steps of CoT, constant-depth transformers ... can solve any problem solvable by boolean circuits of size T." (CoT 길이 = serial 연산 예산)
- Merrill & Sabharwal, *Exact Expressive Power of Transformers with Padding* (NeurIPS 2025, arXiv:2505.18948): looped+padded = O(log^d n) loop ⇒ uniform TC^d. (**loop 횟수 = 회로 깊이 지수** — 이미 정형화됨)
- *A Little Depth Goes a Long Way* (Merrill & Sabharwal, 2025, arXiv:2503.03961): 깊이 Θ(log n)(=가변 looping)이 정규언어·그래프 연결성 표현. "depth scaling is more efficient than scaling width or chain-of-thought steps."
- *On Expressive Power of Looped Transformers ... Timestep Encoding* (2024, arXiv:2410.01405): 순수 looping의 근사력 한계 + timestep(loop-index) encoding으로 보완.

**ICL/알고리즘 학습**:
- von Oswald et al., *Transformers Learn In-Context by Gradient Descent* (ICML 2023, arXiv:2212.07677): 1 linear-attention 층 = 1 GD step.
- Yang et al., *Looped Transformers are Better at Learning Learning Algorithms* (ICLR 2024, arXiv:2311.12424): "comparable to the standard transformer ... while utilizing less than 10% of the parameter count."
- Gatmiry et al., *Can Looped Transformers Learn to Implement Multi-step GD ...* (ICML 2024, arXiv:2410.08292): linear looped의 전역최소 = multi-step preconditioned GD(학습가능성 증명).
- Fan et al., *Looped Transformers for Length Generalization* (2024, arXiv:2409.15647): 적응적 step 수로 길이 일반화.

### 2.2 Latent reasoning / recurrent depth (C2) — 핵심 클러스터

**Scaling up Test-Time Compute with Latent Reasoning (Huginn-3.5B)** (Geiping et al., NeurIPS 2025 spotlight) — arXiv:2502.05171.
- prelude→recurrent core→coda. 코어 블록을 test-time에 임의 횟수 반복. 800B 토큰, 3.5B.
- "Our model works by iterating a recurrent block, thereby unrolling to arbitrary depth at test-time." / "does not require any specialized training data, can work with small context windows."
- 학습 시 반복 횟수 랜덤 샘플(log-normal-Poisson, 평균 ~32). **정지는 휴리스틱 사후**: 연속 step 출력분포 KL < 5e-4면 정지. 학습된 게이트 아님.
- 한계(저자 명시): exit가 "zero-shot" 휴리스틱, latent 해석 불가.

**Scaling Latent Reasoning via Looped Language Models (Ouro / LoopLM)** (Zhu, Wang, ... Bengio, Eshraghian, 2025) — arXiv:2510.25741. **(가장 강력한 선행)**
- from-scratch 7.7T 토큰 looped LM(1.4B/2.6B). 학습된 적응 깊이. 1.4B/2.6B가 최대 12B SOTA에 필적.
- "(i) iterative computation in latent space, (ii) an entropy-regularized objective for learned depth allocation, and (iii) scaling to 7.7T tokens." / "This advantage stems not from increased knowledge capacity, but from superior knowledge manipulation."
- **per-step exit gate λ_t(x)** + entropy 정규화(uniform prior로 난이도-구동 exit과 전역 compute 선호 분리). **Stage-II adaptive loss: looping이 loss를 줄이면 stop에 페널티, 안 줄이면 overthinking에 페널티.** ← 이게 "loss-improvement gated halting"을 이미 점유.
- 한계(저자 명시): **깊이 T=4로 캡**, 학습 깊이 초과 외삽 시 degrade, **RL alignment(DAPO/GRPO) 실패**, 가변 깊이가 표준 추론 프레임워크 깨뜨림.

**Reasoning with Latent Thoughts: On the Power of Looped Transformers** (Saunshi et al., Google, ICLR 2025) — arXiv:2502.17416.
- k층 블록 L회 looping ≈ kL층 비-looped, ≫ k층. "Accuracy on downstream tasks scale as logarithm of the effective depth." / "Looped models generate latent thoughts and can, in theory, simulate CoT reasoning." (T loop = T CoT step)
- 한계(저자 명시): looped는 iso-flops 대비 perplexity 나쁨(파라미터 의존), "Leveraging looping explicitly for inference-time scaling is a very promising future direction"(당시 미완).

**Training LLMs to Reason in a Continuous Latent Space (Coconut)** (Hao et al., Meta, COLM 2025) — arXiv:2412.06769.
- 마지막 hidden state를 다음 입력 임베딩으로 되먹임("continuous thought"). 토큰공간 latent CoT.
- "continuous thoughts can encode multiple alternative next steps, allowing ... breadth-first search (BFS)."
- 커리큘럼으로 CoT step을 latent로 치환(고정 개수). 한계(저자 명시): "further research is definitely needed to develop better and more general strategies for learning reasoning in latent space, especially without the supervision from language reasoning chains"; c=3에서 불안정.

**Hierarchical Reasoning Model (HRM)** (Wang et al., 2025) — arXiv:2506.21734 / **Tiny Recursive Model (TRM)** (Jolicoeur-Martineau, 2025) — arXiv:2510.04871.
- HRM: 2개 시간스케일 recurrent 모듈 + 1-step gradient(BPTT 회피) + ACT halting. 27M, Sudoku-Extreme 거의 완벽.
- TRM: 단일 2층 net 재귀, 7M으로 ARC-AGI-1 45%. "a single tiny network with only 2 layers."
- 한계: 퍼즐 특화, 일반 LM 아님.

**Mechanistic / 해석 비판**(중요 — 메인 아이디어의 근거):
- *Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer* (Lu et al., 2025, arXiv:2507.02199): Huginn에서 "limited evidence of interpretable latent CoT" / "increasing recurrence depth yields only marginal gains and falls well short of models that explicitly externalize reasoning." (GSM8K 4.93% vs 명시 CoT 24.87%)
- *Do Latent-CoT Models Think Step-by-Step?* (Liang & Pan, 2026, arXiv:2602.00449): Coconut/Huginn이 진짜 순차추론을 하지 않고 shortcut/parallel 전략을 쓴다고 결론.
- *Loop, Think, & Generalize* (Kohli et al., 2026, arXiv:2604.07822): "generalization beyond training depth can be unlocked by scaling inference-time recurrence" 그러나 **"overthinking, where excessive recurrence degrades predictions and limits generalization."** ← 비단조성의 직접 증거.
- *Two-Scale Latent Dynamics* (Pappone et al., 2025, arXiv:2509.23314): loop step이 점점 작아지고 직교화("spiral"). 2차 차분(가속도) exit이 Huginn KL exit보다 우수. (여전히 휴리스틱·수렴기반)
- *A Mechanistic Analysis of Looped Reasoning LMs* (2026, arXiv:2604.11791): "Recurrent blocks learn stages of inference that closely mirror those of feedforward models, repeating these stages in depth." cyclic fixed point으로 수렴.

**Your Latent Reasoning is Secretly a Policy Improvement Operator / Deep Improvement Supervision** (2025, arXiv:2511.16886): monotonic corruption schedule로 각 재귀 step을 supervised sub-goal로. "LLM-generated trajectories fail to provide a monotonic improvement path." ← monotonic-improvement supervision 점유.

### 2.3 적응 연산·halting (C3)

**PonderNet** (Banino et al., 2021) — arXiv:2107.05407: 확률적(Bernoulli/geometric) halting, 비편향 gradient, geometric prior KL. ACT의 개선.

**Mixture-of-Depths (MoD)** (Raposo et al., DeepMind 2024) — arXiv:2404.02258: per-token per-layer top-k 라우팅(고정 예산). "learn to dynamically allocate FLOPs ... to specific positions."

**Mixture-of-Recursions (MoR)** (Bae et al., NeurIPS 2025) — arXiv:2507.10524: 가중치공유 재귀 블록 + 경량 라우터가 per-token 재귀 깊이. **명시적 난이도 metric에 halting을 연결하지 않음**(암묵적 라우팅).

**CALM (Confident Adaptive LM)** (Schuster et al., NeurIPS 2022) — arXiv:2207.07061: per-token early-exit(고정깊이 stack) + **local→global 통계 보장**. "global, sequence-level constraints ... are provably maintained with arbitrarily high probability." (단, **깊을수록 좋다는 단조 가정**, layer-exit이지 loop 아님)

**AdaPonderLM** (2026, arXiv:2603.01914): self-supervised recurrent LM, pretraining 중 per-token early-exit 학습. "spends fewer recurrent steps on 'easy' tokens while reserving deeper computation for 'hard' tokens." 게이트가 **high-NLL(어려운) 토큰에 더 많은 step 할당**. (난이도=NLL, 자기참조)

**PonderLM-3** (2026, arXiv:2603.02023): 미분가능 attention mask로 per-token 적응 refinement. "extra steps deliver substantially larger gains on hard tokens, while improvements on easy tokens saturate quickly." 한계(저자 명시): **각 토큰 halting 독립 — 한 토큰의 pondering이 이후 토큰에 주는 영향(lookahead) 미처리.**

**명시 CoT 길이 제어**(토큰공간, 잠재 깊이와 구별): L1/LCPO(arXiv:2503.04697), SelfBudgeter(2505.11274), AdaptThink(2505.13417, think/no-think 선택), CODA(2603.08659, 난이도→CoT 길이), TALE(2412.18547), BudgetThinker(2508.17196).

**Rational Metareasoning for LLMs** (De Sabbata et al., 2024) — arXiv:2410.05563: **Value-of-Computation으로 LLM 추론 비용 최적화**(토큰 20–37% 절감). "If no computation has positive VOC, the agent should stop thinking." ← VOC-기반 halting을 명시 추론에 이미 적용.

**LYNX: Learning Dynamic Exits for Confidence-Controlled Reasoning** (2025, arXiv:2512.05325): "wraps probe scores in split conformal prediction, turning them into an adaptive stop/continue rule ... distribution-free guarantee on the rate of premature exits." ← **conformal early-exit 점유**(단, 명시 CoT stop/continue, 단조 가정, premature-exit 단방향 bound).

### 2.4 효율 recursive / 시스템 / 최신 2026 아키텍처 (C4)

- **RecurrentGemma/Griffin/Hawk** (arXiv:2404.07839, 2402.19427): 게이트 선형 recurrence + local attention. **시간축 recurrence(깊이 looping 아님)**, 고정상태 장문 효율.
- **Relaxed Recursive Transformers** (Bae et al., ICLR 2025, arXiv:2410.20672): pretrained LLM → recursive, per-loop LoRA로 tying 완화 + Continuous Depth-wise Batching.
- **Teaching Pretrained LMs to Think Deeper with Retrofitted Recurrence** (McLeish et al., 2025, arXiv:2511.07384): 기존 dense LM을 model surgery로 depth-recurrent화 + 재귀 커리큘럼 + Muon. "increasing compute does not increase memory consumption or context size."
- **Sparse Layers are Critical to Scaling Looped LMs** (Lee et al., 2026, arXiv:2605.09165): dense looped는 baseline에 미달하나 **FFN을 MoE로 바꾸면 격차 해소/역전**(각 loop마다 다른 expert 활성→loop별 표현력 회복). "Sparse experts resolve the looped scaling deficit." ← **looped+MoE 점유.**
- **Parallel Loop Transformer (PLT)** (ByteDance, 2025, arXiv:2510.24824): Cross-Loop Parallelism로 looped 추론 지연 제거 + loop간 KV 공유. (L≤3만 검증)
- **Hyperloop Transformers** (Zeitoun et al., MIT, 2026, arXiv:2604.21254), **SpiralFormer**(multi-resolution 재귀, 2602.11698), **Attractor Models / Solve the Loop**(DEQ↔loop, 2605.12466), **Universal YOCO**(2604.01220), **Mixture-of-LoRAs / ModernALBERT**(2512.12880).
- KV 공유: YOCO(2405.05254), 9-config taxonomy(2410.14442), MiniCache(2405.14366), KVSharer(2410.18517, "dissimilar 공유가 더 좋다"), xKV(2503.18893).
- 시스템: Diagonal Batching(2506.05229), DREX 동적 rebatching(2512.15705).

### 2.5 인접 패러다임·하이브리드 (C5)

- **Diffusion-as-depth**: Diffusion of Thoughts(2402.07754), DCoLT(2505.10446, denoising step = thinking action + outcome RL), Jot(per-token early-stop, 2602.11133).
- **Energy-Based Transformer (EBT)** (Gladstone et al., 2025, arXiv:2507.02092): test-time GD로 energy 최소화 = "thinking". "humans naturally allocate varying amounts of effort ... depending on difficulty." **한계(저자 명시): "No adaptive stopping mechanism based on energy convergence or uncertainty thresholds."** ← energy 신호를 갖고도 적응 정지에 안 씀.
- **System-1/2**: Dualformer(fast/slow, auto-mode, arXiv:2410.09918, 둘 다 토큰공간), CDR(2508.16636), CogRouter(2602.12662, step별 cognitive depth RL).
- **Latent↔explicit 스위칭**: SwiReasoning(2510.05069, **training-free** entropy-trend 스위치, 비-looped), LT-Tuning(2602.10229, per-token 학습 스위치), **Selective Latent Thinking(SLT)**(2605.25745, span별 latent 압축 + 불확실/정밀 부분은 explicit 유지), HybridCoT(latent+text interleave).
- **Test-Time Training (TTT)** (Sun et al., ICML 2025, arXiv:2407.04620): hidden state가 ML 모델, per-token self-supervised step(고정 1 inner step).
- **Pause/thinking token**: Goyal et al.(2310.02226), Quiet-STaR(2403.09629), Soft Thinking(2505.15778, training-free concept token).
- **명시 난이도 모델**: Zhao et al.(2511.03808, mid-layer hidden으로 난이도/정답확률 예측 → **모델 간 라우팅**), CODA(2603.08659).
- **latent overthinking / 비판**: FR-Ponder(2509.24238, frozen 모델 위 <1M 컨트롤러 per-token halting), Learning When to Stop(2511.21581, looped latent + binary halting + GRPO, "best latent variant still 7pp below CoT"), *Are Latent Reasoning Models Easily Interpretable?*(2604.04902, "latent tokens underutilized ... may partially explain why LRMs do not consistently outperform explicit reasoning"), *Demystifying Hybrid Thinking*(2510.12680, think/no-think "affords only limited control").
- **Thinking in Latents: Adaptive Anchor Refinement** (2603.15051): "instance-wise compute allocation under a shared maximum-step budget" + 수렴 시 정지.

---

## 3. "이미 점유된 아이디어" 마스터 리스트 (표절 가드레일)

아래는 **이미 존재하므로 새 논문 주제가 될 수 없는** 핵심 아이디어들. 제안 전 반드시 대조하라.

| # | 점유된 아이디어 | 대표 논문 |
|---|---|---|
| 1 | 가중치공유 블록 깊이 반복 + per-position ACT halting | UT (1807.03819) |
| 2 | 손-프로그래밍 looped = 범용 컴퓨터(SUBLEQ) | Giannou (2301.13196) |
| 3 | 무한깊이 고정점 + implicit diff(상수 메모리) | DEQ (1909.01377), Attractor (2605.12466) |
| 4 | loop 횟수 = 회로 깊이(TC^d) / T loop = T CoT step | Padding (2505.18948), Saunshi (2502.17416) |
| 5 | from-scratch large-scale looped pretraining | Huginn (2502.05171), Ouro (2510.25741) |
| 6 | 학습된 per-token 적응 halting(난이도=entropy/NLL) | Ouro, AdaPonderLM (2603.01914), PonderLM-3 (2603.02023) |
| 7 | **loss-improvement 게이트 halting(overthinking 페널티 포함)** | **Ouro Stage-II adaptive loss** |
| 8 | per-token 재귀 깊이 라우팅(예산 top-k) | MoR (2507.10524), MoD (2404.02258) |
| 9 | elastic/any-budget 추론 깊이(time+Δt 조건화) | LoopFormer (2602.11451) |
| 10 | looped + MoE 스케일링(loop별 다른 expert) | Sparse-Layers (2605.09165), MoEUT (2405.16039) |
| 11 | CoT step ↔ loop iteration 정렬 + intermediate supervision | RELAY (2502.08482) |
| 12 | monotonic improvement supervision(재귀 step=sub-goal) | Deep Improvement Supervision (2511.16886) |
| 13 | latent↔explicit 적응 스위칭/하이브리드 | SwiReasoning (2510.05069), SLT (2605.25745), LT-Tuning (2602.10229) |
| 14 | Value-of-Computation 기반 halting | Rational Metareasoning (2410.05563) |
| 15 | conformal/통계보장 early-exit | LYNX (2512.05325), CALM (2207.07061), SAFE-KD (2602.03043) |
| 16 | 명시 난이도 모델 → 연산/CoT길이 할당 | Zhao (2511.03808), CODA (2603.08659) |
| 17 | frozen 모델 위 학습 latent halting 컨트롤러 | FR-Ponder (2509.24238), Learning When to Stop (2511.21581) |
| 18 | 2차 차분/가속도 early-exit, fixed-point 수렴 정지 | Two-Scale (2509.23314) |
| 19 | think/no-think 모드 선택 | AdaptThink (2505.13417), Qwen3 |
| 20 | 재귀 커리큘럼(학습 중 평균 깊이 annealing) | Retrofitted Recurrence (2511.07384) |

---

## 4. 신규 아이디어 + 실험 방향

### 4.1 메인 제안 — "Overthinking은 잠재 깊이의 내부 최적점이다" + **PeakDepth halting**

#### 한 줄 요약
> 모든 기존 looped halting은 정답률–깊이 곡선이 **단조(깊을수록 ≥ 좋다)**라고 가정한다. 우리는 이 곡선이 실제로는 **봉우리형(interior optimum)** — 문제마다 특정 깊이에서 정답률이 정점을 찍고 그 이후엔 능동적으로 하락 — 임을 4B 규모에서 실증하고, **per-instance 정점 깊이 d\*(x)를 정답으로 supervise하여 예측**하며, **under-shoot(미사고)와 over-shoot(과사고)를 동시에 bound하는 양방향(two-sided) conformal**로 보정하는 halting을 제안한다.

#### 왜 이게 사용자에게 맞나
사용자의 명시적 적(敵)은 **overthinking**(쉬운 문제 과사고)이다. 메모리상 hybrid reasoning 목표 = 어려운 태스크는 길게, 쉬운 태스크는 짧게/전혀. PeakDepth는 정확히 이 둘을 한 메커니즘으로 해결한다: 쉬운 문제는 낮은 d\*(빠른 정점)에서, 어려운 문제는 높은 d\*에서 정지하되 **정점을 넘는 looping을 능동 차단**. 사용자는 답이 달린 reasoning 데이터(곡선 측정 가능)와 explicit CoT 길이(난이도 prior)를 이미 보유 → mid-training에서 바로 실험 가능.

#### 핵심 가설 (3개)
- **H1 (실증)**: 충분히 깊게 unroll하면(예: 1~32 loop), 상당 비율의 문제에서 p(정답)–깊이 곡선이 **비단조(humped)**다. 정점 위치 d\*는 문제 난이도와 단조 관계를 갖되, 쉬운 문제는 d\*가 작고(심지어 1) 깊이 더 가면 오히려 틀린다.
- **H2 (방법)**: hidden state로부터 d\*(x)를 예측하는 경량 head를, **학습 중 측정한 per-example argmax-정답률 깊이**로 supervise하면, 수렴/entropy 기반 halting보다 정확도-FLOPs Pareto에서 우월하다. 특히 **over-shoot 영역(정점 이후)에서 큰 이득**.
- **H3 (보정)**: 봉우리형 곡선에서는 단방향 conformal(LYNX/CALM류, "너무 일찍 멈추지 마")로 불충분하다. **양방향 risk control**(너무 일찍 멈춤 + 너무 늦게 멈춤 둘 다 bound)이 필요하며, 이는 새 calibration 절차를 요구한다.

#### 기존 최근접 이웃과의 차별화 (표절 방어)
| 최근접 이웃 | 그들이 한 것 | 우리가 다른 점 |
|---|---|---|
| **Ouro adaptive-loss** (2510.25741) | looping이 loss를 "줄이면 stop 페널티, 안 줄이면 overthink 페널티" | Ouro 신호 = **Δloss ≈ 0(plateau)**. 곡선이 **다시 올라가는(정답률 하락)** 영역을 정지점으로 잡지 못함. 또 **T=4로 캡**되어 봉우리 자체를 관측 못 함. 우리는 깊게 unroll해 곡선의 봉우리를 관측하고 **global argmax**로 supervise. |
| **LYNX / CALM** (conformal early-exit) | "premature exit 비율"을 단방향 bound, 깊을수록 안전 가정 | 봉우리형에선 **late exit(과사고)도 위험**. 우리는 **two-sided** conformal(d* 양쪽 모두 bound). 또 layer-exit/CoT-exit가 아니라 **loop-depth**. |
| **AdaPonderLM / PonderLM-3** | 난이도=NLL/confidence로 깊이 할당(자기참조, 단조 가정) | 자기참조 신호는 "수렴=정답"을 가정 → 봉우리 이후 degrade를 못 잡음. 우리는 **정답(ground-truth correctness)으로 직접 supervise**. |
| **Two-Scale acceleration exit** | 궤적이 안정되면 정지(수렴) | 수렴 ≠ 정답 최대. 안정된 fixed point가 틀릴 수 있음(2604.11791의 cyclic fixed point). |
| **Loop, Think & Generalize** (2604.07822) | overthinking을 **관찰**하고 entropy-augmented halting **휴리스틱** 제안 | 우리는 overthinking을 **정량 곡선으로 특성화 + argmax supervise + conformal 보정**(휴리스틱 아님). 그들의 관찰이 H1의 정황 근거. |

> 핵심 신규성 = **"정답률–깊이 비단조성(interior optimum)을 1급 시민으로 다루는 최초의 looped halting"**. 기존 전부 단조 가정. 표적 검색(2026-05)에서 "accuracy peaks then degrades as a central exploited phenomenon"은 확립되지 않음을 확인.

#### 방법 (구체)
1. **백본**: from-scratch가 이상적이나 비용↑. 1차 검증은 **mid-training**으로 충분 — 기존 dense 4B를 Retrofitted Recurrence(2511.07384) 방식 model surgery(prelude/recurrent core/coda)로 depth-recurrent화. core를 1~D_max(예: 32) loop. (이 surgery 자체는 신규 아님 — 우리 기여는 halting.)
2. **곡선 측정(H1)**: 학습/검증 문제 x마다 d=1..D_max 각각에서 답을 디코드, p(정답|d) 곡선과 argmax d\*(x) 기록. 곡선 모양(단조/봉우리/평탄) 통계를 난이도·도메인별로 집계 → **그 자체가 Contribution 1(실증)**.
3. **PeakDepth head**: coda 직전 hidden(또는 loop별 출력)으로 d\*(x)를 회귀/분류. 손실: (a) d\* 직접 회귀 + (b) loop별 "정점 도달 여부" 이진 분류(monotonic mask 아님 — 봉우리 양쪽 구분). explicit CoT 길이를 보조 prior로(있을 때).
4. **양방향 conformal 보정(H3)**: held-out calibration set에서, 사용자 지정 위험 α에 대해 "PeakDepth가 멈춘 깊이에서의 정답률이 d\* 정답률 대비 ε 이상 떨어지지 않을" 보장을 **early/late 양쪽** nonconformity score로 동시 제어(split conformal risk control 확장). LYNX의 단방향을 양방향으로 일반화하는 것이 방법론적 기여.
5. **(선택) latent→explicit 에스컬레이션**: 정점 정답률 자체가 임계 미만(=잠재 looping으로 안 풀림)이면 명시 CoT 디코딩으로 전환. **단, 이 부분은 SLT/SwiReasoning과 인접하므로 메인 기여로 내세우지 말고 보조 실험으로.**

#### 실험 설계
- **모델**: 4B(주), 1B(ablation 속도). Gemma4 토크나이저. D_max=32.
- **데이터**: 사용자 mid-training reasoning 풀(수학/코드/과학/논리). decontamination 유지(AIME24/25, MATH-500, GPQA-D 등 학습 제외).
- **평가 벤치마크**: GSM8K, MATH-500, AIME24/25, GPQA-Diamond, BBH(논리), HumanEval+/LiveCodeBench. **난이도 계층별**(easy/med/hard) 분해 필수.
- **베이스라인**: (i) 고정 깊이 d∈{1,4,8,16,32}, (ii) Huginn KL-exit, (iii) Two-Scale acceleration-exit, (iv) Ouro-style entropy-gate, (v) confidence-threshold, (vi) oracle d\*(상한).
- **주요 지표**:
  - **정확도-FLOPs Pareto**(가장 중요): 동일 평균 깊이에서 정확도, 동일 정확도에서 평균 깊이.
  - **Overthinking rate**: 모델이 d\* 이후로 looping해 정답→오답으로 뒤집힌 비율. PeakDepth가 이를 베이스라인 대비 낮추는가.
  - **Underthinking rate**: d\* 이전에 멈춰 놓친 비율.
  - **곡선 형태 통계**(H1): humped 비율, d\*와 난이도 상관, 도메인별 차이.
  - **conformal 보장 검증**(H3): 명목 α 대비 실제 위반율.
- **Ablation**: (a) supervise를 argmax-correctness vs loss-improvement(Ouro식)로 교체 시 over-shoot 영역 성능차, (b) 단방향 vs 양방향 conformal, (c) D_max를 4로 제한(Ouro 조건 재현) 시 봉우리 관측 가능 여부, (d) head 입력을 coda-hidden vs loop-trajectory.

#### 예상 기여 (논문 셀링 포인트)
- **C1(실증·놀라움)**: "latent depth에서 overthinking은 plateau가 아니라 내부 최적점이며, 정점을 넘으면 정답이 능동적으로 뒤집힌다"는 정량 증거(난이도·도메인별). — 단조 가정하는 기존 전체에 대한 반례.
- **C2(방법)**: 정답-supervised PeakDepth head + 양방향 conformal. Pareto 개선 + overthinking rate 감소.
- **C3(정렬)**: hybrid reasoning(easy→얕게, hard→정점에서 정지)을 단일 메커니즘으로.

#### 리스크와 대응
- **R1: 곡선이 사실 단조면?** → 그 경우 C1이 "looped LM의 depth-accuracy는 (조건 하에) 단조"라는 **음성 결과도 가치 있는 실증**. 사전 파일럿(1B, D_max=32)으로 humped 비율을 먼저 측정해 go/no-go. (Ouro가 T=4에서 정점 후 하락을 보고했으므로 신호는 있음.)
- **R2: argmax supervision이 noisy(곡선 평탄해 argmax 불안정)** → 정점 대신 "정답률이 정점의 (1−ε) 이상인 최소 깊이"로 robust target 정의.
- **R3: Ouro와의 차별성 심사위원 의심** → ablation (a),(c)로 "loss-improvement는 봉우리 못 잡음 + T≤4는 봉우리 못 봄"을 직접 시연.

---

### 4.2 대안 A — Loop–Token 교환비(Exchange Rate) + 공동 예산 컨트롤러

#### 아이디어
잠재 loop와 명시 토큰을 **두 종류의 "연산 화폐"**로 보고, **"1 explicit CoT token ≈ 몇 latent loop인가?"** 라는 교환비 r(난이도, 도메인)을 정량 측정한다. 그리고 from-scratch로 **두 화폐를 모두 유창하게 쓰고 문제별로 최적 혼합을 고르는** 모델을 학습한다(easy→0+0, med→loop만, hard→loop+token).

#### 차별화
- SwiReasoning/SLT/LT-Tuning/HybridCoT는 (a) 대부분 training-free거나 사후 적용, (b) 휴리스틱 스위치, (c) **교환비를 측정하거나 공동 예산을 최적 배분하지 않음.** 우리는 **정량 교환비 곡선 + Pareto-최적 공동 배분 정책**을 새 과학적 객체로 제시하고, **from pretraining**으로 둘에 모두 native한 모델을 만든다(사후 하이브리드와 구별).
- 이론적 훅: Saunshi "T loop ≈ T CoT step"은 **표현력** 등가만 말함. 우리는 **실제 학습된 모델에서의 정확도 등가 교환비**를 실측(이론 등가 ≠ 실측 등가가 핵심 발견 가능).

#### 실험
- 동일 백본에서 (loop=k, token=0), (loop=0, token=m), (loop=k, token=m) 격자로 정확도 측정 → iso-accuracy 곡선의 기울기 = 교환비.
- 컨트롤러: 총 FLOPs 예산 B 하에 (k,m) 배분을 예측하는 head, Pareto frontier로 평가.
- 위험: 측정 노이즈 큼, 백본이 둘 다 잘하도록 학습하는 것이 까다로움(from-scratch 비용). 메인보다 비용↑.

---

### 4.3 대안 B — 자기참조가 아닌 Robustness Halting ("수렴했지만 틀린" 문제)

#### 아이디어
기존 halting 신호(KL/entropy/confidence/fixed-point)는 전부 **자기참조적 수렴**이라 "확신에 차서 틀린" fixed point에서 멈춘다(2604.11791의 cyclic fixed point, 2604.04902의 underutilization이 정황). 대신 **잠재 궤적을 작게 섭동**(dropout/noise/입력재주입 변형)시킨 K개 궤적의 **답이 일치할 때** 정지한다(섭동-일관성). 필요한 K 자체가 난이도 신호.

#### 차별화
- self-consistency/DeepConf는 **명시 CoT 토큰**을 샘플링해 일치 검사. 우리는 **잠재 궤적 자체**를 섭동(토큰 미생성, 저비용). EBT의 energy-verify와도 다름(별도 energy head 불필요).
- 위험: K배 비용(쉬운 문제는 K=1로 충분하므로 평균 비용은 관리 가능하나 분석 필요), 섭동 설계가 민감.

---

### 4.4 (참고) 효율·시스템 트랙 — 추론 정렬은 약하지만 비어 있는 곳
사용자 목표(reasoning)와는 덜 맞지만, 진짜로 비어 있는 효율 공백(C4 에이전트 보고):
- **depth를 변화시키는 depth-recurrence scaling law**: loss(N_unique_params, R_loops, D_data) 동시 sweep. Sparse-Layers는 깊이 고정(μP가 깊이 전이 안 됨)이라 미완.
- **loop 반복 간 KV 공유 topology**(9-config taxonomy를 loop 축에 적용), KVSharer "dissimilar 공유"를 loop 축에서 검증.
이들은 별도 트랙으로, 추론 논문과 묶기보다 단독 효율 논문이 적합.

---

## 5. 착수 전 표절 회피 체크리스트

1. **arXiv 재검색**(착수 직전): "looped/recurrent-depth" × "non-monotonic depth accuracy / interior optimum / overthinking peak", "two-sided conformal early exit", "peak depth halting" 등. 이 분야는 월 단위로 논문이 나오므로 필수.
2. **Ouro(2510.25741) §adaptive-loss를 정독**하고, 우리 supervision이 그들의 Δloss 신호와 수식 수준에서 다른지 확정. (우리: global argmax-correctness; 그들: 인접 step Δloss.)
3. **LYNX(2512.05325)·CALM·SAFE-KD의 conformal 정식화**를 확인하고, two-sided 확장이 그들의 단방향과 정리(theorem) 수준에서 구별되는지 명시.
4. **RELAY(2502.08482)**: 우리는 loop↔CoT-step 정렬이 아니라 **halting 깊이 예측**임을 분명히.
5. **"Loop, Think & Generalize"(2604.07822)**를 related work에서 정직하게 인용 — 그들이 overthinking을 관찰했고, 우리가 그것을 정량화·예측·보정으로 확장함을 positioning.
6. closed-source distill 0% 정책 준수: 모든 곡선/supervision은 **자체 모델 forward**로 생성(teacher distill 불필요).

---

## 6. 권장 진행 순서

1. **파일럿(2~3주, 1B)**: 기존 1B를 depth-recurrent화(D_max=32) → 검증셋에서 p(정답)–깊이 곡선 측정. **H1(humped 비율) go/no-go.** ← 가장 중요. 여기서 봉우리가 충분히 나오면 메인 확정.
2. humped 확인 시: PeakDepth head + 양방향 conformal 구현, 4B로 확장, 베이스라인 대비 Pareto·overthinking rate.
3. humped 약하면: 대안 A(교환비) 또는 대안 B(robustness halting)로 피벗.

---

## 부록: 참고문헌 (arXiv)

핵심만 (전체 85편 중):
- ACT 1603.08983 · UT 1807.03819 · ALBERT 1909.11942 · DEQ 1909.01377 · Giannou 2301.13196
- Saturated-TC0 2106.16213 · Parallelism-Tradeoff 2207.00729 · CoT-serial 2402.12875 · Padding-TCd 2505.18948 · LogDepth 2503.03961 · Timestep-encoding 2410.01405
- von Oswald 2212.07677 · Yang-looped-ICL 2311.12424 · Gatmiry 2410.08292 · Fan-lengthgen 2409.15647
- Huginn 2502.05171 · Ouro 2510.25741 · Saunshi 2502.17416 · Coconut 2412.06769 · HRM 2506.21734 · TRM 2510.04871
- Decoding-Depth-Recurrent 2507.02199 · LatentCoT-stepwise 2602.00449 · Loop-Think-Generalize 2604.07822 · Two-Scale 2509.23314 · Mechanistic-looped 2604.11791 · Deep-Improvement-Supervision 2511.16886
- PonderNet 2107.05407 · MoD 2404.02258 · MoR 2507.10524 · CALM 2207.07061 · AdaPonderLM 2603.01914 · PonderLM-3 2603.02023 · Rational-Metareasoning 2410.05563 · LYNX 2512.05325 · SAFE-KD 2602.03043
- RecurrentGemma 2404.07839 · Griffin 2402.19427 · Relaxed-Recursive 2410.20672 · Retrofitted-Recurrence 2511.07384 · Sparse-Layers 2605.09165 · MoEUT 2405.16039 · PLT 2510.24824 · Hyperloop 2604.21254 · SpiralFormer 2602.11698 · Attractor/Solve-the-Loop 2605.12466 · Universal-YOCO 2604.01220
- YOCO 2405.05254 · KV-9config 2410.14442 · MiniCache 2405.14366 · KVSharer 2410.18517 · xKV 2503.18893
- DoT 2402.07754 · DCoLT 2505.10446 · EBT 2507.02092 · Dualformer 2410.09918 · CogRouter 2602.12662 · SwiReasoning 2510.05069 · LT-Tuning 2602.10229 · SLT 2605.25745 · TTT 2407.04620 · Pause-token 2310.02226 · Quiet-STaR 2403.09629 · Soft-Thinking 2505.15778 · Difficulty-probe 2511.03808 · CODA 2603.08659 · FR-Ponder 2509.24238 · Learning-When-to-Stop 2511.21581 · LRM-interpretable 2604.04902 · Demystify-Hybrid 2510.12680 · Thinking-in-Latents 2603.15051
- 서베이: Latent-Reasoning-Survey 2507.06203 · Latent-CoT-Survey 2505.16782 · Stop-Overthinking 2503.16419 · Reasoning-on-a-Budget 2507.02076

> 검증 주의: 2026년 preprint(2602.\*, 2603.\*, 2604.\*, 2605.\* 등)는 제목·초록·arXiv ID를 검색으로 확인했으나 일부는 전문 미정독. 인용 전 원문 재확인 권장. 일부 2026 논문은 동명이인 주의(LoopFormer, Attractor, CODA가 각각 둘씩 존재).
