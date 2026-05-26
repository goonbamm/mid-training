# 하이브리드 추론 LLM을 위한 적응적 추론 길이 제어 연구 보고서

> **목표**: 추론이 필요한 태스크에는 길게, 필요 없는 태스크에는 짧게(혹은 전혀 안 하게) 추론하는 LLM을 pre-training 단계부터 설계하기.
> **조사 범위**: 95개 사례 (논문/기술보고서/모델 릴리즈, 2023–2026-05). 학습 단계(pre-training / mid-training / SFT / RL)와 제어 메커니즘(reward / data / routing / inference-time) 양 축으로 분류. **§11.4 한국 LLM 5건 + §11.5 2026년 1-5월 신작 19건이 추가 검증 단계에서 보완됨.**
> **인용 원칙**: 모든 핵심 주장에는 해당 논문의 원문을 직접 인용(verbatim, 영문)으로 첨부. 추가 검증으로 모든 arXiv ID 실재 확인.

---

## 0. 핵심 요약 (TL;DR)

1. **현 SOTA 합의**: "동일한 단일 모델이 두 모드(thinking / non-thinking)를 모두 지원하는 hybrid reasoning 모델"이 산업 표준이 되었다 — Qwen3, Claude 3.7 Sonnet, DeepSeek-V3.1, GLM-4.5, Nemotron Nano 2, SmolLM3 모두 이 패러다임.
2. **제어 입력**: 사용자 측에서는 (a) chat-template 플래그(`/think`, `/no_think`), (b) `thinking_budget` 토큰 수, (c) `reasoning_effort` 레벨(minimal/low/medium/high) 세 가지 방식이 자리잡았다.
3. **학습 측면**:
   - **Pre-training**에서 직접 길이 제어를 학습시키는 사례는 드물지만, **Pause Token(2310.02226)** 과 **Quiet-STaR(2403.09629)** 가 prototype.
   - **Mid-training**(continued pretraining)이 2025년에 명확한 별도 단계로 자리잡았고, 이 단계가 **추론 능력의 ceiling**, **RL 안정성**, **추론 길이 dynamics**를 결정한다는 증거가 다수 (OctoThinker, Interplay paper, Front-Loading Reasoning, Thinking Augmented Pre-training, RA3, Survey on Mid-training).
   - **Post-training (SFT + RL)** 에서 hybrid 모드를 implant 하는 게 표준. Qwen3는 별도의 **"Thinking Mode Fusion"** stage를 두었다.
4. **RL reward 패턴**: (i) **정답일 때만 길이 penalty 적용**(Kimi k1.5, Short-RL), (ii) **난이도 적응 penalty**(DAST, ALP, Laser-D), (iii) **soft / cosine 길이 reward**(Demystifying Long CoT, Magistral), (iv) **target-length 명시**(L1/LCPO, e1), (v) **non-thinking을 선호하되 thinking 정확도 유지**의 제약 최적화(AdaptThink, AdaCoT).
5. **사용자(당신) 시나리오에 대한 권고** (§13에서 상세):
   - **Pre-train**: 길이 제어와 직접 연관하지 말되, 추론-친화적 데이터(수학/코드/proof) 비중을 마지막 25% 토큰에서 점진적으로 늘려라.
   - **Mid-train**: OctoThinker식 *Stable-then-Decay* 2-stage + short/long/hybrid 3-branch 데이터 mixture로 **추론 모드 토글 ability를 mid-training에서 implant**. SmolLM3, Quiet-STaR, Thinking Augmented Pre-training, RA3가 직접 근거.
   - **SFT**: long-CoT 데이터와 short-CoT(혹은 빈 think 블록) 데이터를 mix해 chat template로 `/think`/`/no_think`를 학습.
   - **RL**: GRPO + **난이도 적응 길이 penalty** (Laser-D / ALP / DAST 중 택일). DAPO 식 overlong soft-punishment를 합쳐 reward hacking 방지.

---

## 1. 서론 — 왜 "추론 길이"가 문제인가

R1, o1, Qwen QwQ 등 long-CoT reasoning 모델은 모든 입력에 수천~수만 토큰의 추론을 강제하기 때문에:

- **단순 질문에 overthinking**이 발생해 latency·비용이 폭증.
- 어떤 경우 길게 생각할수록 오히려 *정답률이 떨어짐*. CAR 논문(2505.15154)은 *"excessive reliance on chain-of-thought (CoT) reasoning can impair model performance and brings unnecessarily lengthened outputs"* 라고 명시.
- Surveys (2507.02076; 2508.02120; 2507.09662) 모두 *"models apply fixed inference-time compute regardless of task complexity, often overthinking simple problems while underthinking hard ones."* (2507.02076) 라고 진단.

사용자의 목표("어려운 건 길게, 쉬운 건 짧게")는 학계 용어로 **adaptive reasoning length control** = **L2-adaptiveness** (2507.02076)다. 본 보고서는 이 목표를 달성한 64편 논문을 7개 카테고리로 정리한다.

---

## 2. 분류 체계 (Taxonomy)

| 축 | 값 |
|---|---|
| **학습 단계** | Pre-train / Mid-train / SFT / Preference (DPO·APO) / RL / Inference-time |
| **제어 입력** | User flag (`/think`) / Token budget / Effort level / Internal classifier / Self-judgement |
| **제어 메커니즘** | Data mixture / Reward shaping / LoRA direction / Routing / Early-exit |

이 두 축으로 보면 각 논문은 명확히 자리잡는다. 예) Qwen3 = mid-train + post-train SFT+RL / user-flag / data mixture+reward.

---

## 3. Section A — Production Frontier 모델의 Hybrid Thinking

### 3.1 Qwen3 (Alibaba, 2025-05) — arXiv 2505.09388
- **메커니즘**: 단일 모델, `/think` & `/no_think` chat-template 플래그 + `thinking_budget` 토큰 상한.
- **학습 stage**: post-training의 **stage 3 "Thinking Mode Fusion"**.

> "Qwen3 integrates two distinct operating modes, thinking mode and non-thinking mode, into a single model." (Abstract)

> "The goal of the Thinking Mode Fusion stage is to integrate the 'non-thinking' capabilities into the previously developed 'thinking' model." (§4.3)

> "Qwen3 incorporates thinking budgets, providing users with fine-grained control over the level of reasoning effort." (Introduction)

> "When the length of the model's thinking reaches a user-defined threshold, we manually halt the thinking process." (§4.3 Thinking Budget)

> "We design chat templates to fuse the two modes" using "/think and /no_think flags in the user query." (§4.3 Chat Template Design)

**해석**: Qwen3의 hybrid 능력은 *thinking-only* 모델을 먼저 만든 뒤(stage 1,2), stage 3에서 short-CoT/no-think 데이터로 *fuse* 하는 형태. 즉 mid-train이 아니라 post-train의 추가 SFT stage. 다만 pre-training은 "general → reasoning phase → long-context phase"의 3-stage로 reasoning data 비중 증가가 명시되어 있어 *암묵적 mid-training* 으로 볼 수 있다.

### 3.2 Claude 3.7 Sonnet (Anthropic, 2025-02)
> "Claude 3.7 Sonnet, our most intelligent model to date and the first hybrid reasoning model on the market."
> "Just as humans use a single brain for both quick responses and deep reflection, reasoning should be an integrated capability of frontier models rather than a separate model entirely."
> "API users also have fine-grained control over how long the model can think for."
> "When using Claude 3.7 Sonnet through the API, users can also control the budget for thinking: you can tell Claude to think for no more than N tokens."

**해석**: Anthropic은 명시적 dual-mode를 *철학적 결정*으로 채택. API에서 `thinking.budget_tokens` 파라미터 (최소 1024, target이며 strict cap 아님).

### 3.3 DeepSeek-V3.1 (DeepSeek, 2025-08) — HF model card
> "DeepSeek-V3.1 is a hybrid model that supports both thinking mode and non-thinking mode."
> "Hybrid thinking mode: One model supports both thinking mode and non-thinking mode by changing the chat template."
> "DeepSeek-V3.1 is post-trained on the top of DeepSeek-V3.1-Base, which is built upon the original V3 base checkpoint through a two-phase long context extension approach."
> "The 32K extension phase has been increased 10-fold to 630B tokens, while the 128K extension phase has been extended by 3.3x to 209B tokens."

**해석**: V3.1은 V3 base에 **long-context continued pretraining = mid-training**을 거대화한 뒤, post-train에서 hybrid를 implant.

### 3.4 GLM-4.5 (Zhipu AI, 2025-08) — arXiv 2508.06471
> "Both GLM-4.5 and GLM-4.5-Air are hybrid reasoning models that provide two modes: thinking mode for complex reasoning and tool usage, and non-thinking mode for immediate responses." (README)
> "The model supports per-turn control over reasoning within a session—disable thinking for lightweight requests to reduce latency/cost, enable it for complex tasks to improve accuracy and stability." (README)

### 3.5 NVIDIA Nemotron Nano 2 / Nemotron-Nano-9B-v2 (NVIDIA, 2025-08) — arXiv 2508.14444
- **하이브리드 아키텍처**: 대부분의 self-attention을 Mamba-2로 대체한 hybrid Mamba-Transformer.
- **추론 토글**: control token `/think`, `/no_think` + thinking-budget 관리.

> "Nemotron Nano V2 allows users to specify how many thinking tokens the model may generate before producing the final answer."
> "Once the budget is reached, the inference setup attempts to insert a closing `</think>` tag."
> "About 5% of the data contained deliberately truncated reasoning traces, enabling fine-grained thinking budget control at inference time."
> "Stage 1 uses the full dataset described in Section 3.1, augmented with a subsample of roughly 10% of prompts paired with outputs stripped of reasoning traces."
> "Nemotron Nano 2 achieves comparable or better accuracies at up to 6× higher throughput than Qwen3-8B."

**해석**: ★ 매우 중요한 레시피 — *SFT 데이터의 10%* 를 think 블록 없는 답으로 두고, *5%* 를 일부러 truncated 시켜 budget 학습에 활용. **하드 토글 + soft budget**을 한 모델에 동시에 implant.

### 3.6 Magistral (Mistral, 2025-06) — arXiv 2506.10910
> "We use Group Relative Policy Optimization (GRPO) as our RL algorithm."
> "We remove the KL penalty entirely" because "maintaining a copy of the reference model for KL computation incurs a compute cost we find unjustified."
> "Following Yu et al. [2025], we use soft length penalty to signal the model that the hard cutoff on maximal completion length is near."
> "We observe a ubiquitous log scaling" between reward and output length.
> "Training was done in multiple stages with distinct hyper-parameters." (lmax: 16k→24k→32k)

**해석**: Magistral은 *pure-RL* 노선(DeepSeek-R1-Zero와 같이) + iterative context extension. *soft* length penalty는 hard cutoff 근처에서만 작동.

### 3.7 SmolLM3 (HuggingFace, 2025) — model card + blog
- **단계 명명**이 매우 분명: pre-train → **mid-train(=long-context+reasoning adaptation)** → SFT → APO.

> "We call the long context adaptation and reasoning adaptation 'mid-training'." (★)
> "After the main pretraining, we trained SmolLM3 on an additional 100B tokens to extend its context length."
> "Our mid-training dataset contained 35B tokens sourced from Open Thought's OpenThoughts3-1.2M and a subset from NVIDIA's Llama-Nemotron-Post-Training-Dataset-v1.1 with reasoning traces from R1."
> "SmolLM3's chat template allows users to control the reasoning mode during a conversation."
> "Users can activate reasoning or non-reasoning modes by including the `/think` and `/no_think` flags, respectively, in the system prompt."
> "In non-reasoning mode, we pre-fill the model's response with empty think blocks, similar to Qwen3, to ensure direct answers without explicit reasoning."

**해석**: ★★★ 사용자의 시나리오와 가장 밀접. **추론 능력 자체를 mid-training으로 implant** 한 사례. Reasoning-data 35B 토큰을 4 epoch (≈140B) 학습한 checkpoint를 SFT 입력으로 사용.

### 3.8 GPT-5 (OpenAI) — cookbook + docs
> "If not specified, effort defaults to medium."
> "Set reasoning effort: 'minimal'. Avoid for multi-step planning or tool-heavy workflows."
> "Runs GPT-5 with few or no reasoning tokens to minimize latency and speed time-to-first-token."
> "Use it for deterministic, lightweight tasks (extraction, formatting, short rewrites, simple classification) where explanations aren't needed."
> "The verbosity parameter lets you hint the model to be more or less expansive in its replies."

→ `reasoning_effort ∈ {minimal, low, medium, high, xhigh}` + `verbosity ∈ {low, medium, high}`. Effort = thinking depth, verbosity = output length. **두 축이 분리**되어 있다는 점이 설계 포인트.

### 3.9 Gemini 2.5 (Google, 2025) — official docs
> "The `thinkingBudget` parameter, introduced with the Gemini 2.5 series, guides the model on the specific number of thinking tokens to use for reasoning."
> "You can disable thinking by setting `thinkingBudget` to 0."
> "Setting the `thinkingBudget` to -1 turns on dynamic thinking, meaning the model will adjust the budget based on the complexity of the request."
> "Gemini models engage in dynamic thinking by default, automatically adjusting the amount of reasoning effort based on the complexity of the user's request."

→ -1 / 0 / N 의 3가지 모드. *dynamic thinking* 이 default라는 점이 사용자 시나리오와 가장 가깝다.

---

## 4. Section B — RL 기반 길이 제어 (가장 활발한 분야)

### 4.1 L1 / LCPO — arXiv 2503.04697 (Aggarwal & Welleck, 2025)
> "We introduce Length Controlled Policy Optimization (LCPO), a simple reinforcement learning method that optimizes for accuracy and adherence to user-specified length constraints."
> "L1's length control allows for smoothly trading off computational cost and accuracy on a wide range of tasks, and outperforms the state-of-the-art S1 method for length control."
> "Furthermore, we uncover an unexpected short chain-of-thought capability in models trained with LCPO. Specifically, using LCPO we derive Short Reasoning Models (SRMs), that exhibit similar reasoning patterns as full-length reasoning models, but can generate CoT lengths comparable to non-reasoning models."
> "Our 1.5B L1 model surpasses GPT-4o at equal reasoning lengths."

핵심 reward: `r(y; n_gold) = I(ŷ = y_gold) - α · |n_gold - n_y|` — **사용자가 프롬프트에 길이 지정**.

### 4.2 Kimi k1.5 — arXiv 2501.12599
> "We promote shorter responses and penalize longer responses among correct ones, while explicitly penalizing long responses with incorrect answers."
> Reward: "If r(x,y_i,y*)=1" then "λ = 0.5 − (len(i)−min_len)/(max_len−min_len)" is applied.
> "It is possible to transfer the thinking priors from long-CoT models to short-CoT models so that performance can be improved even with limited test-time token budgets."
> "We select the shortest correct response for supervised fine-tuning" through rejection sampling methodology.
> "The proposed long2short RL algorithm demonstrates the highest token efficiency compared other methods such as DPO and model merge."

**해석**: ★ 산업적 검증을 거친 가장 안정적인 reward 패턴 — *"correct에 한해 짧을수록 가산, incorrect는 길이와 무관하게 페널티"*. "long2short" 라는 용어를 만든 원조.

### 4.3 O1-Pruner — arXiv 2501.12570
> "Defining the Length-Harmonizing Reward R_LH(x,y) = L̄_ref(x)/L(y) − 1 + λ(A(x,y) − Ā_ref(x))"
> "L̄_ref(x) = 1/K ∑_{i=1}^K L(y'_i), y'_i ∼ π_ref(·|x)"
> "we adopt an off-policy training approach by directly sampling from the π_ref instead of π_θ."
> "We use a PPO-style loss to optimize the objective function, which helps for our off-policy training strategy."
> "we aim to ensure that the model's accuracy does not decrease, or even improves, during the process of optimizing for length."

**해석**: reference 모델의 길이를 기준으로 *상대적 단축* + 정확도 보전. Off-policy로 안정.

### 4.4 DAST — arXiv 2503.04472
> "We first propose a Token Length Budget (TLB) metric to quantify difficulty"
> "enables models to autonomously adjust the length of Chain-of-Thought (CoT) based on problem difficulty"
> "leverage budget-aware reward shaping and budget preference optimization to implement DAST."
> "DAST penalizes overlong responses for simple tasks while incentivizing sufficient reasoning for complex problems"
> "reducing token usage by over 30% on average"

**해석**: **난이도-적응 reward의 시초**. TLB가 곧 사용자의 시나리오를 가장 직접적으로 구현.

### 4.5 ThinkPrune — arXiv 2504.01296
> "ThinkPrune offers a simple solution that continuously trains the long-thinking LLMs via reinforcement learning (RL) with an added token limit"
> "beyond which any unfinished thoughts and answers will be discarded, resulting in a zero reward"
> "we introduce an iterative length pruning approach, where multiple rounds of RL are conducted, each with an increasingly more stringent token limit"
> "the reasoning length of DeepSeek-R1-Distill-Qwen-1.5B can be reduced by half with only 2% drop in performance"

### 4.6 Demystifying Long CoT — arXiv 2502.03373 (Yeo et al.)
> "Cosine Reward varies with generation length" with three constraints: correct > wrong; "shorter correct CoTs receive higher rewards than longer ones"; "shorter wrong CoTs should receive higher penalties than longer wrong CoTs."
> "Both models increased their CoT length during training, eventually reaching the context window limit. This led to a decline in training accuracy due to CoTs exceeding the allowable window size."
> "CoT length does not always scale up in a stable fashion" without explicit reward shaping.
> "Verifiable reward signals like ones based on ground-truth answers are essential for stabilizing long CoT RL for reasoning tasks."
> "The repetition penalty resulted in better downstream task performance and also shorter CoTs."
> "SFT with long CoT can scale up to a higher performance upper limit than short CoT."

**해석**: 길이 reward 설계의 *교과서*. **Cosine reward + repetition penalty**의 조합이 stability를 보장.

### 4.7 DAPO (ByteDance, 2025-03) — arXiv 2503.14476
> "By default, we assign a punitive reward to truncated samples. This approach may introduce noise into the training process, as a sound reasoning process can be penalized solely due to its excessive length."
> Soft punishment: "within this interval, the longer the response, the greater the punishment it receives."
> Token-level loss: "tokens within longer responses (which contain more tokens) may have a disproportionately lower contribution to the overall loss."
> Dynamic Sampling: "we propose to over-sample and filter out prompts with the accuracy equal to 1 and 0."
> Clip-Higher: "we decouple the lower and higher clipping range as ε_low and ε_high."

**해석**: 대규모 RL의 안정성 trick 모음. **"Overlong reward shaping"** = soft punishment가 핵심 기여.

### 4.8 DeepSeek-R1 — arXiv 2501.12948
> "We do not apply the outcome or process neural reward model." (accuracy + format rewards only)
> "During this phase, DeepSeek-R1-Zero learns to allocate more thinking time to a problem by reevaluating its initial approach."
> "Behaviors such as reflection—where the model revisits and reevaluates its previous steps—and the exploration of alternative approaches to problem-solving arise spontaneously."
> "The thinking time of DeepSeek-R1-Zero shows consistent improvement throughout the training process."
> "DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step"
> "To mitigate the issue of language mixing, we introduce a language consistency reward during RL training, which is calculated as the proportion of target language words in the CoT."

**해석**: R1은 직접 길이를 제어하지 않는다 — 길이는 *emergent* 부산물. 하지만 R1-Zero는 *너무 길어지는 경향*을 보인다는 점이 후속 length-control 연구의 동기.

### 4.9 AdaCoT — arXiv 2505.11896
> "AdaCoT framed adaptive reasoning as a Pareto optimization problem that seeks to balance model performance with the costs associated with CoT invocation"
> "We propose a reinforcement learning (RL) based method, specifically utilizing Proximal Policy Optimization (PPO), to dynamically control the CoT triggering decision boundary"
> "CoT necessity based on implicit query complexity"
> "AdaCoT reduced CoT triggering rates to as low as 3.18% and decreased average response tokens by 69.06%, while maintaining high performance on complex tasks"
> "A key technical contribution is Selective Loss Masking (SLM), designed to counteract decision boundary collapse during multi-stage RL training, ensuring robust and stable adaptive triggering"

**해석**: ★★★ 사용자가 원하는 그대로 — "쉬운 건 CoT 자체를 건너뛰는 비율을 96.82%" — 단 chat 인터페이스에서. 직접 적용 가능.

### 4.10 Ada-R1 — arXiv 2504.21659
> "First, we construct a hybrid reasoning model by merging long and short CoT models to enable diverse reasoning styles."
> "Second, we apply bi-level preference training to guide the model to select suitable reasoning styles (group-level), and prefer concise and correct reasoning within each style group (instance-level)."
> "This motivates adaptive reasoning strategies that tailor reasoning depth to the input."
> "Notably, on five mathematical datasets, the average length of reasoning is reduced by more than 50%."

### 4.11 AdaptThink — arXiv 2505.13417
> "AdaptThink, a novel RL algorithm to teach reasoning models to choose the optimal thinking mode adaptively based on problem difficulty."
> "An importance sampling strategy that balances Thinking and NoThinking samples during on-policy training."
> "A constrained optimization objective that encourages the model to choose NoThinking while maintaining the overall performance."
> "AdaptThink reduces the average response length of DeepSeek-R1-Distill-Qwen-1.5B by 53% and improves its accuracy by 2.4%."

**해석**: ★ "기본은 NoThinking을 선호하되 정확도가 떨어지지 않는 범위에서만"이라는 제약 최적화가 핵심.

### 4.12 AutoThink — arXiv 2505.10832
*정식 제목: "Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL". 'AutoThink'은 본 보고서가 부여한 약칭.*

> "Inserting a simple ellipsis ("...") into the prompt can stochastically trigger either a thinking or no-thinking mode, revealing a latent controllability in the reasoning behavior."
> "AutoThink, a multi-stage reinforcement learning (RL) framework that progressively optimizes reasoning policies via stage-wise reward shaping."
> "AutoThink learns to invoke explicit reasoning only when necessary, while defaulting to succinct responses for simpler tasks."
> "AutoThink improves relative accuracy by 6.4 percent while reducing token usage by 52 percent on DeepSeek-R1-Distill-Qwen-1.5B."

### 4.13 LASER (Laser-D/DE) — arXiv 2505.15612
> "Laser adopts a step reward function guided by a target length, rather than performing hard truncation."
> Reward term: "α · I(L(y) ≤ L_T)" with "step reward function"
> Laser-D: "decouples the target length hyperparameter across different queries, allowing distinct target lengths to be assigned to various queries. Easier questions receive smaller target lengths (i.e. smaller scaling factor β), while harder questions receive larger ones."
> "We leverage rule-based reward designed as a simple scoring system: +1 for correct responses, -0.5 for incorrect responses, and -1 for responses with invalid format."
> "Applying Laser-D/Laser-DE to DeepSeek-R1-Distill-Qwen-1.5B improves accuracy by +6.1 percentage points while reducing token usage by 63% on AIME24."

**해석**: ★ 난이도 적응 reward의 가장 명확한 구현. **target length를 query-별로 다르게**.

### 4.14 ALP / Just Enough Thinking — arXiv 2506.05256
> "ALP monitors each prompt's online solve rate through multiple rollouts and adds a differentiable penalty whose magnitude scales inversely with that rate, so confident (easy) prompts incur a high cost for extra tokens while hard prompts remain unhindered."
> "Posttraining DeepScaleR-1.5B with ALP cuts average token usage by 50% without significantly dropping performance."

**해석**: solve-rate(=난이도)에 *반비례*하는 penalty.

### 4.15 ASRR — arXiv 2505.15400
*정식 제목: "When to Continue Thinking: Adaptive Thinking Mode Switching". 'ASRR'은 본문 내 방법명(Adaptive Self-Recovery Reasoning) 약자로, paper 제목에는 등장하지 않음.*

> "ASRR adaptively allocates reasoning effort according to problem difficulty"
> "Adaptive Self-Recovery Reasoning (ASRR), a framework that suppresses unnecessary reasoning and enables implicit recovery"
> "By introducing accuracy-aware length reward regulation, ASRR adaptively allocates reasoning effort"
> "reduces reasoning budget by up to 32.5% (1.5B) and 25.7% (7B) with minimal accuracy loss (1.2% and 0.6% pass@1)"

### 4.16 Short-RL (Length-Aware Optimization) — arXiv 2505.12284
*정식 제목: "Shorten After You're Right: Lazy Length Penalties for Reasoning RL" (Yuan et al., MSRA). 'Short-RL'은 본 보고서가 부여한 약칭.*

> "A straightforward approach to reducing reasoning length is to incorporate a length penalty into the original reward function."
> "the length reward is applied only to correct responses. Specifically, the length reward is computed exclusively for correct answers."
> "we propose three simple yet effective reward designs to reduce the average step-wise response length during RL training while preserving or enhancing model performance."
> "in a logic reasoning setting, we achieve a 40% reduction in response length averaged by steps alongside a 14% gain in performance."

### 4.17 SelfBudgeter — arXiv 2505.11274
> "We first train the model to self-estimate the required reasoning budget based on the query."
> "We then introduce budget-guided GRPO for reinforcement learning, which effectively maintains accuracy while reducing output length."
> "SelfBudgeter dynamically allocates budgets according to problem complexity"
> "achieving an average response length compression of 61% on math reasoning tasks while maintaining accuracy"
> "users can directly control the reasoning length by setting token budgets upfront"

**해석**: ★ 모델이 *스스로* 토큰 예산을 출력하는 패턴. UX 측면에서도 깔끔.

### 4.18 e1 / Adaptive Effort Control — arXiv 2510.27042
> "Users can dynamically adjust the cost-accuracy trade-off through a continuous effort parameter specified at inference time."
> "Adaptive Effort Control, a self-adaptive reinforcement learning method that trains models to use a user-specified fraction of tokens relative to the current average chain-of-thought length for each query."
> "The model automatically learns to allocate resources proportionally to the task difficulty and, across model scales ranging from 1.5B to 32B parameters, our approach enables a 2-3x reduction in chain-of-thought length while maintaining or improving performance relative to the base model used for RL training."

### 4.19 Thinking Fast and Right — arXiv 2505.18298
> "Our method dynamically adjusts the reward trade-off between accuracy and response length based on model performance"
> "when accuracy is high, the length penalty increases to encourage faster length reduction; when accuracy drops, the penalty is relaxed to preserve correctness"

→ **dynamic reward weight** 패턴.

### 4.20 MRT (Meta-RL Fine-Tuning) — arXiv 2503.07572
> "we formalize the problem of optimizing test-time compute as a meta-RL problem"
> "progress made by 𝐳ⱼ as r_prog^μ(𝐳ⱼ;𝐜) := J_r(μ(·|𝐳ⱼ,𝐜)) − J_r(μ(·|𝐜))" — progress reward.
> Cumulative regret minimization: training aims to minimize the gap by ensuring "each subsequent episode should ideally increase the probability" of success.
> The approach trains "budget-agnostic" models that work across token budgets.

### 4.21 HiPO — arXiv 2509.23967
> "enables LLMs to selectively decide when to engage in detailed reasoning (Think-on) and when to respond directly (Think-off)"
> "HiPO combines a hybrid data pipeline providing paired Think-on and Think-off responses with a hybrid reinforcement learning reward system that balances accuracy and efficiency"
> "HiPO can substantially reduce token length while maintaining or improving accuracy"

### 4.22 ADR (Adaptive Dual Reasoner) — arXiv 2510.10207
> "ADR supports two reasoning modes: fast thinking and slow thinking. ADR dynamically alternates between these modes based on the contextual complexity during reasoning."
> "Entropy-guided Hybrid Policy Optimization EHPO, an RL training framework employing an entropy-guided dynamic rollout strategy for branching at high-entropy units."
> "a difficulty-aware penalty to balance fast and slow reasoning."
> "ADR yields a performance gain of up to 6.1%, while reducing the reasoning output length by 49.5% to 59.3%."

### 4.23 ODIN (Disentangled length-reward) — arXiv 2402.07319
> "A well-formatted, verbose but less helpful response from the LLMs can often deceive LLMs or even human evaluators to achieve high scores."
> "We further propose to improve the reward model by jointly training two linear heads on shared feature representations to predict the rewards, one trained to correlate with length, and the other trained to decorrelate with length and therefore focus more on the actual content."
> "our approach almost eliminates the reward correlation with length"

**해석**: ★ length reward-hacking을 정면으로 다룬 선구작.

### 4.24 Light-R1 — arXiv 2503.10460
> "Our curriculum training progressively increases data difficulty, combined with multi-staged post-training."
> "Our Light-R1-32B model, trained from Qwen2.5-32B-Instruct, outperforms DeepSeek-R1-Distill-Qwen-32B in math reasoning."

### 4.25 LIMR — arXiv 2502.11886
> "a strategically selected subset of just 1,389 samples can outperform the full 8,523-sample dataset"
> "Learning Impact Measurement (LIM), an automated method to evaluate and prioritize training samples based on their alignment with model learning trajectories"
> "our RL-based LIMR achieves 16.7% higher accuracy on AIME24 and outperforms LIMO and s1 by 13.0% and 22.2% on MATH500"

### 4.26 ProRL (NVIDIA, 2025-05) — arXiv 2505.24864 (summary fetched)
- Prolonged RL은 *novel reasoning strategies* 를 base model에서는 sampling으로 찾을 수 없는 영역까지 확장한다 — KL control + reference reset 으로 안정화.

---

## 5. Section C — SFT / Data-based 길이 제어

### 5.1 CoT-Valve — arXiv 2502.09601 (LoRA로 길이 dial)
> "by manipulating it, acts as increasing or decreasing the length of CoT. Taking a large step in this direction leads the model to generate a short sequence, while a small step still produces a long and complex reasoning trajectory."
> "Δθ denotes the change in the parameter space that steers the model towards generating a more concise chain."
> "MixChain, a dataset inherently generated by our method that contains reasoning paths of varying lengths. This dataset is structured such that each question is associated with multiple reasoning paths, with lengths progressively decreasing from long to short."
> "we choose to incorporate this update direction by LoRA, enabling it to function as an additional branch that facilitates easy modulation of intensity while imposing minimal extra parameters on the model."
> "reducing reasoning chains on GSM8K from 741 to 225 tokens with a minor performance drop (95.07% to 94.92%) and on AIME from 6827 to 4629 tokens."
> "the model is trained with a shorter reasoning path from the dataset at each iteration, rather than training directly with the shortest reasoning CoT. This gradual compression method allows the model to progressively reduce the length of its reasoning paths."

**해석**: ★★ "길이 = LoRA direction의 scalar"라는 가장 우아한 접근. **Inference-time에 α 하나로 길이 조절**.

### 5.2 C3oT — arXiv 2412.11664
> "a compressor to compress an original longer CoT into a shorter CoT while maintaining key information and interpretability"
> "a conditioned training method to train LLMs with both longer CoT and shorter CoT simultaneously to learn the corresponding relationships between them"
> "a conditioned inference method to gain the reasoning ability learned from longer CoT by generating shorter CoT"
> "the proposed method is capable of compressing the length of generated CoT by up to more than 50% without compromising its effectiveness"

### 5.3 LIMO — arXiv 2502.03387
> "Synthesizing these findings, we propose the Less-Is-More Reasoning Hypothesis (LIMO Hypothesis): In foundation models where domain knowledge has been comprehensively encoded during pre-training, sophisticated reasoning can emerge through minimal but strategically designed demonstrations of cognitive processes."
> "LIMO achieves 63.3% accuracy on AIME24 and 95.6% on MATH500, surpassing previous fine-tuned models...while using only 1% of the training data required by prior approaches."
> "The threshold for eliciting complex reasoning is not dictated by task complexity but rather by two key factors: (1) the completeness of the model's pre-trained knowledge base..."

**해석**: 함의가 큰 가설. *Pre-train(혹은 mid-train) 단계에서 충분한 도메인 지식만 갖추면 SFT는 적은 양으로도 충분하다*. → 사용자에게 직접 적용 가능한 메시지.

### 5.4 s1: Simple Test-Time Scaling — arXiv 2501.19393
> "We develop budget forcing to control test-time compute by forcefully terminating the model's thinking process or lengthening it by appending 'Wait' multiple times to the model's generation when it tries to end."
> "We curate a small dataset s1K of 1,000 questions paired with reasoning traces relying on three criteria we validate through ablations: difficulty, diversity, and quality."
> "After supervised finetuning the Qwen2.5-32B-Instruct language model on s1K and equipping it with budget forcing, our model s1-32B exceeds o1-preview on competition math questions by up to 27% (MATH and AIME24)."
> "scaling s1-32B with budget forcing allows extrapolating beyond its performance without test-time intervention: from 50% to 57% on AIME24."

### 5.5 Train Long, Think Short — arXiv 2508.08940
> "Starting from an initial budget B0 and progressively tighten it via an exponential decay schedule" with budget at step t: "max(1, B_0·γ^⌊t/T⌋)"
> "During training, the model can explore a long chain-of-thought to discover effective reasoning patterns; as the budget shrinks, it is forced to distill these patterns into more concise and efficient reasoning traces."
> "Curriculum learning yields consistent gains in both accuracy and token efficiency at the same final budget."
> "On MATH500: curriculum learning raises accuracy from 38.80% (fixed-budget) to 43.40% while compressing average reasoning length from 179.3 to 137.1 tokens."

---

## 6. Section D — Prompting / Inference-time 압축

### 6.1 Chain of Draft (CoD) — arXiv 2502.18600
> "LLMs generate minimalistic yet informative intermediate reasoning outputs while solving tasks."
> "CoD matches or surpasses CoT in accuracy while using as little as only 7.6% of the tokens, significantly reducing cost and latency."
> "A novel paradigm inspired by human cognitive processes, where LLMs generate minimalistic yet informative intermediate reasoning outputs."
> "humans typically employ a more efficient strategy: drafting concise intermediate thoughts that capture only essential information."

### 6.2 Concise CoT (Renze 2024) — arXiv 2401.05618
> "In this paper, we introduce Concise Chain-of-Thought (CCoT) prompting."
> "CCoT reduced average response length by 48.70% for both GPT-3.5 and GPT-4."
> "CCoT leads to an average per-token cost reduction of 22.67%."
> "However, on math problems, GPT-3.5 with CCoT incurs a performance penalty of 27.69%."

**해석**: 단순 압축은 *수학에서는 성능 손실*이 큼 → SFT/RL 없는 prompt-only 압축의 한계.

### 6.3 TALE — arXiv 2412.18547
> "TALE follows a two-phase approach: budget estimation and prompt construction."
> "Token elasticity describes a critical phenomenon: when the token budget is reduced beyond a certain range, the token cost increases, indicating that further reductions in the budget lead to increasing token consumption."
> "With 0 as the left boundary, the current possible budget β is computed as the midpoint." (binary-search)
> "TALE reduces output token costs by 68.64% on average" with <5% accuracy decrease.
> Prompt format: "Let's think step by step and use less than [budget] tokens"

**해석**: ★ **"token elasticity"** 개념이 중요. 너무 작은 budget을 강제하면 오히려 토큰이 늘어남.

### 6.4 Sketch of Thought (SoT) — arXiv 2503.05179
> "SoT is designed as a flexible, modular approach and is instantiated with three paradigms — Conceptual Chaining, Chunked Symbolism, and Expert Lexicons — each tailored to distinct reasoning tasks"
> "selected dynamically at test-time by a lightweight routing model"
> "SoT achieves token reductions of up to 84% with minimal accuracy loss"
> "In tasks such as mathematical and multi-hop reasoning, it even improves accuracy while shortening outputs"

### 6.5 Coconut — arXiv 2412.06769
> "Coconut utilizes the last hidden state of the LLM as a representation of the reasoning state, termed 'continuous thought.'"
> "Instead of decoding this state into words, we feed it back to the model as the next input embedding directly in the continuous space."
> "Most word tokens primarily ensure textual coherence and are not essential for reasoning, while some critical tokens require complex planning and pose challenges to LLMs."
> "This latent reasoning paradigm enables an advanced reasoning pattern, where continuous thoughts can encode multiple alternative next steps, allowing the model to perform a breadth-first search (BFS)."

### 6.6 Compressed CoT (Cheng & Van Durme) — arXiv 2412.13171
> "special tokens used during inference to allow for extra computation"
> "generate contentful and continuous contemplation tokens of variable sequence length"
> "The generated contemplation tokens are compressed representations of explicit reasoning chains"
> "the reasoning improvements can be adaptively modified on demand by controlling the number of contemplation tokens generated"

### 6.7 Soft Thinking — arXiv 2505.15778
> "Soft Thinking, a training-free method that emulates human-like 'soft' reasoning by generating soft, abstract concept tokens in a continuous concept space."
> "These concept tokens are created by the probability-weighted mixture of token embeddings, which form the continuous concept space."
> "each generated concept token encapsulates multiple meanings from related discrete tokens, implicitly exploring various reasoning paths to converge effectively toward the correct answer."
> "improving pass@1 accuracy by up to 2.48 points while simultaneously reducing token usage by up to 22.4% compared to standard CoT."

### 6.8 Markovian Thinker — arXiv 2510.06557
> "the policy advances reasoning while conditioning on a constant-size state, decoupling thinking length from context size."
> "As an immediate consequence this yields linear compute with constant memory."
> "Delethink, an RL environment that structures reasoning into fixed-size chunks. Within each chunk, the model thinks as usual; at the boundary, the environment resets the context and reinitializes the prompt with a short carryover."
> "an R1-Distill 1.5B model reasons in 8K-token chunks yet thinks up to 24K tokens, matching or surpassing LongCoT-RL."

---

## 7. Section E — Routing / Classifier 기반

### 7.1 RouteLLM — arXiv 2406.18665
> "We propose several efficient router models that dynamically select between a stronger and a weaker LLM during inference, aiming to optimize the balance between cost and response quality."
> "We develop a training framework for these routers leveraging human preference data and data augmentation techniques."
> "Our approach significantly reduces costs—by over 2 times in certain cases—without compromising the quality of responses."

### 7.2 ThinkSwitcher — arXiv 2505.14183
> "ThinkSwitcher introduces a lightweight switching module trained with supervision signals derived from the relative performance of each reasoning mode across tasks."
> "a lightweight switcher module is employed to predict the reasoning mode likely to yield optimal performance for a given query."
> Decision rule: "m(q) = LC, if ŷ_LC(q) − ŷ_SC(q) ≥ τ; SC, otherwise"
> "The final training objective is the sum of the standard MSE loss and the margin loss: ℒ_switch = ℒ_MSE + λ_margin · ℒ_margin."
> "ThinkSwitcher reduces computational cost by 20–30% while maintaining high accuracy on complex tasks."
> "on simpler datasets such as GSM8K, it reduces inference tokens by around 30% with a performance loss of less than 1%."

### 7.3 CAR (Certainty-based Adaptive Routing) — arXiv 2505.15154
> "excessive reliance on chain-of-thought (CoT) reasoning can impair model performance and brings unnecessarily lengthened outputs"
> "prolonged reasoning does not universally improve accuracy and even degrade performance on simpler tasks"
> "dynamically switches between short answers and long-form reasoning based on the model perplexity"
> "CAR outperforms both short-answer and long-form reasoning approaches, striking an optimal balance between accuracy and efficiency"
> "triggering reasoning only when the model exhibits low confidence (i.e., high perplexity)"

### 7.4 Hawkeye (model collaboration) — arXiv 2504.00424
- Large model이 *concise reasoning instructions* 생성 → small model이 expand. 35% CoT만 사용해 comparable 응답 + 3.4× faster + 60% cost saving.

---

## 8. Section F — Confidence / Early-exit

### 8.1 DEER — arXiv 2504.15895
> "the proposed method monitors model behavior at potential reasoning transition points and dynamically terminates the next reasoning chain's generation"
> "when the model exhibits high confidence in a trial answer"
> "Our method requires no additional training and can be seamlessly integrated into existing o1-like reasoning LLMs."
> "reducing the length of CoT sequences by an average of 19.1% to 80.1% while improving accuracy by 0.3% to 5.0%."

### 8.2 Certaindex / Dynasor — arXiv 2412.20993
> "Certaindex generalizes certainty into a normalized confidence score that serves as a proxy to measure reasoning progress."
> "After a set number of generated tokens, we append the prompt: 'Oh, I suddenly got the answer to the whole problem. Final Answer: boxed{', and record the output answer."
> "All probe tokens and their responses are then discarded before resuming the original decoding path."
> "We measure consistency over a sliding window of width w steps: C_k = 1/w · ∑_{j=k−w+1}^{k} I[y_j = y_k]"
> "Once C_k ≥ τ for some threshold τ ∈ (0,1], we deem the model sufficiently certain and terminate generation at step k."
> "Dynasor dynamically adjusts token budgets for individual requests based on their real-time Certaindex values."
> "Demonstrate up to 50% compute savings and 3.3× higher throughput in real workloads with no accuracy drop."

**해석**: ★ training-free 시스템-level 해법. 자체적인 *probe* 메커니즘이 인상적.

---

## 9. Section G ★★★ — Mid-training 단계의 추론 능력 implant (사용자 핵심 요청)

> **결론을 먼저**: 2025-2026년에 mid-training이 *공식 명명된 단계*로 자리잡았고, "**hybrid 추론 능력과 길이 제어의 기반은 mid-training에서 만든다**" 라는 흐름이 명확하다. SmolLM3, Qwen3, DeepSeek-V3.1, OctoThinker, Thinking Augmented Pre-training이 직접 근거.

### 9.1 Mid-training Survey — arXiv 2510.23081
> "Mid-training is positioned as the critical bridge between general pre-training and post-training stages, characterized by intermediate computational demands and targeted large-scale data utilization."
> "Mid-training aims to preserve these foundations while amplifying targeted competencies, demonstrating steeper performance gains with less data and computation than pre-training."
> "Contemporary LLM development frameworks position pre-training as the bedrock stage, where models acquire broad competencies through next-token prediction on web-scale corpora."
> "Mid-training preserves these foundational capabilities with the same next-token prediction objective, while systematically enhancing specialized skills through curated data blends."
> "In contrast, post-training adopts specialized objectives for SFT/RL and task-specific alignment datasets."
> "Prevailing methodologies center on the strategic enrichment of reasoning-intensive data by synthesizing high-quality reasoning samples at scale, including CoT sequences, mathematical proof chains, and structured decision trajectories."
> "These approaches also dynamically calibrate data mixture ratios to progressively elevate the weight of reasoning data without compromising general capabilities."

→ Qwen3, DeepSeek-V3, Llama-3, Nemotron-4 모두 mid-training을 채택 (survey Table 4).

### 9.2 OctoThinker — arXiv 2506.20512 ★★
> "Mid-training is a mid-stage whose computational and data (token) requirements are intermediate between pre-training and post-training."
> "In the first stable stage, we train Llama models on a high-quality mixture of pre-training corpus for 200B tokens using a constant learning rate."
> "In the second decay stage, we anneal the learning rate and introduce distinct data mixtures—short CoT, long CoT, and a hybrid of both—to mid-train three separate branches."
> "OctoThinker-Long (long-reasoning data), OctoThinker-Short (short-reasoning data), OctoThinker-Hybrid (a mix of both) with decayed learning rate."
> "Long CoT patterns often induce excessive responses and sudden performance drops in RL-tuned models."
> "increasing the mid-training budget can lead to noticeable improvements in downstream RL performance."
> "The length of correct responses from the Qwen model increases steadily and reasonably throughout training, whereas Llama exhibits abnormal behavior."

**해석**: ★★★★ 본 보고서에서 사용자의 시나리오에 가장 직접적인 청사진. 
- *Stable-then-Decay* 2-stage scheduler
- short/long/**hybrid** 3-branch mid-train  
- **mid-train의 데이터 mix가 RL 단계의 길이 dynamics를 결정**

### 9.3 Interplay paper — arXiv 2512.07783
> "Mid-training is an intermediate phase between pre-training and post-training, gaining attention for its role in improving downstream fine-tuning and RL performance."
> "Mid-training acts as an intermediate distributional bridge between broad pre-training corpora and specialized post-training objectives, expanding the model's primitive coverage."
> "Mid-training stabilizes optimization and facilitates RL scaling by providing structured reasoning supervision, bridging the gap between broad pre-training corpora and reward-oriented RL data."
> "By focusing supervision on this boundary, we aim to strengthen higher-level reasoning priors that RL can amplify."
> "Mid-training typically involves using higher-quality or instruction-formatted data with next-token prediction or SFT objectives."
> "Mid-training significantly enhances performance under fixed compute compared with RL only, demonstrating its central but underexplored role in training pipelines."
> "Introducing a mid-training phase that bridges pre- and post-training distributions substantially strengthens both in-domain and out-of-domain performance under a fixed compute budget."
> "Process-level rewards reduce reward hacking and improve reasoning fidelity."

### 9.4 Thinking Augmented Pre-training (TPT) — arXiv 2509.20186
> "Mid-training, alternatively referred to as continual pre-training, enhances the capabilities of existing LLMs by further training on carefully curated datasets."
> "Thinking augmented Pre-Training (TPT), a simple and scalable approach to enhance the pre-training data efficiency by augmenting existing text data with thinking trajectories."
> "Given a document d from the pre-training dataset, a thinking trajectory t is generated using an off-the-shelf model with the specified prompt, where the placeholder {{CONTEXT}} is replaced by the document text."
> "The original document and the generated thinking trajectory are concatenated to form the augmented training sample x = [d; t]. We then minimize the standard next-token prediction loss to train LLMs on this augmented dataset."
> "Thinking augmentation breaks down complex tokens into smaller, more explainable steps, thereby effectively allocating more training compute to them."
> "TPT reduces the required training tokens by a factor of 3, underscoring its effectiveness in maximizing the utility of existing data."
> "TPT-8B achieves substantial improvements over the vanilla baseline," with GSM8k "from 19.2% to 50.1%."

**해석**: ★★★ 사용자의 pre/mid-train 단계에서 직접 차용 가능한 데이터 augmentation 패턴. 일반 웹 문서에 *생각*을 붙여 학습. Quiet-STaR의 산업적 진화판.

### 9.5 RA3 / Action Abstractions — arXiv 2509.25810
> "An effective mid-training phase should identify a compact set of useful actions and enable fast selection among them through online RL."
> "it characterizes an action subspace that minimizes both the value approximation error from pruning and the RL error during subsequent planning."
> "the importance of operating in the space of action abstractions rather than primitive actions."
> "These results suggest that mid-training is most effective when the decision space is compact and the effective horizon is short."
> "RA3 improves the average performance on HumanEval and MBPP by 8 and 4 points over the base model and the next-token prediction baseline."
> "RA3 achieves faster convergence and higher asymptotic performance in RLVR on HumanEval+, MBPP+, LiveCodeBench, and Codeforces."

### 9.6 Front-Loading Reasoning — arXiv 2510.03264
> "reasoning data is increasingly incorporated also during the mid-training stage"
> "Front-loading reasoning data into pretraining is critical (19% avg gain), establishing foundational capabilities that cannot be fully replicated by later-stage SFT"
> "We uncover an asymmetric principle for optimal data allocation: pretraining benefits most from broad diversity in reasoning patterns (11% avg gain), while SFT is more sensitive to data quality (15% avg gain)."
> "high-quality pretraining data has latent effects, activated only after SFT"

**해석**: ★★★ Pre/mid-train에 추론 데이터를 "front-load" 해야 한다 + 단계마다 *데이터 특성이 달라야 한다* (pre/mid: diversity, SFT: quality).

### 9.7 Quiet-STaR — arXiv 2403.09629 (mid-training 의 시초)
> "LMs learn to generate rationales at each token to explain future text, improving their predictions."
> "learnable tokens indicating a thought's start and end"
> "tokenwise parallel sampling algorithm"
> "after continued pretraining of an LM on a corpus of internet text with Quiet-STaR, we find zero-shot improvements on GSM8K and CommonsenseQA"

### 9.8 Pause Token — arXiv 2310.02226 (pre-train 단계!)
> "a (learnable) pause token, a sequence of which is appended to the input prefix. We then delay extracting the model's outputs until the last pause token is seen, thereby allowing the model to process extra computation before committing to an answer."
> "inference-time delays show gains when the model is both pre-trained and finetuned with delays."
> "a gain of 18% EM score on the QA task of SQuAD, 8% on CommonSenseQA and 1% accuracy on the reasoning task of GSM8k."

**해석**: ★ Pre-training 단계에서 *think token*을 학습시키는 가장 오래된 작품. 사용자 시나리오에서 **pre-train부터 think 토큰 표기를 학습** 시키려면 직접 참고.

### 9.9 SmolLM3 — HuggingFace blog (재인용, §3.7 참조)
> "We call the long context adaptation and reasoning adaptation 'mid-training'."
> "Our mid-training dataset contained 35B tokens sourced from Open Thought's OpenThoughts3-1.2M and a subset from NVIDIA's Llama-Nemotron-Post-Training-Dataset-v1.1 with reasoning traces from R1."
> "In non-reasoning mode, we pre-fill the model's response with empty think blocks, similar to Qwen3, to ensure direct answers without explicit reasoning."

### 9.10 DeepSeek-V3.1 long-context mid-training (재인용)
> "DeepSeek-V3.1-Base ... is built upon the original V3 base checkpoint through a two-phase long context extension approach."
> "The 32K extension phase has been increased 10-fold to 630B tokens, while the 128K extension phase has been extended by 3.3x to 209B tokens."

---

## 10. Section H — 길이 reward의 함정과 reward-hacking

길이 제어 RL에는 reward hacking 함정이 있어 별도로 짚는다.

### 10.1 Demystifying Long CoT (재인용, §4.6)
- *반복(repetition)* 으로 길이를 늘리는 hacking이 발생 → **repetition penalty 필수**.
- > "reward hacking, where it increased the lengths of its CoTs on hard questions using repetition rather than learning to solve them."

### 10.2 ODIN (재인용, §4.23)
- Length head를 disentangle해 RM에 그 head를 *버리고* RL.

### 10.3 "Is It Thinking or Cheating?" — arXiv 2510.01367
- TRACE 지표: *추론을 일찍 truncate 했을 때도 reward를 받는다면 hacking*. AUC of accuracy-vs-length curve로 판별.

### 10.4 함의
사용자 RL 단계에서:
- 길이 reward는 **정답에만 적용** (Kimi k1.5, Short-RL 합의)
- **soft punishment in interval** (DAPO)
- **repetition penalty** (Demystifying Long CoT)
- length-disentangled RM (ODIN) 옵션
- *추론 길이 0에 가까울 때도 정답이면 의심*해 보는 evaluation loop (TRACE)

---

## 11. Section I — 두 개의 통합 Survey

### 11.1 Reasoning on a Budget — arXiv 2507.02076
> "We introduce a two-tiered taxonomy that distinguishes between L1-controllability, methods that operate under fixed compute budgets, and L2-adaptiveness, methods that dynamically scale inference based on input difficulty or model confidence."
> "they apply fixed inference-time compute regardless of task complexity, often overthinking simple problems while underthinking hard ones."

### 11.2 Concise and Adaptive Thinking — arXiv 2507.09662
> "We categorize existing works into two directions: (1) Training-free methods... (2) Training-based methods, which focus on shortening the reasoning length and teaching the LRMs to think adaptively by fine-tuning (SFT/DPO) or reinforcement learning (RL)."
> "There exist optimal reasoning lengths for different tasks that allow the model to think adaptively based on input difficulty."
> "Studies reveal that deliberative reasoning capability by extremely long reasoning chains does not consistently improve model performance across diverse tasks."
> "The majority of these methods are exclusively evaluated on reasoning-focused benchmarks, leaving their impacts on models' fundamental capabilities underexplored."

### 11.3 Don't Overthink It — arXiv 2508.02120
> "when generating answers, these models often construct excessively long reasoning chains with redundant or repetitive steps, which leads to reduced reasoning efficiency and may affect the accuracy of the final answer"
> "two main directions based on the lens of single-model optimization versus model collaboration"

---

## 11.4 한국 LLM의 Hybrid Reasoning 처리 (사용자 직결)

> 한국에서 from-scratch로 hybrid reasoning LLM을 만든다면 직접 참고할 사례. Mid-training/data 측면은 *mid_training_dataset_report.md* §2.7 참조. 여기서는 **길이 제어·hybrid mode 메커니즘**만 정리.

### 11.4.1 HyperCLOVA X THINK (Naver) — arXiv 2506.22403 ★★
- **Chat template 채널 분리**: `<|im_start|>assistant/think ... <|im_end|>` 형태로 think를 *별도 role*로 처리. Qwen3/SmolLM3가 think를 assistant 응답 안의 plain text 블록으로 두는 것과 대비.
- **Hybrid 토글**: `force_reasoning` / `skip_reasoning` 플래그 (Qwen3의 `/think`·`/no_think` 직접 대응).
- **★ Length Controllability (LC)**: 학습 시점부터 *"Think for maximum 1024 tokens"* 형태의 prompt를 SFT/RL 단계에 implant. **사용자가 자연어로 budget을 지정**하는 가장 정교한 한국 사례. Qwen3의 `thinking_budget` 메타 슬롯보다 자연어 친화적.
- **Multi-turn 정책**: 직전 turn의 reasoning을 다음 prompt에 *포함하지 않음* → context window 절약 + reasoning이 다음 답을 contaminate 안 함.
- **Post-train 순서**: SFT → RLVR → LC → RLHF+RLVR joint. **LC를 RLVR 직후 별도 stage로 둔다**는 점이 독특.

### 11.4.2 EXAONE 4.0 (LG AI Research) — arXiv 2507.11407 ★★
- **★ Token-less hybrid**: Reasoning/Non-Reasoning SFT를 **단일 데이터셋에 1.5:1 비율로 섞어 동시 학습**. Special token도, chat template 플래그도 사용하지 않음. Qwen3/SmolLM3의 *데이터 분리 + 플래그 토글* 패러다임과 정반대의 단순화.
- **모드 구분 시그널**: generation length와 temperature 만으로 처리 (분기 학습된 결과로 자연스레 emerge).
- **AGAPO (Asymmetric Sampling + Global Advantage PO)**: PPO clipping 제거로 exploratory token 보존, all-incorrect sample에 *음의 작은 reward* 유지(0이 아닌), sequence-level KL penalty. **DAPO·GRPO 한국식 변형 가장 구체적 공개 사례**.
- **Preference learning 2단계**: (1) verifiable + 간결성 → (2) preference + language consistency. 한국어 빌더 특유의 *언어 일관성* reward 추가.

### 11.4.3 EXAONE Deep (LG) — arXiv 2503.12524
- **`<thought>...</thought>` XML 태그**: DeepSeek-R1과 Qwen3 사이의 중간 포맷.
- **데이터 규모**: 12B 토큰 (1.6M instances) SFT + 20K DPO + 10K Online RL.
- **SimPER (DPO 변형) + GRPO 변형** 조합. SimPER는 length-normalized DPO로 길이 인플레 방어.

### 11.4.4 Mi:dm K 2.5 Pro (KT, 2026-03) — arXiv 2603.18788
- **Channel tokens (가장 정교한 한국 hybrid 포맷)**: `analysis` / `commentary` / `final` 3-channel 분리. think를 단일 블록이 아닌 *기능별 채널*로 쪼개는 시도. OpenAI o-series의 hidden reasoning + visible response 구조에 가장 근접한 오픈 모델.
- **Multi-turn 정책**: 마지막 reasoning만 유지, 이전 trace는 폐기 (HyperCLOVA X THINK와 공통).
- **Pipeline**: Reasoning SFT → 모델 merging → Reasoning RL → Fusion SFT → Fusion RL. 5-stage post-training은 본 보고서 sample 중 가장 정교.
- **Asynchronous RL**: trainer-rollout 완전 분리로 variable-length reasoning에 유리. 길이 가변성이 큰 reasoning RL의 throughput 문제 해결 방향.

### 11.4.5 Trillion 7B (Trillion Labs) — arXiv 2504.15431
- Reasoning 변형은 없으나 **Cross-lingual Document Attention (XLDA)** 가 길이/추론과 간접 연관: 다국어 문서를 같은 시퀀스에 packing할 때 cross-document attention을 *차단하지 않음* → 영어 reasoning 능력이 한국어로 전이.
- **MTP (Multi-Token Prediction) loss** 도입. 추론 시점 speculative decoding 가능 → 길이가 길어져도 throughput 손해 적음.
- **비용 reference**: $148K / 59.4K H100-h / 2T tokens.

### 11.4.6 한국 사례 종합 시사점 (length-control 관점)

| 패턴 | 채택 모델 | 사용자 시나리오 적합도 |
|---|---|---|
| Chat template **별도 role 채널** | HyperCLOVA X THINK | ★★ 가장 정교, 단 chat template 복잡도↑ |
| **XML tag** (`<think>`/`<thought>`) | EXAONE Deep, Qwen3, R1 | ★★★ 산업 표준, BPE 자연스러움 |
| **Channel tokens** (analysis/commentary/final) | Mi:dm K 2.5 Pro | ★ 오버엔지니어링 위험, 단 functional 분리 가능 |
| **Token-less data mixing** | EXAONE 4.0 | ★★ 구현 가장 단순, 모드 구분이 학습된 emergent property |
| **자연어 budget prompt** | HyperCLOVA X THINK (LC stage) | ★★★ UX 친화, Length Controllability 학습 필요 |

**한국 RL 변형 정리**:
- **AGAPO** (EXAONE 4.0): clip 제거 + all-incorrect에 음의 reward + seq-level KL
- **SimPER + GRPO 변형** (EXAONE Deep): length-normalized DPO + GRPO
- **Asynchronous RL** (Mi:dm K 2.5 Pro): variable-length 효율
→ DeepSeek-R1 vanilla GRPO를 그대로 쓴 한국 모델 0건. 모두 *언어 일관성 / 길이 인플레 / variable-length throughput* 중 한두 가지에 대응한 변형.

---

## 11.5 2026년 1-5월 신작 (cutoff 이후)

> 본 보고서 본문이 2025년 12월까지를 cutoff로 잡고 있어, 그 이후 발표된 주요 작품을 별도 섹션으로 정리. 모두 arXiv 확인 완료. 길이 제어 분야가 2026년에도 활발한 후속 연구를 만들어내고 있음을 확인.

### 11.5.1 Hybrid Thinking — Mode Separation & Reward Hacking
- **Thinking-Based Non-Thinking** — arXiv 2601.04805 (Gan et al., 2026-01). **AdaptThink의 직접 후속**. /no_think 모드가 collapse하거나 /think를 mimicking하는 *reward hacking* 발견 → asymmetric reward로 해결.
- **Path-Lock Expert** — arXiv 2604.27201. **MoE-style 아키텍처 분리**로 think/no-think expert가 서로 leak되지 않도록 강제. Qwen3 식 *데이터-only 모드 토글*에 대한 아키텍처 레벨 대안. ★ 사용자가 from-scratch면 직접 채택 검토 가치.
- **Reasoning Primitives in Hybrid and Non-Hybrid LLMs** — arXiv 2604.21454. Mechanistic study. Hybrid 모델이 난이도 증가에도 robust한 이유를 reasoning token과 architectural inductive bias의 *서로 다른 computational level* 기여로 설명.

### 11.5.2 Length-Penalty RL 신작
- **Tackling Length Inflation Without Trade-offs (GR3)** — arXiv 2603.10535. 정답이지만 긴 답이 GRPO에서 음의 advantage가 되는 알려진 pathology를 *group-relative reward rescaling*으로 해결. **ALP/Laser-D 직접 개선판**.
- **Stepwise Penalization for Length-Efficient CoT** — arXiv 2603.00296. Sequence-level 길이 페널티를 step-level로 이동 — *어떤 step만 짧게* 가능. DAST/Laser-D 후속.
- **Compress the Easy, Explore the Hard (CEEH)** — arXiv 2602.22642. 명시적 길이 페널티 대신 *난이도-conditioned entropy regularization*. Laser-D의 difficulty bucket과 자연스레 결합 가능.
- **CODA (Difficulty-Aware Compute Allocation)** — arXiv 2603.08659. ALP/DAST 계보의 difficulty-adaptive 형제.

### 11.5.3 Self-Estimated / Budget-Aware
- **Conformal Thinking** — arXiv 2602.03814. **Distribution-free conformal early-stop threshold**. 통계적으로 grounded한 BudgetThinker 대안.
- **Variational Posterior Guidance with Efficiency Awareness** — arXiv 2605.11019 (ICLR 2026). 추론 길이에 대한 variational posterior + adaptive length-shaped reward. **Laser-D 직접 후속**.
- **LEAD (Length-Efficient Adaptive and Dynamic Reasoning)** — arXiv 2605.09806. *Symmetric reward*: over-thinking과 over-compression 둘 다 페널티. 모델 자체 correct rollout에서 per-problem target length를 online 추정. **Laser-D의 asymmetric-only 약점을 막은 후속**.
- **Ares (Adaptive Reasoning Effort Selection)** — arXiv 2603.07915. *Agent loop의 매 step별* effort 선택. 기존 보고서는 one-shot QA 위주였는데 agent 시나리오 확장.

### 11.5.4 Overthinking 완화 (Training-free)
- **ROM (Real-time Overthinking Mitigation)** — arXiv 2603.22016. Streaming detector + mid-generation 개입. DEER의 발전형.
- **Reasoning Path Deviation Monitoring** — arXiv 2603.14251. High-entropy transition token을 deviation signal로 early-exit. Plug-and-play.
- **Precedent-Informed Reasoning** — arXiv 2602.14451. Exhaustive self-exploration 대신 *case retrieval로 제약된 추론*.
- **Difficulty-aware RL (DiPO)** — arXiv 2601.21418. DAST/Laser-D 계보의 또 하나의 entrant.

### 11.5.5 Routing 신작
- **ThinkRouter** — arXiv 2602.11683. *Latent reasoning과 explicit CoT 간 routing*. Coconut + 명시적 reasoning을 confidence로 분기. +19.7 Pass@1 / -15.5% 길이.
- **R2-Router** — arXiv 2602.02823. *모델 + budget* 을 동시에 선택하는 routing.

### 11.5.6 Frontier 모델 신작 (adaptive thinking 명시 채택)
- **Qwen3.5 / Qwen3.5-Plus** (2026-02-15/17): 397B-total/17B-active MoE + Gated Delta Networks. /think /no_think hybrid를 default로 *상속*.
- **DeepSeek V4-Pro / V4-Flash** (2026-04-24): **3-level thinking ladder** 첫 메이저 오픈 모델 (Non-think / Think High / Think Max). 1M context.
- **GLM-5 / GLM-5.1 Reasoning** (Z.ai, 2026-02/04): 744B-total/40B-active MoE. **Turn-level thinking 토글**(Qwen3는 session-level이었음).
- **Claude Opus 4.5 / 4.6** (Anthropic, 2026): 4.6에서 "adaptive thinking" — **server-side effort routing**으로 사용자 지정 budget 대체. *"effort routing replaces model routing"*.
- **GPT-5.4 / 5.5** (OpenAI, 2026): `reasoning.effort ∈ {none, low, medium, high, xhigh}` — **5-level + true zero-thinking**. 같은 weight에서 think를 완전히 끌 수 있는 첫 사례.
- **Gemini 3 / 3.1 Pro** (Google, 2026): 숫자 `thinkingBudget` → 질적 `thinkingLevel` (LOW/MEDIUM/HIGH). Dynamic thinking이 default.
- **Mistral Medium 3.5** (2026-04-30): Magistral(reasoning) + Devstral 2(coding)를 128B dense 단일 모델로 **병합**. 분리 추세에 대한 역방향.

### 11.5.7 2026 신작 종합 시사점

1. **Hybrid 모델의 reward hacking**이 새로운 핵심 risk로 부상 — 2601.04805 (Thinking-Based Non-Thinking)가 신호탄. 사용자는 RL 단계에서 *think-off 모드의 quality*를 별도 평가 지표로 둘 것.
2. **Architecture-level 모드 분리**(MoE expert lock)가 데이터-only 분리의 한계를 보완 — *from-scratch 빌더라면 base 모델 단계에서 think/no-think expert를 미리 분리하는 설계 고려*.
3. **Budget 표현의 추상화**: GPT-5.5의 `none/low/medium/high/xhigh`, Gemini 3의 `LOW/MEDIUM/HIGH` → 정수 토큰 budget보다 *질적 effort level*이 UX 표준화 중. SmolLM3/Qwen3식 `/think /no_think`보다 한 단계 더 진화.
4. **Symmetric length reward** (LEAD)가 Laser-D의 asymmetric-only를 보완 — 권고: §13의 RL 레시피에 LEAD 식 *under-thinking penalty*도 추가 검토.
5. **Server-side effort routing** (Claude 4.6) — *모델이 자체적으로* 적절한 effort를 정함. 사용자 시나리오의 궁극 목표인 *"어려운 건 길게, 쉬운 건 짧게"*와 가장 일치하는 패러다임. 학습 시점부터 internal effort predictor를 implant하는 방향이 차세대 표준이 될 가능성.

---

## 12. 전체 논문 64편 요약 표

| # | 논문/모델 | arXiv ID / 출처 | 단계 | 핵심 메커니즘 | 길이 감소 |
|---|---|---|---|---|---|
| 1 | Qwen3 | 2505.09388 | post-train Stage 3 | /think + /no_think + budget | budget-controlled |
| 2 | Claude 3.7 Sonnet | Anthropic blog | post-train | budget_tokens API | user-controlled |
| 3 | DeepSeek-V3.1 | HF card | post-train + long ext. | chat template hybrid | mode-switched |
| 4 | GLM-4.5 | 2508.06471 | multi-stage post-train | thinking/non-thinking mode | per-turn control |
| 5 | Nemotron Nano 2 | 2508.14444 | SFT (10% no-think + 5% truncated) | /think /no_think + budget | 6× throughput vs Qwen3 |
| 6 | Magistral | 2506.10910 | pure-RL GRPO | soft length penalty | log scaling |
| 7 | SmolLM3 | HF blog | **mid-training** | /think /no_think + empty think blocks | mode-switched |
| 8 | GPT-5 | OpenAI docs | n/a | reasoning_effort + verbosity | minimal/low/med/high/xhigh |
| 9 | Gemini 2.5 | Google docs | n/a | thinkingBudget (0,-1,N) | dynamic by default |
| 10 | L1 / LCPO | 2503.04697 | RL | r = I(correct) - α|n_gold - n_y| | sub-GPT-4o length, equal acc |
| 11 | Kimi k1.5 | 2501.12599 | RL + long2short | length penalty only on correct | shortest correct selected |
| 12 | O1-Pruner | 2501.12570 | offline RL (PPO-style) | L_ref/L(y)−1+λ·Δacc | maintained acc |
| 13 | DAST | 2503.04472 | RL preference | Token Length Budget (TLB) | 30%+ reduction |
| 14 | ThinkPrune | 2504.01296 | iterative RL | hard token cap, reward=0 if exceed | 50% length, 2% acc drop |
| 15 | Demystifying Long CoT | 2502.03373 | RL | Cosine Reward + repetition penalty | stable scaling |
| 16 | DeepSeek-R1 | 2501.12948 | RL | accuracy + format only | length emergent |
| 17 | AdaCoT | 2505.11896 | PPO + SLM | Pareto opt of CoT trigger | 3.18% trigger, 69% token cut |
| 18 | Ada-R1 | 2504.21659 | model merge + bi-level DPO | group + instance preference | 50%+ length cut |
| 19 | AdaptThink | 2505.13417 | RL constrained | prefer NoThinking subject to acc | 53% length, +2.4% acc |
| 20 | AutoThink | 2505.10832 | multi-stage RL | "..." trigger, 3-stage shaping | 52% token, +6.4% acc |
| 21 | LASER / Laser-D/DE | 2505.15612 | GRPO | per-query target length | 63% token, +6.1% AIME24 |
| 22 | ALP | 2506.05256 | RL | penalty ∝ 1/solve-rate | 50% token |
| 23 | ASRR | 2505.15400 | RL | accuracy-aware length reward | 32.5% budget cut |
| 24 | Short-RL | 2505.12284 | RL | length reward on correct only | 40% length, +14% acc (logic) |
| 25 | SelfBudgeter | 2505.11274 | budget-guided GRPO | self-estimate budget | 61% length cut |
| 26 | e1 | 2510.27042 | Adaptive Effort Control RL | user-specified fraction of tokens | 2-3× length cut |
| 27 | Thinking Fast & Right | 2505.18298 | RL | dynamic reward weight | balanced |
| 28 | MRT | 2503.07572 | meta-RL | progress reward | budget-agnostic |
| 29 | HiPO | 2509.23967 | hybrid RL | Think-on / Think-off pair | balanced |
| 30 | ADR | 2510.10207 | EHPO | entropy + difficulty penalty | 49-59% length, +6.1% acc |
| 31 | ODIN | 2402.07319 | RM | disentangled length head | removes hacking |
| 32 | Light-R1 | 2503.10460 | curriculum SFT+RL | 3-stage difficulty curriculum | n/a |
| 33 | LIMR | 2502.11886 | RL data sel | LIM importance metric | 1389 vs 8523 samples |
| 34 | ProRL | 2505.24864 | prolonged RL | KL ctrl + ref reset | novel reasoning |
| 35 | DAPO | 2503.14476 | RL | overlong soft, token-level loss | AIME 50/100 |
| 36 | CoT-Valve | 2502.09601 | SFT + LoRA | parameter-space direction | 741→225 tok @94.92% acc |
| 37 | C3oT | 2412.11664 | conditioned SFT | compressor + cond inference | 50%+ |
| 38 | LIMO | 2502.03387 | minimal SFT | LIMO hypothesis | 1% data, 63.3% AIME24 |
| 39 | s1 | 2501.19393 | SFT 1K + budget forcing | append "Wait" | 50→57% AIME24 |
| 40 | Train Long, Think Short | 2508.08940 | GRPO curriculum | exponential budget decay | 38.8→43.4% |
| 41 | Chain of Draft | 2502.18600 | prompt | drafting style | 7.6% tokens |
| 42 | Concise CoT (Renze) | 2401.05618 | prompt | "concise" instruction | 48.7% length |
| 43 | TALE | 2412.18547 | prompt | binary-search budget | 68.6% token cut |
| 44 | Sketch of Thought | 2503.05179 | prompt + router | 3 paradigm router | 84% reduction |
| 45 | Coconut | 2412.06769 | architecture | continuous latent thought | latent BFS |
| 46 | Compressed CoT | 2412.13171 | architecture | contemplation tokens | variable length |
| 47 | Soft Thinking | 2505.15778 | training-free | prob-weighted concept tokens | -22.4% tokens, +2.48 acc |
| 48 | Markovian Thinker | 2510.06557 | RL env | chunked context reset | linear scaling |
| 49 | Pause Token | 2310.02226 | **pre-train + SFT** | learnable pause tokens | +18% SQuAD |
| 50 | Quiet-STaR | 2403.09629 | **continued pretrain** | per-token rationale | zero-shot GSM8K↑ |
| 51 | TPT (Thinking Augmented) | 2509.20186 | **mid-training** | thinking trajectory aug | 3× data efficiency |
| 52 | OctoThinker | 2506.20512 | **mid-training** | Stable-then-Decay, 3 branches | RL-stabilized |
| 53 | RA3 | 2509.25810 | **mid-training RL** | action abstractions | +8 HumanEval |
| 54 | Front-Loading Reasoning | 2510.03264 | **pre/mid + SFT** | asymmetric data allocation | +19% pre-train gain |
| 55 | Interplay paper | 2512.07783 | **pre/mid/RL** | mid as distributional bridge | foundational |
| 56 | Mid-training Survey | 2510.23081 | **survey** | data recipes, comparisons | systematic |
| 57 | RouteLLM | 2406.18665 | router train | preference data | 2×+ cost cut |
| 58 | ThinkSwitcher | 2505.14183 | switcher MSE+margin | LC vs SC routing | 20-30% cost cut |
| 59 | CAR | 2505.15154 | perplexity-based | low-confidence → reason | best of both |
| 60 | Hawkeye | 2504.00424 | model collab | big→concise, small→expand | 35% tokens, 3.4× faster |
| 61 | DEER | 2504.15895 | training-free | confidence at "Wait" tokens | 19-80% CoT cut, +0.3-5% acc |
| 62 | Certaindex / Dynasor | 2412.20993 | training-free | semantic entropy + probe | 50% compute, 3.3× tput |
| 63 | Reasoning on a Budget survey | 2507.02076 | survey | L1/L2 taxonomy | overview |
| 64 | Concise & Adaptive survey | 2507.09662 | survey | training-free / training-based | overview |
| 65 | Don't Overthink It survey | 2508.02120 | survey | single / collab | overview |
| 66 | "Is It Thinking or Cheating?" | 2510.01367 | eval | TRACE AUC | hacking detection |
| **— 한국 LLM (§11.4) —** | | | | | |
| 67 | HyperCLOVA X THINK | 2506.22403 | SFT+RLVR+LC+RLHF | think 별도 role channel + Length Controllability prompt | budget by NL prompt |
| 68 | EXAONE Deep | 2503.12524 | SFT+SimPER+GRPO 변형 | `<thought>` XML tag | 12B SFT + 20K DPO + 10K RL |
| 69 | EXAONE 4.0 | 2507.11407 | SFT 1.5:1 mix + AGAPO | **token-less hybrid** (모드 분리 없음) | length+temperature로 자연 분기 |
| 70 | Mi:dm K 2.5 Pro | 2603.18788 | 5-stage post-train + Async RL | analysis/commentary/final 3-channel | functional 채널 분리 |
| 71 | Trillion 7B | 2504.15431 | XLDA + MTP | cross-lingual attention 안 차단 | $148K / 2T tokens |
| **— 2026 신작 (§11.5) —** | | | | | |
| 72 | Thinking-Based Non-Thinking | 2601.04805 | RL | asymmetric reward로 /no_think collapse 방지 | reward hacking fix |
| 73 | Path-Lock Expert | 2604.27201 | MoE arch | think/no-think expert 분리 | architecture-level toggle |
| 74 | Reasoning Primitives (mechanistic) | 2604.21454 | analysis | hybrid robustness 원인 분석 | conceptual |
| 75 | Tackling Length Inflation (GR3) | 2603.10535 | GRPO 개선 | group-relative reward rescaling | ALP/Laser-D 개선 |
| 76 | Stepwise Penalization | 2603.00296 | RL | step-level 길이 페널티 | DAST/Laser-D 후속 |
| 77 | Compress Easy Explore Hard (CEEH) | 2602.22642 | RL | difficulty-conditioned entropy reg | length penalty 대체 |
| 78 | CODA | 2603.08659 | inference | difficulty-aware compute allocation | ALP 형제 |
| 79 | Conformal Thinking | 2602.03814 | inference | conformal early-stop | distribution-free |
| 80 | Variational Posterior Guidance | 2605.11019 | RL (ICLR 2026) | variational posterior over length | Laser-D 후속 |
| 81 | LEAD | 2605.09806 | RL | **symmetric** reward (over+under both) | Laser-D 약점 보완 |
| 82 | Ares | 2603.07915 | agent | per-step effort selection | agent loop extension |
| 83 | ROM | 2603.22016 | inference | streaming detector + intervention | DEER 발전형 |
| 84 | Reasoning Path Deviation | 2603.14251 | inference | high-entropy token으로 exit | plug-and-play |
| 85 | Precedent-Informed | 2602.14451 | inference | case-retrieval로 제약 | exhaustive search 대체 |
| 86 | DiPO | 2601.21418 | RL | difficulty-aware penalty | DAST family |
| 87 | ThinkRouter | 2602.11683 | inference | latent vs discrete routing | Coconut hybrid |
| 88 | R2-Router | 2602.02823 | routing | model + budget 동시 선택 | RouteLLM 확장 |
| 89 | Qwen3.5 / -Plus | 2026-02 release | post-train | /think /no_think (Qwen3 상속) | MoE 397B/17B-active |
| 90 | DeepSeek V4 Pro/Flash | 2026-04 release | post-train | **3-level ladder** (Non/High/Max) | 1M context |
| 91 | GLM-5 / 5.1 Reasoning | 2026-02/04 release | post-train | turn-level hybrid toggle | 744B-total/40B-active |
| 92 | Claude Opus 4.5 / 4.6 | 2026 release | post-train | **server-side effort routing** | "effort routing > model routing" |
| 93 | GPT-5.4 / 5.5 | 2026 release | post-train | 5-level + true zero-thinking | reasoning.effort=none 가능 |
| 94 | Gemini 3 / 3.1 Pro | 2026 release | post-train | thinkingLevel (qualitative) | dynamic by default |
| 95 | Mistral Medium 3.5 | 2026-04-30 release | post-train | Magistral+Devstral 병합 | 분리→통합 역추세 |

(총 95개 사례 — 본 보고서 작성 이후 추가 검증으로 한국 LLM 5건 + 2026 신작 19건 추가)

---

## 13. 사용자의 시나리오에 대한 종합 권고

> 사용자는 **pre-training 단계부터** hybrid reasoning LLM을 만든다. 즉 base 모델은 외부에서 가져오는 게 아니다. 아래는 위 64편의 합의를 종합한 **단계별 레시피**.

### Stage 1 — Pre-training (general phase)

- **길이 제어는 여기서 직접 다루지 않는다.** 어떤 논문도 *pre-train 단계에서 직접 길이를 학습*시키지 않는다 (Pause Token이 유일한 예외이지만 1B 규모의 학술 데모).
- 다만 **다음 단계(mid-train)에서 reasoning data를 효과적으로 받아들이려면** 충분한 도메인 지식이 pre-train에 *encoded* 되어 있어야 한다 — 이것이 LIMO 가설의 핵심.
  > "In foundation models where domain knowledge has been comprehensively encoded during pre-training, sophisticated reasoning can emerge through minimal but strategically designed demonstrations of cognitive processes." — LIMO

**구체적 제안**:
1. 전체 pre-train 토큰의 마지막 ~25%에서 **수학/코드/논리/proof 비중을 점진적으로 늘려라** (Front-Loading Reasoning, 2510.03264).
2. *Pause Token*을 attention sink로 미리 깔아두는 것을 고려 (단, 효과 보장은 작은 모델에서만 검증됨).
3. *Think token* (`<think>`, `</think>`)을 vocabulary에 미리 reserve 해두고, pre-train 코퍼스의 일부에 R1-스타일 reasoning trace를 자연 분포로 섞는다 (TPT 2509.20186 — *thinking trajectory를 prepend*).

### Stage 2 — Mid-training (★ 가장 중요한 단계)

**OctoThinker의 *Stable-then-Decay* 2-stage 레시피**가 가장 검증된 청사진:

1. **Stable phase**: pre-train 끝에서 이어 LR 고정, 약 200B 토큰. 일반 코퍼스의 quality-filtered subset + math/code 비중 증가.
2. **Decay phase**: LR cosine decay 시작 + **3-branch 데이터 mixture** (★ 사용자 시나리오에 직결):
   - **short-CoT branch**: 단순/직접 답변, no-think 데이터
   - **long-CoT branch**: R1-style long reasoning traces (OpenThoughts3, Llama-Nemotron-Post-Training-Dataset 권장)
   - **hybrid branch**: 두 가지를 섞고 chat template로 `/think` `/no_think` 토큰을 자연스럽게 학습
   - 약 20-35B 토큰, 3-4 epoch (SmolLM3과 동일)

이 단계에서 모델은 **"어떤 토큰을 받았을 때 thinking을 시작하고 어떤 토큰을 받았을 때 즉답해야 하는지"** 의 prior를 얻는다.

- 추가로 **TPT (2509.20186)** 식 augmentation을 mid-train 중에 적용 가능 — *일반 웹 문서에 thinking trajectory를 자동 생성해 prepend*. 3× data efficiency.
- **추론 길이 dynamics는 mid-train의 long-CoT 비율로 조절된다**. OctoThinker:
  > "Long CoT patterns often induce excessive responses and sudden performance drops in RL-tuned models."
  → long-CoT만 강하게 학습하면 RL 단계에서 폭주. **hybrid branch가 RL 안정에 결정적**.

### Stage 3 — SFT (Hybrid SFT)

Qwen3, SmolLM3, Nemotron Nano 2의 합치된 레시피:

1. **데이터 구성**:
   - thinking 활성: long-CoT 응답 (90%)
   - thinking 비활성: **빈 `<think></think>` 블록 + 직접 답변** (Qwen3/SmolLM3 패턴) (5-10%)
   - **truncated reasoning trace**: 일부러 중간에 잘린 CoT (Nemotron Nano 2: 5%) → thinking_budget 학습용
2. **Chat template**:
   - `/think`, `/no_think` 시스템 프롬프트 플래그 (Qwen3/SmolLM3)
   - 또는 explicit `budget_tokens` 필드 (Claude 3.7 패턴) — chat template의 메타 슬롯으로 첨부
3. **Critical**: non-think 모드에서는 `<think></think>` 빈 블록 prefill — 이것이 *thinking 토큰 자체를 안 만드는* 게 아니라 *항상 같은 구조를 유지하면서 비워두는* 것이 안정성에 도움. (SmolLM3, Qwen3 공통)

### Stage 4 — Preference (선택, but 권장)

- SmolLM3는 **APO (Anchored Preference Optimization)** 를 사용: thinking-mode 응답 페어 + non-thinking 페어 둘 다 학습.
- 단순화 옵션: DPO with thinking-mode marker.

### Stage 5 — RL (길이 제어의 마지막 단계)

가장 검증된 조합:
- **알고리즘**: GRPO (혹은 DAPO improvements)
- **Reward 구성** (composite):
  1. **Accuracy reward**: rule-based +1 / -0.5 / format -1 (Laser, DeepSeek-R1)
  2. **Length reward**: *난이도 적응* — Laser-D / DAST 패턴
     - 한 query를 K번 rollout → solve-rate p̂
     - target length L_T = L_min + (L_max − L_min) · (1 − p̂)  (쉬울수록 짧은 target)
     - length penalty = β · max(0, L(y) − L_T) — *correct에 한해서* (Kimi k1.5)
  3. **Overlong soft punishment** (DAPO):
     - L(y) ∈ [L_max − Δ, L_max] 구간에서 부드럽게 페널티 증가, 그 위는 truncate
  4. **Repetition penalty** (Demystifying Long CoT, 2502.03373): n-gram repetition 비율에 비례한 페널티
- **Iterative context lengthening** (DeepScaleR / Magistral): L_max = 16k → 24k → 32k → 64k 점진적으로
- **Reward-hacking 방어**:
  - ODIN처럼 length-disentangled RM (옵션)
  - TRACE (2510.01367) 평가 loop을 별도로 두어 *"길이를 잘랐을 때도 reward 잘 받는다"* = hacking 신호로 모니터.

### Stage 6 — Inference-time 보조 (training과 무관)

훈련이 끝난 모델에 추가로 얹을 수 있는 *training-free* 메커니즘:
- **DEER 식 confidence-based early exit**: "Wait" 토큰 등에서 모델이 trial answer에 confidence 높으면 종료
- **Dynasor/Certaindex 식 probing**: 일정 토큰 간격으로 `"Final answer: ..."` 를 주입해 답을 받아보고, 3번 연속 일치하면 종료 — 자체 모델 변경 없이 19-80% 토큰 감소.

### 우선순위 시퀀스 (resource 제약 시)

만약 모든 단계를 다 못 한다면:
1. **반드시**: Mid-training 3-branch hybrid (OctoThinker) + SFT hybrid (Qwen3/SmolLM3)
2. **강력 추천**: 난이도 적응 length-reward RL (Laser-D 또는 DAST 또는 ALP 중 택1)
3. **있으면 좋음**: Inference-time DEER/Dynasor 또는 ThinkSwitcher 식 routing

---

## 14. 결론

- "추론이 필요할 때만 길게 추론하는 LLM"의 학계 명칭은 **adaptive reasoning length control / L2-adaptiveness**이다.
- 산업적 합의는 **mid-training + hybrid SFT + 난이도 적응 length-reward RL**의 3단 콤보. Qwen3, SmolLM3, DeepSeek-V3.1, Nemotron Nano 2, GLM-4.5가 모두 이 패턴의 변형.
- 사용자가 *pre-training 단계부터* 모델을 만든다는 이점은: **mid-training 단계에서 OctoThinker식 3-branch hybrid를 자유롭게 설계할 수 있다는 것**. 기존 hybrid 모델들이 ex-post 로 갖춘 능력을 *기초부터 일관성 있게* 심을 수 있다.
- 가장 큰 리스크는 두 가지: (i) 길이 RL의 *reward hacking* (반복으로 길이 부풀리기), (ii) long-CoT 데이터가 mid-train에서 과다하면 RL 단계에서 폭주. **두 가지 모두 DAPO식 soft penalty + 난이도 적응 reward + repetition penalty 조합으로 통제 가능**.
- 가장 적게 검증된 영역(=연구 여지): **pre-train 단계에서의 think-token implant** (Pause Token, Quiet-STaR만 존재) — 큰 규모에서는 누구도 본격 시도하지 않음. 사용자의 시도가 이 gap을 메울 가능성이 있음.

---

## 부록 A — 참고 URL 목록

(arXiv 외에는 official source)

- Qwen3: https://arxiv.org/abs/2505.09388
- Kimi k1.5: https://arxiv.org/abs/2501.12599
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- L1 / LCPO: https://arxiv.org/abs/2503.04697
- O1-Pruner: https://arxiv.org/abs/2501.12570
- DAST: https://arxiv.org/abs/2503.04472
- ThinkPrune: https://arxiv.org/abs/2504.01296
- Demystifying Long CoT: https://arxiv.org/abs/2502.03373
- Chain of Draft: https://arxiv.org/abs/2502.18600
- TALE: https://arxiv.org/abs/2412.18547
- Sketch of Thought: https://arxiv.org/abs/2503.05179
- CoT-Valve: https://arxiv.org/abs/2502.09601
- Coconut: https://arxiv.org/abs/2412.06769
- C3oT: https://arxiv.org/abs/2412.11664
- Concise CoT (Renze): https://arxiv.org/abs/2401.05618
- Pause Token: https://arxiv.org/abs/2310.02226
- Quiet-STaR: https://arxiv.org/abs/2403.09629
- Compressed CoT: https://arxiv.org/abs/2412.13171
- RouteLLM: https://arxiv.org/abs/2406.18665
- ThinkSwitcher: https://arxiv.org/abs/2505.14183
- AdaCoT: https://arxiv.org/abs/2505.11896
- Ada-R1: https://arxiv.org/abs/2504.21659
- Interplay paper: https://arxiv.org/abs/2512.07783
- Thinking Augmented Pre-Training (TPT): https://arxiv.org/abs/2509.20186
- RA3: https://arxiv.org/abs/2509.25810
- Front-Loading Reasoning: https://arxiv.org/abs/2510.03264
- OctoThinker: https://arxiv.org/abs/2506.20512
- Mid-training Survey: https://arxiv.org/abs/2510.23081
- DEER: https://arxiv.org/abs/2504.15895
- Certaindex / Dynasor: https://arxiv.org/abs/2412.20993
- CAR: https://arxiv.org/abs/2505.15154
- Magistral: https://arxiv.org/abs/2506.10910
- Nemotron Nano 2: https://arxiv.org/abs/2508.14444
- DeepSeek-V3.1: https://huggingface.co/deepseek-ai/DeepSeek-V3.1
- Claude 3.7 Sonnet: https://www.anthropic.com/news/claude-3-7-sonnet
- GPT-5 reasoning_effort: https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_new_params_and_tools
- Gemini 2.5 thinking: https://ai.google.dev/gemini-api/docs/thinking
- s1: https://arxiv.org/abs/2501.19393
- MRT: https://arxiv.org/abs/2503.07572
- LIMR: https://arxiv.org/abs/2502.11886
- LIMO: https://arxiv.org/abs/2502.03387
- Short-RL / Length-Aware: https://arxiv.org/abs/2505.12284
- Don't Overthink It Survey: https://arxiv.org/abs/2508.02120
- Hawkeye: https://arxiv.org/abs/2504.00424
- Markovian Thinker: https://arxiv.org/abs/2510.06557
- Light-R1: https://arxiv.org/abs/2503.10460
- Soft Thinking: https://arxiv.org/abs/2505.15778
- SelfBudgeter: https://arxiv.org/abs/2505.11274
- LASER: https://arxiv.org/abs/2505.15612
- Thinking Fast and Right: https://arxiv.org/abs/2505.18298
- AutoThink: https://arxiv.org/abs/2505.10832
- SmolLM3 blog: https://huggingface.co/blog/smollm3
- Reasoning on a Budget Survey: https://arxiv.org/abs/2507.02076
- Concise & Adaptive Survey: https://arxiv.org/abs/2507.09662
- ALP: https://arxiv.org/abs/2506.05256
- ASRR: https://arxiv.org/abs/2505.15400
- ODIN: https://arxiv.org/abs/2402.07319
- HiPO: https://arxiv.org/abs/2509.23967
- ADR: https://arxiv.org/abs/2510.10207
- Train Long Think Short: https://arxiv.org/abs/2508.08940
- DAPO: https://arxiv.org/abs/2503.14476
- e1: https://arxiv.org/abs/2510.27042
- GLM-4.5: https://arxiv.org/abs/2508.06471
- AdaptThink: https://arxiv.org/abs/2505.13417
- Is It Thinking or Cheating? (TRACE): https://arxiv.org/abs/2510.01367
- ProRL: https://arxiv.org/abs/2505.24864

### 한국 LLM (§11.4)
- HyperCLOVA X THINK: https://arxiv.org/abs/2506.22403
- EXAONE Deep: https://arxiv.org/abs/2503.12524
- EXAONE 4.0: https://arxiv.org/abs/2507.11407
- Mi:dm K 2.5 Pro: https://arxiv.org/abs/2603.18788
- Trillion 7B: https://arxiv.org/abs/2504.15431

### 2026년 1-5월 신작 (§11.5)
- Thinking-Based Non-Thinking: https://arxiv.org/abs/2601.04805
- Path-Lock Expert: https://arxiv.org/abs/2604.27201
- Reasoning Primitives in Hybrid LLMs: https://arxiv.org/abs/2604.21454
- Tackling Length Inflation (GR3): https://arxiv.org/abs/2603.10535
- Stepwise Penalization: https://arxiv.org/abs/2603.00296
- Compress Easy Explore Hard (CEEH): https://arxiv.org/abs/2602.22642
- CODA: https://arxiv.org/abs/2603.08659
- Conformal Thinking: https://arxiv.org/abs/2602.03814
- Variational Posterior Guidance: https://arxiv.org/abs/2605.11019
- LEAD: https://arxiv.org/abs/2605.09806
- Ares: https://arxiv.org/abs/2603.07915
- ROM: https://arxiv.org/abs/2603.22016
- Reasoning Path Deviation Monitoring: https://arxiv.org/abs/2603.14251
- Precedent-Informed Reasoning: https://arxiv.org/abs/2602.14451
- DiPO (Difficulty-aware RL Overthinking): https://arxiv.org/abs/2601.21418
- ThinkRouter: https://arxiv.org/abs/2602.11683
- R2-Router: https://arxiv.org/abs/2602.02823

---

## 부록 B — 검증 메모 (2026-05-26)

본 보고서는 사용자 요청으로 한 차례 cross-check를 거쳤음:
1. **arXiv ID 88건 전수 검증**: 모두 실재 확인. 날조 0건. (의심받던 2512.07783, 2510.06826 등 모두 실재.)
2. **영어 verbatim 인용 24건 spot-check**: 모두 원 paper와 일치. 인용 충실도 매우 높음.
3. **별칭 정정**: §4.12 AutoThink, §4.15 ASRR, §4.16 Short-RL은 본 보고서 부여 약칭이며 paper 정식 제목이 다름 — 각 섹션에 정식 제목 병기.
4. **추가 보강 항목**: §11.4 한국 LLM 5건, §11.5 2026년 1-5월 신작 19건 추가 (모두 검증 완료).

---

*보고서 끝.*
