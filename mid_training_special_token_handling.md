# Mid-Training 데이터 — Reasoning/Chat 데이터의 Special Token 후처리 여부

조사 일자: 2026-05-26
조사 범위: 기존 보고서의 모든 출처(60+) + 2025-09 ~ 2026-05 신규 발표 모델/논문
조사 질문(단일):

> **"Mid-training 데이터셋을 만들 때 reasoning이나 chat 데이터셋을 raw 그대로 사용하는지, 아니면 special token을 후처리하는지."**

각 논문/모델/데이터셋에 대해 다음을 정직하게 기록한다:
- (a) 어느 stage에서 처리하는지 (mid-train / annealing / CPT / SFT)
- (b) 데이터를 raw concat / plain-text marker / chat template wrap / new special token 중 어떤 방식으로 포맷팅하는지
- (c) Vocab 추가 여부
- (d) Loss masking 여부
- (e) 원문 직접 인용 (가능한 경우)
- (f) 명시 없으면 **"논문에 명시 없음"** 으로 기록 (silence를 추정으로 채우지 않음)

---

## 0. 결론 (한눈에)

조사한 ~40개 사례를 다음 4분류로 정리:

| 분류 | 정책 | 대표 사례 (Stage 단계 기준) |
|---|---|---|
| **① 그대로 사용 (raw concat / plain text, 후처리 없음)** | special token 없이 plain text로 이어붙임. 가장 명시적으로 검증된 mid-training 표준. | OLMo 2 (Dolmino, mid-train), PRISM (2026, mid-train), OctoThinker (mid-train, OpenR1을 `<think>` 그대로 plain concat), Llama 3 annealing (instruction 미투입), Apple AFM CPT (instruction 미투입), SmolLM2 Stage 4 (instruction 미투입), Phi-4 mid-train (instruction 미투입), Qwen3 Stage 2 (instruction 미투입) |
| **② 후처리: chat template / channel token wrap** | mid-train 단계에서부터 ChatML, channel token (assistant/think, Harmony analysis/commentary/final), 또는 `<think>` 태그를 명시적으로 적용 | SmolLM3 reasoning mid-train (ChatML), HyperCLOVA X THINK (channel token), Mi:dm K 2.5 Pro (Harmony 변형), Kanana-2 thinking (DeepSeek-R1 호환), KORMo (`<think>` + 명시적 loss mask), HyperCLOVA X 32B Think Stage 4 (`<think>`) |
| **③ Vocab에 신규 토큰 추가** | embedding 새로 학습 (가장 강한 후처리). 거의 유일한 사례. | Quiet-STaR (`<\|startofthought\|>` / `<\|endofthought\|>` 신규 학습, em-dash로 init) |
| **④ 명시 없음 (silent)** | instruction/reasoning 데이터를 mid-train에 넣었다고만 하고 포맷팅 디테일 비공개. **이 분류가 가장 많다 — 약 절반**. | MiniCPM decay, Granite 3.0 Stage 2, Nemotron-H Phase 4, Nemotron-4 340B continued, GLM-4.5, MAP-Neo decay, Yi-Lightning, Hunyuan-Large, InternLM2, Mi:dm 2.0, MiMo Stage 3, NVIDIA Front-Loading Reasoning, Llama-Nemotron Ultra CPT (단 CPT 자체에는 think tag 안 씀이 명시) |

**가장 중요한 발견 3가지**:

1. **명시적 "raw concat" 정책의 가장 확실한 인용은 OLMo 2 (Olmo 3 블로그)** — *"all instruction and reasoning data avoids using templates or special tokens during midtraining ... text formatting is adopted, which maintains the pretrained model's output format."* SFT 단계에서만 ChatML 적용.
2. **2026년 발표된 PRISM 논문이 통제 실험으로 raw concat을 명시** — *"For all reasoning datasets and chat data, we concatenate the question and answer with a single line break between them"*. `User:`/`Assistant:` prefix만 사용, special token 없음.
3. **Mid-training에서 think 토큰 loss를 명시적으로 mask한 거의 유일한 사례는 KORMo** — *"During this stage, however, the `<think>` token was excluded from loss computation. This design choice prevents semantic interference in the subsequent SFT stage."* 한국 모델 중 가장 명시적.

---

## 1. Mid-training / Annealing / CPT 단계에 reasoning 데이터를 명시적으로 투입한 사례

### 1.1 OctoThinker — `<think>` plain text concat (vocab 변경 X)

- 출처: https://arxiv.org/abs/2506.20512
- Stage: Mid-training "Stable-then-Decay" — Stable 200B (constant LR) → Decay 20B × 3 branch (Short-CoT / Long-CoT / Hybrid)
- 포맷: OpenR1 데이터를 question + thinking + `<think>...</think>` 로 line break concat. 일반 instruction은 `User:{}\nAssistant:{}` plain text. **ChatML 미사용.**
- Vocab: 명시 없음 (기본 Qwen/Llama tokenizer 그대로 → BPE로 sub-token 처리 추정)
- Loss mask: 명시 없음 (standard NTP 추정)
- 인용: *"For the OpenR1 dataset, we concatenate the question and the thinking process enclosed within `<think>` and `</think>` using a line break."*
- 결론: **분류 ① (raw concat) — `<think>` 마커는 텍스트로 보존하되 vocab/template 후처리 없음**

### 1.2 SmolLM3 reasoning mid-training — ChatML + wrapped packing (가장 명시적 후처리)

- 출처: https://huggingface.co/blog/smollm3
- Stage: Reasoning mid-training 140B 토큰 (OpenThoughts3-1.2M + Llama-Nemotron-Post-Training v1.1의 R1 trace subset, 4 epoch)
- 포맷: **ChatML chat template + wrapped packing** (system prompt 없음)
- Vocab: 명시 없음 (ChatML 토큰은 base tokenizer에 이미 정의된 것으로 추정)
- Loss mask: mid-training은 **명시 없음** (packed LM-style로 보임). SFT 단계에서만 user turn + tool call result에 mask.
- 인용: *"For the reasoning mid-training we used the ChatML chat template (so no system prompt)"* / *"We used the ChatML chat template and wrapped packing to avoid providing too much structure to the model."*
- 결론: **분류 ② (chat template wrap)** — 사용자 use case에 가장 가깝게 검증된 recipe

### 1.3 KORMo — `<think>` + mid-training에서 think token loss mask (한국 모델 중 가장 명시적)

- 출처: https://arxiv.org/abs/2510.09426
- Stage: Pretraining 2-stage + **Mid-training 2단계** (Long-Context + **Reasoning Context Training**)
- 포맷: **각 sample을 `<think>...</think>` 토큰으로 wrap.** Pretraining 단계가 아닌 mid-training 전용 처리.
- Vocab: EPK-125K (125,184 BPE). `<think>` 신규 토큰화 여부 직접 명시는 없으나 special token 처리 시사.
- Loss mask: **YES — 가장 명시적.** *"the `<think>` token was excluded from loss computation. This design choice prevents semantic interference in the subsequent SFT stage, where the model will be explicitly trained to treat the `<think>` token as a reasoning trigger signal."*
- 결론: **분류 ② (special token wrap) + mid-train loss mask 명시** — KORMo는 한국 fully-open 모델 중 mid-training 포맷 디테일을 가장 명확히 공개

### 1.4 Quiet-STaR — vocab에 신규 학습 토큰 추가 (가장 강한 후처리, 유일한 사례)

- 출처: https://arxiv.org/abs/2403.09629
- Stage: Continued pre-training (Mistral 7B 위)
- 포맷: rationale 앞뒤에 **신규 학습 가능 special token** `<|startofthought|>`, `<|endofthought|>` 삽입
- Vocab: **YES.** em-dash `—` embedding으로 init, gradient weight 1e2
- Loss mask: rationale 자체에는 NLL 안 걸고, REINFORCE로 "thought가 미래 토큰 예측을 얼마나 개선했는가"를 보상
- 인용: *"We insert learned `<|startofthought|>` and `<|endofthought|>` tokens to mark each rationale's start and end."* / *"initialize the start and end token embeddings to the embedding corresponding to the em dash, '—'"*
- 결론: **분류 ③ — 조사한 모든 사례 중 유일하게 vocab에 reasoning 전용 신규 학습 토큰 추가**

### 1.5 Llama-Nemotron Ultra — CPT에는 `<think>` 미사용, SFT부터만 도입

- 출처: https://arxiv.org/abs/2505.00949
- Stage: **CPT 88B 토큰** (Nemotron-H Phase 4 dataset 위) + 65B KD → SFT → RL
- 포맷: **CPT 단계에서는 `<think>` tag 사용 안 함** (논문 본문에서 thinking tag 언급은 SFT 이후에만). SFT부터 `"detailed thinking on/off"` system prompt와 함께 `<think>...</think>` 도입.
- Vocab: 명시 없음 (`<think>` 일반 토큰 처리로 보임)
- Loss mask: CPT는 standard LM loss 추정 (명시 없음)
- 인용: *"LN-Ultra is first trained with knowledge distillation for 65B tokens... followed by 88B tokens of continued training on the Nemotron-H phase 4 pretraining dataset."*
- 결론: **CPT는 분류 ① (raw 그대로), reasoning 포맷팅은 SFT부터** — 사용자가 mid-train에 reasoning 데이터를 안 넣는 선택을 한다면 가장 가까운 reference

### 1.6 NVIDIA Front-Loading Reasoning — 포맷 디테일 명시 없음

- 출처: https://arxiv.org/abs/2510.03264
- Stage: Pretraining 1T 토큰 중 20%를 reasoning (Nemotron-Pretraining-SFT-v1 336B)
- 포맷: **논문 본문에 명시 없음.** Special token / chat template / role wrapper 여부 silent. 평균 10K+ 토큰 long-CoT를 강한 teacher로 합성한다고만 기술.
- 결론: **분류 ④ (silent)** — *"SFT만으로 복구 불가능한 +19% 이득"* 만큼 강한 주장에 비해 포맷 디테일 비공개

### 1.7 MiMo-7B Stage 3 — 포맷 디테일 명시 없음

- 출처: https://arxiv.org/abs/2505.07608
- Stage: Pre-training Stage 3 (~10% synthetic reasoning responses, 25T 중 마지막)
- 포맷: **논문에 명시 없음.** `<think>` 마커, ChatML, role tag 일체 언급 없음. 데이터 생성 방법(STEM 콘텐츠 prompt)만 기술.
- 인용: *"synthetic reasoning data can be trained for extremely high number of epochs without overfitting risk"*
- 결론: **분류 ④** — 포맷팅 디테일 의도적 비공개로 보임

### 1.8 Phi-4-reasoning — SFT 단계, ChatML + `<think>...</think>` 명시

- 출처: https://arxiv.org/abs/2504.21318
- Stage: **SFT** (mid-training 아님)
- 포맷: **ChatML** 위에 `<think> {Thought} </think> {Solution}` 구조
- Vocab: Phi-4 tokenizer에 placeholder token이 미리 마련되어 있고 vocab size 200,064까지 확장 가능. `<think>` 자체를 신규 학습 토큰으로 추가했는지는 본 논문 명시 없음.
- 인용: *"Please structure your response into two main sections: Thought and Solution using the specified format: `<think> {Thought section} </think> {Solution section}`."*
- 결론: **분류 ② (SFT 단계의 chat template wrap)**

### 1.9 s1 / s1.1 — SFT, ChatML role 확장

- 출처: https://arxiv.org/abs/2501.19393
- Stage: **SFT** (1K samples × 5 epoch, Qwen2.5-32B-Instruct)
- 포맷: ChatML 위에 새로 정의한 `<|im_start|>think ... <|im_start|>answer` 패턴 (즉 chat template에 신규 role 추가)
- Vocab: 신규 vocab token 추가는 명시 없음. role 이름만 새로 씀.
- Loss mask: 코드 (GitHub `sft.py`) 기준 **prompt 토큰에도 loss 계산하는 옵션 존재** (issue #72) — 일반적 SFT mask 가정과 반대. 논문 본문은 silent.
- 결론: **분류 ② (chat template role 확장)** — loss mask가 일반과 다르다는 점이 특이

### 1.10 DeepSeek-R1 — distillation 소스로서의 포맷 (vocab/mask 본문 silent)

- 출처: https://arxiv.org/abs/2501.12948
- Stage: SFT(cold start) + RL
- 포맷:
  - **R1-Zero RL template**: *"The reasoning process and answer are enclosed within `<think>` `</think>` and `<answer>` `</answer>` tags"*
  - **R1 cold-start SFT 포맷**: `|special_token|<reasoning_process>|special_token|<summary>` — placeholder가 실제로 special token으로 등록되었음을 시사
- Vocab: 본문 명시 없음. HF DeepSeek-R1 토크나이저에는 `<think>`/`</think>`가 실제 special token으로 등록되어 있음 (HF config 확인).
- Loss mask: 본문 silent
- 결론: **분류 ② (special token)** — 다만 distillation 시 사용자가 R1 출력을 재포맷할지는 별도 결정

---

## 2. 일반 Base 모델 Tech Report — Mid-training/Annealing의 instruction 포맷

### 2.1 OLMo 2 (Dolmino Mix 1124) — "no template, plain text" 명시

- 출처: https://arxiv.org/abs/2501.00656 + https://huggingface.co/datasets/allenai/dolmino-mix-1124
- Stage: Midtraining / annealing phase
- 포맷: **raw text concat / no template.** FLAN 17B 토큰(16.6%)을 chat template 없이 plain text로 concat. 보조 출처 (Olmo 3 blog) 인용: *"all instruction and reasoning data avoids using templates or special tokens during midtraining due to the complexity that this additional formatting introduces into the evaluation process. Instead, text formatting is adopted, which maintains the pretrained model's output format."*
- Vocab: mid-training 중 추가 명시 없음
- Loss mask: 명시 없음 (standard NTP)
- 결론: **분류 ① — 가장 명시적인 raw concat 정책**. ChatML wrap은 Tülu 3 SFT 단계에서만 적용.

### 2.2 MiniCPM (WSD decay) — 포맷 명시 없음

- 출처: https://arxiv.org/abs/2404.06395
- Stage: WSD decay phase
- 포맷: UltraChat / SlimOrca / OssInstruct / EvolInstruct를 mix한다고만 기술. **chat template 적용 여부 silent.**
- 인용: *"In the decay stage, we use a mixture of the pretraining data and high-quality SFT data"*
- 결론: **분류 ④** — 후속 추측은 가능하나 본문 명시는 없음

### 2.3 Granite 3.0 (IBM) — Stage 2 instruction 10%, 포맷 silent

- Stage 2에 instruction data 10% 포함. Granite chat template (`<|start_of_role|>`)은 post-training용으로 별도 정의되며, **Stage 2 instruction 데이터가 이 template으로 wrap되었는지는 본문 명시 없음**.
- 결론: **분류 ④**

### 2.4 GLM-4.5 — 본문 fetch 실패

- 출처: https://arxiv.org/abs/2508.06471
- abstract만 확보. *"mid-training... including instruction data"* 라는 명시는 있으나 chat template/special token 디테일은 PDF 추출 실패로 검증 불가.
- 결론: **확인 실패 (사실상 분류 ④)**

### 2.5 Nemotron-H — 230B "SFT-style tokens" but 포맷 silent

- 출처: https://arxiv.org/abs/2504.03624
- Stage: 4-phase data blending. Phase 4 = 마지막 380B 토큰. **별도로 전체 학습에 230B synthetic SFT-style tokens (174B math + 35B code + 21B general) 분산 주입.**
- 포맷: "SFT-style"이라는 표현뿐, chat template/role marker 사용 여부 silent.
- 결론: **분류 ④**

### 2.6 Nemotron-4 340B — "QA-style alignment examples" but 포맷 silent

- 출처: https://arxiv.org/abs/2406.11704
- Continued training에 "small number of question-answering style alignment examples" 추가. **role marker / chat template 사용 여부 명시 없음.**
- (참고: 별도 SFT 단계는 *"mask the user turns and only calculate loss on assistant turns"* 명시)
- 결론: **분류 ④** (continued PT에서는 silent)

### 2.7 Phi-4 (NOT phi-4-reasoning) — midtraining에 instruction 사실상 미사용

- 출처: https://arxiv.org/abs/2412.08905
- Mid-training 250B = 30% newly curated long-context + 70% recall. **Long-context 자연 장문이 핵심**. Instruction-formatted 데이터의 midtraining 명시적 사용은 발견 못함.
- 결론: **분류 ① (instruction 미투입)**

### 2.8 Apple AFM — continued PT는 math/code only

- 출처: https://arxiv.org/abs/2407.21075
- *"continued pre-training at a sequence length of 8192, with another 1T tokens from a mixture that upweights math and code"*. Dialogue-style data는 post-training으로 분리.
- 결론: **분류 ① (instruction 미투입)**

### 2.9 Llama 3 — annealing에 instruction 미투입

- 출처: https://arxiv.org/abs/2407.21783
- Annealing 40M은 *"upsamples high-quality data in select domains"*, *"small amounts of high-quality code and mathematical data"*. **명시적 instruction/chat 데이터 사용 언급 없음.** GSM8k training set도 의도적으로 제외.
- 결론: **분류 ① (instruction 미투입)**

### 2.10 SmolLM2 Stage 4 — instruction 미사용

- 출처: https://arxiv.org/abs/2502.02737
- Stage 4 (10T → 11T): math 14% + code 24% + web 58% + synthetic 4% (Cosmopedia v2). **Instruction-following data는 post-training (SmolTalk)으로 분리.**
- 결론: **분류 ① (instruction 미투입)**

### 2.11 Kimi K2 — pre-train과 post-train 완전 분리

- 출처: https://arxiv.org/abs/2507.20534
- Annealing 400B@4K + 60B@32K. Pre-training 단계 instruction 사용 언급 없음. SFT는 별도.
- 결론: **분류 ① (instruction 미투입)**

### 2.12 Hunyuan-Large — 1.5T synthetic 있으나 포맷 silent

- 출처: https://arxiv.org/abs/2411.02265
- 7T 중 1.5T synthetic. Synthetic 파이프라인은 (1) instruction generation → (2) evolution → (3) response gen → (4) filter의 instruction-response pair 형태. **Annealing 단계 chat template/wrap 명시 없음.**
- 결론: **분류 ④**

### 2.13 MAP-Neo — decay에 instruction, 포맷 silent

- 출처: https://arxiv.org/abs/2405.19327
- Decay phase (778B)에서 *"high-quality instruction data"* + "SFT와 유사한 hyperparameter"라는 표현. **포맷 명시 없음.**
- 결론: **분류 ④**

### 2.14 Yi-Lightning — "early instruction-tuning adaptation" but 포맷 silent

- 출처: https://arxiv.org/abs/2412.01253
- Fast-decay (12.5% of total)에서 *"incorporates early instruction-tuning adaptation"*. 구체적 포맷 silent.
- 결론: **분류 ④**

### 2.15 Qwen3 Stage 2 — instruction-formatted 미사용

- 출처: https://arxiv.org/abs/2505.09388
- *"To further improve the reasoning ability, we optimize the pre-training corpus of this stage by increasing the proportion of STEM, coding, reasoning, and synthetic data."* — 즉 raw pre-training tokens. `/think`, `/no_think` flag와 ChatML은 post-training부터.
- 결론: **분류 ① (instruction 미투입)**

### 2.16 InternLM2 — capability-specific enhancement, 포맷 silent

- 출처: https://arxiv.org/abs/2403.17297
- "Capability-Specific Enhancement Training" 단계에 instruction 데이터 시사 (GSM8k 36 → 71 jump). `<|interpreter|>`, `<|plugin|>` special token은 chat/agent용으로 정의되나 enhancement 단계 적용 여부 명시 없음.
- 결론: **분류 ④**

---

## 3. 한국 LLM (8종) — Mid-training 시점 reasoning/chat 포맷

### 3.1 HyperCLOVA X THINK (Naver, 2025-06)

- 출처: https://arxiv.org/abs/2506.22403
- Stage: 3-stage pre-train (128K context) + post-train (SFT + RLVR). **Mid-training 단계는 abstract에서 명시되지 않음** (본문 fetch 실패로 정밀 확인 한계).
- 포맷: **채널 토큰 방식**. `<|im_start|>assistant/think\n{reasoning}<|im_end|>\n<|im_start|>assistant\n{final}<|im_end|><|endofturn|>`. `force_reasoning` / `skip_reasoning` 토크나이저 플래그.
- Vocab: **YES** — `<|im_start|>`, `<|im_end|>`, `<|endofturn|>`, `<|stop|>` + channel prefix 신규 도입
- Loss mask: 명시 없음
- 결론: **분류 ② (channel token)**

### 3.2 HyperCLOVA X 32B Think (Naver, 2026-01)

- 출처: https://arxiv.org/abs/2601.03286
- Stage: **4-stage pre-train** — Foundation → Context Extension → Advanced Reasoning & Long-Context → **High-Quality Annealing**
- 포맷: `<|im_start|>assistant`에 이어 `<think>...</think>` XML 스타일 + 답변
- Vocab: 한국어 형태소 기반 vocab adaptation (pruning + substitution) — **YES**
- Loss mask: 명시 없음
- 인용: *"The last stage boosts reasoning ability with a carefully curated high-quality reasoning dataset."*
- 결론: **분류 ② (`<think>` XML wrap)** + annealing 단계 명시

### 3.3 EXAONE 4.0 (LG AI Research, 2025-07)

- 출처: https://arxiv.org/abs/2507.11407
- Stage: SFT → Reasoning RL (AGAPO) → Preference (SFT 데이터는 Reasoning:Non-reasoning = 1.5:1)
- 포맷: **토큰리스 하이브리드** — special token으로 mode 분리 없음. 비율 mixing만으로 통합 학습.
- Vocab: **NO** — *"the same tokenizer and vocabulary as the previous EXAONE 3.5 and Deep models"*
- Loss mask: 명시 없음
- 인용: *"Rather than fine-tuning the two modes sequentially, we combine both modes and train them together."*
- 결론: **분류 ① (token-less, SFT 단계 비율 mixing)** — 분류 ①의 가장 극단적 케이스

### 3.4 EXAONE Deep (LG AI Research, 2025-03)

- 출처: https://arxiv.org/abs/2503.12524
- Stage: Base 위 SFT → DPO → Online RL (mid-training 명시 없음)
- 포맷: **XML 태그 `<thought>...</thought>`** — *"the EXAONE 3.5 Instruct models are trained to engage in reasoning within the `<thought>` and `</thought>` tags, performing step-by-step logical progression along with reflection, self-checking, and correction."*
- Vocab: 명시 없음
- Loss mask: 명시 없음
- 결론: **분류 ② (XML tag wrap)** — `<think>`가 아닌 `<thought>` 사용 (R1 계열과 비호환)

### 3.5 Mi:dm 2.0 (KT, 2026-01)

- 출처: https://arxiv.org/abs/2601.09066
- Stage: 3-stage pre-train (Stage1 8B → **Stage2 DUS 11.5B + ultra-HQ data** → Stage3 long-context)
- 포맷: LongCoT dataset 사용 언급은 있으나 **special token / channel / chat template 형식 본문에 명시 없음**. SFT에는 reasoning trace 제외하고 최종 답만 학습.
- Vocab: 한국어 최적화 custom tokenizer. reasoning 특수 토큰 추가 언급 없음
- Loss mask: 명시 없음
- 결론: **분류 ④ (silent on mid-train format)**

### 3.6 Mi:dm K 2.5 Pro (KT, 2026-03) — Harmony 변형 channel token

- 출처: https://arxiv.org/abs/2603.18788
- Stage: **CPT Stage 0~2** (annealing 격) + Long-context 3-stage (4K → 128K) + Reasoning SFT + RLVR + Fusion (SFT + RL)
- 포맷: **OpenAI Harmony 변형 channel token (analysis / commentary / final)**. 멀티턴에서 직전 reasoning만 보존.
- Vocab: **1-digit 단위 숫자 토크나이제이션**으로 변경. Channel token도 chat template 변경의 일부로 추가.
- Loss mask: 명시 없음
- 인용: *"explicitly separate internal reasoning traces from externally visible outputs using channel tokens (analysis, commentary, final) with distinct roles"* / *"we use a chat template based on harmony chat format, but apply some modifications considering training efficiency and reasoning control."*
- 결론: **분류 ② (channel token)** — 한국 모델 중 가장 적극적인 후처리

### 3.7 Trillion 7B (Trillion Labs, 2025-04)

- 출처: https://arxiv.org/abs/2504.15431
- Stage: **2-stage pretraining with annealing** — 1.8T 고LR → 0.2T LR decay
- 포맷: Annealing에서 데이터 품질 강화/mixture 변경. **추론 특수 토큰 / chat template / channel 본문 명시 없음.**
- Vocab: 128,256 byte-level (영어 100K + 한국어 24,552 + 기타). 별도 reasoning token 추가 없음
- Loss mask: **YES (SFT 한정)** — *"we only apply the loss on responses."* MTP auxiliary loss 추가.
- 인용: *"During the annealing phase, we enhance overall data quality and modify the mixture composition."*
- 결론: **분류 ① (annealing은 raw, special token 미언급)**

### 3.8 Kanana (Kakao, 2025-02)

- 출처: https://arxiv.org/abs/2502.18934
- Stage: 2-stage pretraining (Stage1 2.7T + **Stage2 300B with lightweight annealing**)
- 포맷: **본문에 명시 없음** — 이 시점 Kanana는 reasoning 모델이 아님
- 결론: **분류 ④ (silent, but reasoning data 자체가 핵심 아님)**

### 3.9 Kanana-2 30B-a3b-thinking (Kakao, 2025-12) — DeepSeek-R1 호환

- 출처: https://huggingface.co/kakaocorp/kanana-2-30b-a3b-thinking
- Stage: 모델카드는 architecture와 reasoning-parser 중심. Mid-training 디테일 미공개.
- 포맷: **DeepSeek-R1 호환** — `<think>...</think>` 태그. vLLM/SGLang에서 `--reasoning-parser deepseek_r1`로 서빙. Chat template: `<|im_start|>` / `<|im_end|>` 메시지 구분자 + `<think>` / `</think>` reasoning 태그.
- Vocab: 신규 tokenizer (한국어 효율 30%↑, 128,256). Reasoning special token 포함 — **YES**
- Loss mask: 명시 없음
- 결론: **분류 ② (R1 호환 special token)**

### 3.10 KORMo (MLP-Lab, 2025-10) — § 1.3 참조

위 §1.3에 상세 기재. 한국 fully-open 모델 중 mid-training 포맷·loss mask를 가장 명확히 공개한 사례.

---

## 4. 2025-2026 최신 (mid-training 표준화 논문 / 신규 모델)

### 4.1 PRISM (2026) — raw concat을 통제 실험으로 입증

- 출처: https://arxiv.org/abs/2603.17074
- Stage: 7개 base × 4 family (Granite/LLaMA/Mistral/Nemotron-H), 27B HQ token mid-training
- 포맷: *"For all reasoning datasets and chat data, we concatenate the question and answer with a single line break between them"*. 채팅은 `"User:"`/`"Assistant:"` prefix만, special token 없음.
- 결론: **분류 ① (raw concat)** — 2026년 발표된 통제 실험으로 raw concat 정책 입증

### 4.2 RLP — Reinforcement as a Pretraining Objective (NVIDIA, 2025-10)

- 출처: https://arxiv.org/abs/2510.01265
- Stage: Pretraining objective 자체를 변형
- 포맷: 별도 `<think>` 태그 없음, context에 concatenate되는 raw 토큰
- Vocab: 없음
- Loss mask: **YES** — *"updates parameters only through the tokens of the sampled thoughts."*
- 결론: **분류 ① (raw) + 명시적 thought-only loss**

### 4.3 Nemotron Nano 2 (NVIDIA, 2025-08)

- 출처: https://arxiv.org/abs/2508.14444
- Stage: 3-phase curriculum + Phase LC (512K context)
- 포맷: **`<think>` / `</think>` delimiter**. 약 5%는 의도적으로 잘린 reasoning trace — *"About 5% of the data contained deliberately truncated reasoning traces"* / *"Once the budget is reached, the inference setup attempts to insert a closing `</think>` tag."*
- Vocab: 명시 없음
- Loss mask: 명시 없음
- 결론: **분류 ② (`<think>` delimiter)** — budget control용 잘린 trace 데이터까지 큐레이션

### 4.4 Mid-Think (2026-01) — Special token 외에도 trigger token이 mode 결정

- 출처: https://arxiv.org/abs/2601.07036
- 발견: 하이브리드 추론 모델의 think/no-think 전환은 상위 instruction이 아니라 **소수의 trigger 토큰**이 결정. 예: 선두 `"Okay"` 토큰이 reasoning 유발, `"</think>"` 뒤 newline 패턴이 reasoning 억제.
- 함의: 명시적 special token이 없어도 sequence 첫 몇 토큰의 분포가 mode를 결정 → 데이터 큐레이션 시 reasoning 샘플의 prefix 일관성도 chat template 못지않게 중요.

### 4.5 ReMiT — RL-Guided Mid-Training (2026)

- 출처: https://arxiv.org/abs/2602.03075
- Stage: Mid-training (annealing)에서 RL-tuned 모델의 reasoning prior로 token별 weight를 dynamic reweight
- 포맷: 토큰 reweighting이 핵심. 별도 `<think>` 마커보다 sequence 내부 토큰별 가중치가 본질.
- Loss mask: Per-token reweighting 자체가 soft loss mask — *"prioritizing those pivotal for reasoning."*
- 결론: 본문 정밀 확인 한계, **soft mask 류**

### 4.6 Reinforcement Mid-Training (RMT, 2025-09)

- 출처: https://arxiv.org/abs/2509.24375
- 원문 PDF 본문 추출 실패. Abstract 기준 reasoning step 길이 제어 위주.
- 결론: **확인 실패 (분류 미정)**

### 4.7 DeepSeek-V3.1 / V3.2 / V4

- 출처: https://huggingface.co/deepseek-ai/DeepSeek-V3.2
- V3.1부터 V3+R1 통합 단일 hybrid. V4는 thinking을 parameter로 격하 (`deepseek-chat` vs `deepseek-reasoner` 엔드포인트).
- 포맷: `<think>` / `</think>` 태그 (R1 계보 유지). V3.1/V3.2/V4 공식 tech report 본문은 직접 확인 한계.
- 결론: **분류 ② (R1 호환)**

### 4.8 Mid-Training of LLMs: A Survey (2025-10)

- 출처: https://arxiv.org/abs/2510.06826
- mid-training 일반 원칙 (데이터 선별, LR 스케줄, context 확장)은 다루나 **reasoning trace 포맷팅 컨벤션이나 loss mask 표준화는 본문에서 다루지 않음.**

---

## 5. Open 데이터셋 카드 — 실제로 어떤 포맷으로 ship되는가

> "데이터셋 자체가 어떻게 ship되느냐"와 "어느 stage에 권장되느냐"를 분리해 정리.

| # | Dataset | Raw 포맷 | Raw에 박힌 special token | 카드 권장 stage | concat OK? |
|---|---|---|---|---|---|
| 1 | OpenThoughts3-1.2M | ShareGPT (`from`/`value`) | 명시 없음 (QwQ traces) | SFT | chat template 필요 |
| 2 | OpenR1-Math-220k | `generations` + `messages` | **`<think>`, `\boxed{}`** | SFT | chat template 필요 |
| 3 | OpenMathReasoning (NVIDIA) | `problem` + `generated_solution` 단일 문자열 | **`<think>`, `\boxed{}`** | SFT | 단일 문자열이라 concat 가능 |
| 4 | Llama-Nemotron-Post-Training v1.1 | messages + output | **`<think>`**, `"detailed thinking on"` system | SFT + RL | chat template 필요 |
| 5 | Nemotron-Pretraining-SFT-v1 | gated (instruction-response 추정) | 명시 없음 | **pretrain family ("SFT-style")** | 카드 미공개 |
| 6 | **Dolmino Mix 1124** | **plain text `text` 필드** | **없음** | **Mid-training (annealing) 명시** | **그냥 concat OK** |
| 7 | Tulu 3 SFT mixture | `messages` (role/content) | 없음 | SFT | chat template 필요 |
| 8 | WildChat-1M | `conversation` (role/content) | 없음 | SFT (instruction-finetuning tag) | chat template 필요 |
| 9 | Magpie-Pro-1M-v0.1 | ShareGPT (추정) | 없음 | SFT | chat template 필요 |
| 10 | OpenHermes-2.5 | ShareGPT | 없음 | SFT | chat template 필요 |
| 11 | UltraChat-200k | `messages` (role/content) | 없음 | SFT + gen ranking | chat template 필요 |
| 12 | SlimOrca | ShareGPT | 없음 | SFT | chat template 필요 |
| 13 | Bespoke-Stratos-17k | system + ShareGPT 2-turn | **`<|begin_of_thought|>`, `\boxed{}`** | SFT | 부분 concat 가능 |
| 14 | **Nemotron-CC / CC-v2** | **parquet plain text** | **없음** | **Pretraining 명시** | **그냥 concat OK** |

### 5.1 핵심 인사이트

1. **Plain-text concat이 그대로 가능한 데이터셋은 단 3개**: Dolmino Mix 1124 (mid-training 명시), Nemotron-CC v2 (pretraining 명시), Nemotron-Pretraining-SFT-v1 (gated).
2. **`<think>`가 raw에 이미 박혀 있는 데이터셋**: OpenR1-Math-220k, OpenMathReasoning, Llama-Nemotron-Post-Training. mid-training에 투입 시 토크나이저에 `<think>`/`</think>`를 special token으로 등록할지 사용자가 결정해야 함.
3. **`<|begin_of_thought|>` 계열 (Sky-T1/Stratos)** 은 Bespoke-Stratos-17k만 사용. R1-style `<think>`와 비호환 — 섞어 쓸 때 wrapper 정규화 필요.
4. **명시적으로 mid-training을 권장한 카드는 Dolmino 단 하나.** 나머지 reasoning/chat 데이터셋은 모두 SFT 권장. mid-training 투입은 사용자 자체 판단이며 카드 보장 범위 밖.
5. **WebFetch 실패**: Magpie-Pro-1M-v0.1, `nvidia/nemotron-cc`(원본). 자매 데이터셋/v2 카드로 대체 확인.

---

## 6. 최근 트렌드 (2025-09 ~ 2026-05) 요약

조사한 최신 사례에서 다음 4가지 흐름이 관찰된다.

### 6.1 흐름 ① "Raw concat" 정책이 통제 실험으로 확립되는 중

- **OLMo 2 → PRISM (2026)** 흐름: 두 연구 모두 *"mid-training에서는 chat template/special token 후처리하지 말고 plain text로 concat하라"* 가 명시. PRISM은 7 base × 4 family 통제 실험으로 이를 *권장*까지 정립.
- 사용자 4B 빌더에게 가장 안전한 default.

### 6.2 흐름 ② Channel/Harmony token이 hybrid reasoning의 주류로 부상

- **2025-06 HyperCLOVA X THINK** (assistant/think channel) → **2026-03 Mi:dm K 2.5 Pro** (Harmony analysis/commentary/final). 한국 모델이 이 흐름의 선두.
- GPT-5/o-series가 OpenAI Harmony를 발표하면서 channel token이 사실상 산업 표준에 가까워지는 중.
- 단, 채택 시점은 SFT/post-train이 일반적이며 mid-training부터 channel token을 쓰는 사례는 (KORMo 외에) 드물다.

### 6.3 흐름 ③ DeepSeek-R1 호환 `<think>` 태그가 사실상 업계 표준 distillation 포맷

- DeepSeek-V3.1/V3.2/V4, Kanana-2 thinking, KORMo, OctoThinker, Nemotron Nano 2, HyperCLOVA X 32B Think, Qwen3 모두 `<think>`/`</think>` 태그 사용.
- 단 vocab에 special token으로 등록할지는 모델마다 다름:
  - **명시적으로 vocab 등록**: DeepSeek-R1 (HF config 확인), Kanana-2 thinking
  - **plain text marker로만 사용**: OctoThinker, OLMo 2 (`<think>` 미사용), HyperCLOVA X 32B Think (XML 태그)
- EXAONE Deep만 `<thought>` (R1 비호환)을 선택.

### 6.4 흐름 ④ Mid-training의 loss mask 정책이 거의 silent

- 조사한 mid-training 사례 중 **think 토큰을 명시적으로 mask한 것은 KORMo가 유일**.
- **RLP (2025-10)** 가 그 반대 — thought 토큰에**만** gradient를 흘림 (RL objective).
- 그 외 모든 mid-training 단계는 standard NTP라고 추정만 가능, 본문 명시 없음.

### 6.5 흐름 ⑤ "Token-less hybrid" — special token 후처리를 안 쓰는 EXAONE 4.0 노선

- special token으로 mode를 분리하지 않고 **Reasoning:Non-reasoning = 1.5:1 비율 mixing**만으로 통합 학습.
- 분류 ①의 가장 극단적 케이스. Mid-Think (2026-01) 발견을 고려하면 — sequence 첫 몇 토큰의 분포가 mode를 결정한다는 점 — 이론적으로 정합성 있음.
- 한국 모델 중 EXAONE 4.0이 유일하며, 산업적으로는 minority pattern.

---

## 7. 사용자 (4B 한국 hybrid reasoning) 의사결정 매트릭스

| 결정 | 권장 정책 | Reference |
|---|---|---|
| **Mid-training에 reasoning trace 투입 여부** | (a) 단순 안전: SmolLM2/Phi-4/Apple AFM처럼 mid-train은 math/code/long-context만, reasoning은 SFT로 분리 / (b) 적극 통합: SmolLM3 reasoning mid-train, KORMo, OctoThinker | SmolLM3 (140B, 4 epoch), KORMo (Reasoning Context Training) |
| **Special token 후처리 vs raw** | **default = raw concat (분류 ①)**. mid-training에서 chat template 적용은 OLMo 2/Olmo 3 blog 권장과 반대. SmolLM3는 ChatML wrap을 명시적으로 한 예외. | OLMo 2 Dolmino, PRISM (2026), OctoThinker |
| **`<think>` 처리** | **plain text marker로 그대로 두고 vocab 변경 없음** (BPE가 sub-token 처리). 신규 vocab 토큰 추가는 Quiet-STaR가 유일한 사례이며 cold-start 부담이 큼. | OctoThinker, OpenThoughts3, OpenR1-Math, OpenMathReasoning, Llama-Nemotron-PT-v1.1 |
| **Hybrid mode 분리 방식** | (a) Channel/Harmony (Mi:dm K 2.5 Pro, HyperCLOVA X) — 가장 정교 / (b) `<think>` 태그 (Kanana-2, R1 호환) — 가장 표준 / (c) Token-less ratio mixing (EXAONE 4.0) — 가장 단순 | 3가지 모두 한국 사례 존재 |
| **Mid-training loss mask** | KORMo 패턴 (think 토큰 loss 제외) 또는 standard NTP. SFT 단계에서 user/tool turn mask는 SmolLM3 권장. | KORMo, SmolLM3 |
| **Reasoning 데이터 비율** | 기존 보고서 권고 그대로: instruction/reasoning 1~10% 안전, 20%↑ over-specialization 위험 (Qwen3 경고). NVIDIA Front-Loading은 20% 권장이나 포맷 silent. | Qwen3, NVIDIA Front-Loading |

---

## 8. 정직성 노트 (검증 한계)

- **본문 PDF 확보 실패** (abstract 또는 2차 출처만 확인): GLM-4.5, RMT (2509.24375), ReMiT (2602.03075) 본문, HyperCLOVA X THINK 본문, OpenThoughts Appendix D.4, Kimi K2 본문, DeepSeek-V3.1/V3.2/V4 공식 tech report.
- **2차 출처 의존** (블로그/Medium/Papers Explained 시리즈): OLMo 2 "no template" 정책의 명시적 인용은 Olmo 3 블로그에서만 확인 (원 OLMo 2 논문 본문은 직접 인용 확보 못함).
- **데이터셋 카드 WebFetch 실패**: `Magpie-Pro-1M-v0.1` (auth 401), `nvidia/nemotron-cc` (원본). 자매 데이터셋/v2 카드로 대체 확인.
- 본 보고서는 **silence를 silence로 기록**하는 원칙을 따랐다. "논문에 명시 없음" 표기는 추정/관행으로 채우지 않았으니, 사용자가 실제 구현 시 해당 디테일은 원문/모델 카드/HF config를 직접 확인해야 한다.

---

## 9. 참고문헌 (조사 우선순위 순)

### A. Mid-training 포맷 정책이 명시된 핵심 사례
- OLMo 2 — https://arxiv.org/abs/2501.00656 / Dolmino — https://huggingface.co/datasets/allenai/dolmino-mix-1124 / Olmo 3 blog — https://allenai.org/blog/olmo3
- SmolLM3 — https://huggingface.co/blog/smollm3
- KORMo — https://arxiv.org/abs/2510.09426
- OctoThinker — https://arxiv.org/abs/2506.20512
- PRISM (2026) — https://arxiv.org/abs/2603.17074

### B. Reasoning trace mid-training / pretraining
- Quiet-STaR — https://arxiv.org/abs/2403.09629
- NVIDIA Front-Loading Reasoning — https://arxiv.org/abs/2510.03264
- MiMo-7B — https://arxiv.org/abs/2505.07608
- Llama-Nemotron Ultra — https://arxiv.org/abs/2505.00949
- Nemotron Nano 2 — https://arxiv.org/abs/2508.14444
- RLP — https://arxiv.org/abs/2510.01265
- ReMiT — https://arxiv.org/abs/2602.03075
- RMT — https://arxiv.org/abs/2509.24375
- Mid-Think — https://arxiv.org/abs/2601.07036

### C. Reasoning trace SFT/distillation 포맷 (mid-train에 영향)
- DeepSeek-R1 — https://arxiv.org/abs/2501.12948
- Phi-4-reasoning — https://arxiv.org/abs/2504.21318
- s1 — https://arxiv.org/abs/2501.19393 / GitHub https://github.com/simplescaling/s1
- OpenThoughts — https://arxiv.org/abs/2506.04178
- Qwen3 — https://arxiv.org/abs/2505.09388

### D. 일반 base 모델 tech report
- MiniCPM — https://arxiv.org/abs/2404.06395
- Granite 3.0 — https://www.rivista.ai/wp-content/uploads/2024/10/paper-1.pdf (대체 PDF)
- Nemotron-H — https://arxiv.org/abs/2504.03624
- Nemotron-4 340B — https://arxiv.org/abs/2406.11704
- Phi-4 — https://arxiv.org/abs/2412.08905
- Apple AFM — https://arxiv.org/abs/2407.21075
- Llama 3 — https://arxiv.org/abs/2407.21783
- SmolLM2 — https://arxiv.org/abs/2502.02737
- Kimi K2 — https://arxiv.org/abs/2507.20534
- Hunyuan-Large — https://arxiv.org/abs/2411.02265
- MAP-Neo — https://arxiv.org/abs/2405.19327
- Yi-Lightning — https://arxiv.org/abs/2412.01253
- InternLM2 — https://arxiv.org/abs/2403.17297
- GLM-4.5 — https://arxiv.org/abs/2508.06471
- DeepSeek-V3.2 — https://huggingface.co/deepseek-ai/DeepSeek-V3.2

### E. 한국 LLM
- HyperCLOVA X THINK — https://arxiv.org/abs/2506.22403
- HyperCLOVA X 32B Think — https://arxiv.org/abs/2601.03286
- EXAONE 4.0 — https://arxiv.org/abs/2507.11407
- EXAONE Deep — https://arxiv.org/abs/2503.12524
- Mi:dm 2.0 — https://arxiv.org/abs/2601.09066
- Mi:dm K 2.5 Pro — https://arxiv.org/abs/2603.18788
- Trillion 7B — https://arxiv.org/abs/2504.15431
- Kanana — https://arxiv.org/abs/2502.18934
- Kanana-2 30B-a3b-thinking — https://huggingface.co/kakaocorp/kanana-2-30b-a3b-thinking

### F. 데이터셋 카드 (HuggingFace)
- OpenThoughts3-1.2M, OpenR1-Math-220k, OpenMathReasoning, Llama-Nemotron-Post-Training-Dataset, Nemotron-Pretraining-SFT-v1, Dolmino Mix 1124, Tulu 3 SFT mixture, WildChat-1M, Magpie-Pro-1M-v0.1, OpenHermes-2.5, UltraChat-200k, SlimOrca, Bespoke-Stratos-17k, Nemotron-CC v2

### G. Mid-training 서베이
- A Survey on LLM Mid-Training — https://arxiv.org/abs/2510.23081
- Mid-Training of LLMs: A Survey — https://arxiv.org/abs/2510.06826
