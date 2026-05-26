# Mid-Training 데이터셋 구성 보고서
## HuggingFace Conversation/Reasoning 데이터를 어떻게 변환·통합하는가

조사 기준일: 2026-05-21 (검증 보강: 2026-05-26)
조사 범위: 19개 + **한국 LLM 7개** 모델 테크리포트 + 60편 논문/서베이/데이터셋 카드

핵심 서베이:
- A Survey on LLM Mid-Training — https://arxiv.org/abs/2510.23081
- Mid-Training of LLMs: A Survey — https://arxiv.org/abs/2510.06826

**검증 메모 (2026-05-26)**: arXiv ID 60건 전수 확인 (날조 0건). 정량 claim 14건 spot-check 결과 다음 항목 정정 완료:
- §2.1 OLMo 2: 7B/13B/32B 토큰 사용량 정정 (보고서 원안 오류)
- §2.3 Nemotron-H: Phase 4 = 380B와 synthetic 230B는 **서로 다른 축의 숫자** (보고서 원안이 합산처럼 읽혔음)
- §2.2 Llama 3 annealing: paper 본문이 "by 24.0% and 6.4%"로 모호하나 absolute percentage point 기준 해석이 맞음 (의미 보존, 표기 명시화)
- §2.7 한국 LLM 케이스 + §2.8 비용/규모 비교 표 신규 추가

---

## 0. 한눈에 보는 결론 (TL;DR)

1. **Mid-training은 사실상 모든 토큰에 LM loss를 주는 단계**다. SFT의 user/system mask와 다르다. 이게 conversation 데이터를 SFT용이 아니라 mid-training용으로 쓸 때 가장 중요한 구분점.
2. **`<think>...</think>` 토큰은 vocab에 추가하지 않고 plain text marker로 두는 것이 production 표준**. DeepSeek-R1·Qwen3·Llama-Nemotron·OpenThoughts·OctoThinker 모두 vocab 변경 없음. 예외: Phi-4-reasoning(placeholder token 재활용), Quiet-STaR(신규 학습 토큰).
3. **Conversation → mid-training plain text 변환은 5가지 큰 패턴**: ① FLAN concat, ② plain-text role marker(`Participant 1:`), ③ ChatML 그대로+wrapped packing, ④ Instruction-Augmented M-shot(Instruction Pre-Training), ⑤ rephrase to textbook/QA(WRAP/Nemotron-CC).
4. **권장 비율**: HQ web 50-70% + math 10-25% + code 10-25% + instruction/QA 5-20%. Instruction은 1-10%가 가장 안전, 20%↑는 over-specialization.
5. **검증된 LR 패턴은 두 가지**: (a) cosine decay-to-0의 마지막 5-10%에서 데이터 교체(Llama 3, OLMo 2), (b) WSD의 decay phase에서 교체(MiniCPM, SmolLM3, MAP-Neo).

---

## 0.5. 용어 해설 및 예시

본 보고서에 자주 등장하는 핵심 용어를 카테고리별로 정리.

### A. 학습 단계 용어

**Pre-training (사전학습)**
대량의 raw text(웹/책/코드)를 next-token prediction으로 학습하는 첫 단계. 전체 학습 토큰의 80-95% 차지.
- 예: Llama 3는 15T 토큰 중 처음 14.6T가 일반 pretrain.

**Mid-training (중간학습)**
pre-training과 SFT 사이의 단계. **high-quality / 도메인 특화 데이터**로 specific capability(수학, 코드, 추론) 강화.
- 예: OLMo 2가 Stage 1(일반 web 4T) 끝낸 뒤, Stage 2에서 Dolmino Mix(수학+코드+FLAN 명령)를 50-300B 토큰 추가 학습.

**Annealing (어닐링)**
학습 후반부에 learning rate를 점진적으로 0에 가깝게 떨어뜨리면서 high-quality 데이터로 마무리. Mid-training과 거의 동의어로 쓰임.
- 비유: 금속을 천천히 식혀 결정구조를 안정화하듯, 모델 weight를 안정화.
- 예: Llama 3가 전체 학습 끝에 마지막 40M 토큰에서 LR을 linear하게 0으로 떨어뜨림.

**Continued Pre-training (CPT)**
이미 학습된 모델을 새로운 도메인 데이터로 추가 학습.
- 예: 일반 LLM을 의료 텍스트로 100B 토큰 추가 학습 → 의료 LLM.
- Mid-training과의 차이: mid-training은 같은 학습 plan 안의 단계, CPT는 별도 도메인 적응.

**SFT (Supervised Fine-Tuning)**
모델에 instruction-following을 가르치는 단계. user/assistant 쌍을 보여주고 assistant 응답을 따라하게 함.
- Mid-training과의 핵심 차이: SFT는 user 토큰에 **loss mask** 적용, mid-training은 안 함.

**RL / RLHF / GRPO**
강화학습. 모델 응답에 reward(인간 선호도 또는 정답 매칭)를 주고 reward 높이는 쪽으로 학습.
- 예: DeepSeek-R1이 reasoning RL에서 "정답이면 +1, 형식 맞으면 +1" reward로 학습.

---

### B. 학습 손실 (Loss) 용어

**LM loss / Causal LM / Next-token prediction**
가장 기본 학습 목적함수. t번째 토큰까지 보고 (t+1)번째 토큰을 맞히도록 학습.
- 예: "The cat sat on the ___" 다음 "mat" 확률을 높임.

**Token-level cross-entropy (CE)**
각 토큰 위치에서 모델 예측 분포와 실제 정답 토큰 간의 cross-entropy 손실. LM loss의 구체적 계산 방법.

**Loss masking**
특정 토큰들에는 loss를 계산하지 않음(gradient가 흐르지 않음).
- SFT 예시 (mask O):
  ```
  토큰:    <user> 사과의 영어는? <assistant> apple <eos>
  mask:    [mask][mask][mask][mask][mask][학습][학습][학습][학습]
  ```
- Mid-training 예시 (mask X):
  ```
  토큰:    <user> 사과의 영어는? <assistant> apple <eos>
  mask:    [학습][학습][학습][학습][학습][학습][학습][학습][학습]
  ```
- **이게 보고서에서 가장 중요한 구분점**: mid-training은 모든 토큰을 학습 대상으로 함.

**MTP loss (Multi-Token Prediction)**
다음 1 토큰만 예측하는 대신, 다음 N개 토큰을 한꺼번에 예측하는 보조 loss.
- 예: DeepSeek-V3, MiMo가 사용. inference 시 speculative decoding 가능해져 속도↑.

---

### C. 토크나이저 / 포맷 용어

**Token (토큰)**
모델이 처리하는 텍스트의 최소 단위.
- 예: `"tokenizer"` → `["token", "izer"]` 같은 sub-word로 분리됨.

**Vocab (어휘)**
모델이 알고 있는 모든 token의 집합. 보통 32K-200K개.
- 예: Llama 3 vocab size = 128,256. Qwen3 = 151,936.

**Special token**
Vocab에 명시적으로 추가된 특수 토큰. 일반 텍스트와 구분되는 단일 ID.
- 예: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, `<eos>`, `<bos>`.
- 새로 추가하려면 embedding layer 확장 + 새 임베딩 초기화 필요 → cold start 문제.

**Plain text marker**
특수 토큰이 아닌 일반 텍스트로 구분자 역할. BPE가 알아서 sub-word로 토크나이즈.
- 예: `<think>`는 BPE에서 보통 `["<", "th", "ink", ">"]`처럼 3-4개 일반 토큰으로 분해됨.
- 장점: vocab 변경 불필요. 대부분의 reasoning 모델이 이 방식.

**BPE (Byte-Pair Encoding)**
가장 흔한 tokenization 알고리즘. 자주 등장하는 byte pair를 점진적으로 합쳐서 sub-word vocab을 만듦.

**Chat template**
Conversation 데이터를 모델 입력 형식으로 변환하는 규칙. 모델마다 다름.
- ChatML 예시 (Qwen, OpenAI 계열):
  ```
  <|im_start|>system
  당신은 도움이 되는 어시스턴트입니다.<|im_end|>
  <|im_start|>user
  사과의 영어는?<|im_end|>
  <|im_start|>assistant
  apple입니다.<|im_end|>
  ```
- Llama 3 chat template:
  ```
  <|start_header_id|>user<|end_header_id|>

  사과의 영어는?<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>

  apple입니다.<|eot_id|>
  ```

**Role tag**
대화의 발화자 역할을 표시하는 marker. `<|user|>`, `<|assistant|>` 같은 special token일 수도, `User:`, `Assistant:` 같은 plain text일 수도 있음.

---

### D. 시퀀스 패킹 (Packing) 용어

**Sequence packing**
짧은 시퀀스 여러 개를 하나의 긴 시퀀스로 합쳐 GPU 효율을 높이는 기법.
- 예시: max_len=8192이고 평균 데이터 길이가 500이면, 16개 정도를 EOS로 구분해 packing.
  ```
  [doc1 토큰들] <eos> [doc2 토큰들] <eos> [doc3 토큰들] <eos> ... [doc16 토큰들] <eos>
  ←─────────────────────── 8192 토큰 ──────────────────────────→
  ```

**Wrapped packing**
시퀀스 경계를 신경 쓰지 않고 그냥 줄줄이 이어붙이고 max_len에서 자름. 단순/빠름.
- SmolLM3 mid-training이 사용.

**BFD packing (Best-Fit Decreasing)**
시퀀스들을 길이 순으로 정렬하고 빈 공간을 최대한 잘 채우도록 배치. Padding 낭비 적음.
- SmolLM3 SFT가 사용.

**Cross-document attention mask**
Packing된 시퀀스에서 다른 document 간 attention을 막는 마스크.
- 예시: `[doc1][doc2]`가 packed면, doc2의 토큰이 doc1을 attention하지 못하도록 마스킹.
- Mid-training에서는 종종 생략 (성능 영향 미미). SFT에서는 권장.

**EOS / BOS token**
- `<eos>` (End of Sequence): 시퀀스/document 종료 표시.
- `<bos>` (Begin of Sequence): 시퀀스 시작 표시.

---

### E. Learning Rate (LR) Scheduler 용어

**Cosine decay**
LR이 cosine 곡선처럼 천천히 떨어짐 (시작 빠르고 끝 느림). 가장 흔하게 사용.
- 예시 곡선:
  ```
  LR
  5e-4 |\___
       |    \__
  5e-5 |       \____________
       └─────────────────────→ steps
  ```

**Linear decay**
LR이 직선으로 떨어짐. 보통 cosine 끝에서 0까지 가는 마지막 구간에 사용.
- 예: Llama 3 annealing 마지막 40M 토큰 → linear to 0.

**WSD scheduler (Warmup-Stable-Decay)**
LR을 (1) 처음엔 점진적 증가(warmup) → (2) 한동안 일정 유지(stable) → (3) 마지막에 빠르게 decay하는 방식. MiniCPM이 제안.
- Cosine과의 차이를 그림으로:
  ```
  Cosine:
  LR  5e-4 \____
                \________
                         \________ 5e-5
  
  WSD:
  LR  5e-4   ____________________ (stable)
                                 \
                                  \__
                                     \__ 5e-5
       ←─warmup─→←──── stable ────→←decay→
  ```
- 장점: stable 단계에서 checkpoint를 여러 개 만들고, 각 checkpoint에서 다른 mid-training 데이터로 decay를 시작할 수 있음 (재사용성).

**Power scheduler**
WSD의 일반화. batch size / token 수에 따라 hyperparameter를 자동 조정. IBM Granite가 사용.

**Polyak averaging / Model averaging / Model soup**
학습 중 여러 checkpoint(또는 다른 학습 run의 결과)의 weight를 평균.
- 예시:
  ```
  Run 1 (데이터 순서 A) → checkpoint_A
  Run 2 (데이터 순서 B) → checkpoint_B
  Run 3 (데이터 순서 C) → checkpoint_C
  
  최종 모델 = (W_A + W_B + W_C) / 3
  ```
- OLMo 2가 50B/100B/300B 세 가지 데이터 mix로 학습 후 평균.

---

### F. 컨텍스트 길이 확장 용어

**Context length / Context window**
모델이 한 번에 볼 수 있는 토큰 수. 보통 4K → 32K → 128K 식으로 점진적 확장.

**RoPE (Rotary Position Embedding)**
위치 정보를 rotation matrix로 임베딩에 곱하는 방식. Llama/Qwen/DeepSeek 등 대부분 사용.
- 핵심 파라미터: `base frequency` (보통 10,000 또는 500,000).
- base 값을 키울수록 더 긴 context에 일반화 가능.

**ABF (Adjusted Base Frequency)**
컨텍스트 확장 시 RoPE base를 직접 조정.
- 예: 4K → 32K 확장하려면 base 10K → 500K 정도로.

**YaRN (Yet another RoPE extensioN)**
RoPE를 더 똑똑하게 보간해서 긴 context로 일반화. ABF보다 정교.
- 예: DeepSeek-V3가 32K → 128K로 확장할 때 사용.

**Long-context extension**
모델의 context length를 늘리는 별도 학습 단계. 보통 long document를 75-100% 비율로 up-sample.
- 예 (Qwen3 Stage 3): 75% text가 16K-32K 길이 + 25%가 4K-16K 길이.

---

### G. 모델 아키텍처 / 학습 기법 용어

**MoE (Mixture of Experts)**
큰 모델을 여러 expert(작은 sub-network)로 나누고 token마다 일부 expert만 활성화.
- 예: DeepSeek-V3 = 671B 파라미터 중 한 번에 37B만 활성화.
- 메모리는 크지만 inference cost는 작아짐.

**Distillation / KD (Knowledge Distillation)**
큰 teacher 모델의 soft probability를 student 모델에 학습시키는 것.
- 예: Gemma 2 9B가 27B teacher의 top-256 logits를 따라하도록 학습.
- 일반 next-token loss보다 정보가 풍부함 (정답뿐 아니라 "두 번째로 그럴듯한 답"도 학습).

**NAS (Neural Architecture Search)**
모델 구조를 자동으로 탐색.
- 예: Llama-Nemotron Ultra가 base 모델을 NAS로 최적화 후 distillation.

**FIM (Fill-in-the-Middle)**
코드 학습 목적함수. 텍스트 중간에 빈칸을 만들어 채우게 함.
- 예: 
  ```
  <prefix>def hello():</prefix>
  <suffix>print(x)</suffix>
  <middle>x = "world"\n</middle>
  ```
- 사용: DeepSeek-V3 (rate 0.1), Codex 계열.

---

### H. 데이터 처리 용어

**MinHash dedup**
텍스트의 fingerprint를 만들어 비슷한 문서를 빠르게 찾고 제거.
- 예: 같은 위키 글이 여러 dump에 있을 때 한 번만 남김.
- 보통 1% 미만 데이터 손실.

**FineWeb-Edu classifier**
웹 텍스트의 "교육적 가치"를 1-5점으로 매기는 모델 (Llama-3-70B annotation 기반).
- 예: ≥3점만 남기면 FineWeb 15T 토큰 중 약 1.3T의 고품질 데이터.

**fastText classifier**
가벼운 분류기. 언어 식별/품질 필터링에 자주 사용.

**Replay data**
새 도메인 학습 시 이전 데이터 일부를 섞어주는 것.
- 예: 의료 LLM 학습 시 일반 web 5%를 섞으면 일반 상식 점수 보존.

**Catastrophic forgetting (재앙적 망각)**
새 데이터를 학습하면서 이전에 학습한 능력을 잃어버리는 현상.
- 예: 일반 LLM을 의료 데이터로 100B 학습 → 일반 상식 MMLU 점수 5-10점 하락.
- 대응: replay data, LR re-warming, mid-training data mixing.

**Decontamination (탈오염)**
학습 데이터에서 평가 벤치마크 문제와 비슷한 텍스트를 제거.
- 예: MMLU 문제가 학습 데이터에 있으면 점수 부풀려짐 → 10-gram overlap 매칭으로 제거.
- 도구: `difflib.SequenceMatcher` ratio > 0.5 텍스트 삭제.

---

### I. 추론(Reasoning) 용어

**CoT (Chain-of-Thought)**
모델이 정답 전에 단계별 추론 과정을 생성하는 방식.
- 짧은 예 (단순 CoT):
  ```
  Q: 사과 5개에서 3개 더 사면?
  A: 5 + 3 = 8. 답은 8.
  ```

**Long-CoT / Reasoning trace / Think token**
CoT를 훨씬 길게(수천~수만 토큰) 작성. 보통 `<think>...</think>`로 감쌈. 검산, 자기반성, 재시도가 포함됨.
- DeepSeek-R1 예시:
  ```
  Question: 12 × 13 = ?
  <think>
  12 × 13을 계산하자. 분해하면 12 × 10 + 12 × 3 = 120 + 36 = 156.
  검산: 13 × 12 = 13 × 10 + 13 × 2 = 130 + 26 = 156. 맞다.
  다른 방법: (12+1)(12-1) + 12 = 143 + 13 = 156. 맞다.
  </think>
  Answer: 156
  ```

**Budget forcing**
test time에 think 길이를 인위적으로 제어.
- 너무 길면: `</think>` 강제 삽입 → 답변 단계 강제 진입.
- 너무 짧으면: "Wait" 같은 단어를 삽입 → 더 생각하게 함.

**Cold start**
RL 학습 전에 모델에게 reasoning format을 가르치는 작은 SFT 단계.
- 예: DeepSeek-R1이 RL 전에 수천 개 CoT example로 cold start.

---

### J. 합성 데이터 / 변환 용어

**Synthetic data (합성 데이터)**
LLM이 생성한 학습 데이터. 기존 텍스트 재작성 또는 새로 생성.
- 예: Mixtral-8x7B로 위키 문서를 "초등 교과서 스타일"로 재작성 → Cosmopedia.

**Rephrasing**
기존 raw 텍스트를 다른 스타일로 다시 쓰기. 데이터 다양성/품질↑.
- WRAP의 4가지 스타일:
  - Wikipedia-style: 백과사전 톤으로 재작성
  - QA-format: 질문답변 형식으로 변환
  - Easy text: 초등 수준 단순화
  - Medium text: 중급 톤
- Nemotron-CC의 5가지 prompt:
  1. **Diverse QA Pairs** — 한 문서에서 yes/no, open-ended, multi-choice 등 최대 8 QA 쌍 생성
  2. **Distill** — 핵심 보존, 정보 밀도 높은 요약
  3. **Knowledge List** — 사실/개념/수치를 list로 추출
  4. **Extract Knowledge** — Wikipedia 스타일 재작성
  5. **Wikipedia-style** — 저품질 텍스트 정제용

**Instruction synthesizer**
raw 문서로부터 (instruction, response) pair를 자동 생성하는 학습된 모델.
- 예 (Instruction Pre-Training, Cheng et al.):
  - Input: 웹 문서 1개
  - Output: 그 문서에 대한 8-10개 QA pair
  - 결과: 200M pair 생성, Llama3-8B 성능이 Llama3-70B에 근접.

**Self-Instruct / Evol-Instruct**
- Self-Instruct: seed instruction 175개로 시작해 LLM이 자가 합성으로 52K instruction 생성.
- Evol-Instruct (WizardLM): instruction을 in-breadth(범위)와 in-depth(깊이)로 진화시켜 복잡도 증가.

**Magpie**
Llama-3-Instruct 같은 instruction-tuned 모델에 **pre-query template만 주면** auto-regressive하게 instruction을 스스로 생성하는 기법. 4M pair 자동 생성.
- 예: `<|user|>` 토큰만 prompt로 주면 모델이 "다음 식을 계산해주세요: ..." 같은 instruction을 만들어냄.

---

### K. 보고서에 나오는 주요 데이터셋 카드 한 줄 요약

| 데이터셋 | 무엇 | 출처 |
|---|---|---|
| **FineWeb / FineWeb-Edu** | 15T web token + 교육적 가치 분류기 | HF |
| **DCLM** | DataComp-LM, 240T Common Crawl + 모델 기반 필터 | DCLM |
| **Dolmino Mix 1124** | OLMo 2의 mid-training mix (843B) | AI2 |
| **FLAN** | 수백 task × 10 template instruction collection | Google |
| **OpenHermes** | Nous Research의 conversation/instruction 컬렉션 | HF |
| **ShareGPT** | 사용자가 공유한 ChatGPT 대화 | HF |
| **Tulu 3** | AI2의 fully-open SFT/DPO post-training mix | AI2 |
| **WildChat-1M** | 실제 ChatGPT-style 대화 1M | NeurIPS 2024 |
| **Magpie** | Llama-3-Instruct로 자가합성한 4M instruction | HF |
| **UltraChat** | Mistral 기반 multi-turn instruction 1.4M | HF |
| **OpenThoughts** | R1/QwQ로 distill한 reasoning trace 114K~1.2M | HF |
| **OpenR1-Math** | R1으로 NuminaMath에 trace 추가한 reasoning 데이터 220K | HF |
| **OpenMathReasoning** | NVIDIA, R1/QwQ distilled math reasoning | HF |
| **Nemotron-CC** | NVIDIA, Common Crawl + 5-way rephrasing | NVIDIA |
| **Nemotron-MIND** | NVIDIA, OpenWebMath 위 7-style dialogue 합성 | NVIDIA |
| **Cosmopedia** | Mixtral-8x7B로 합성한 textbook/blog/WikiHow 30M | HF |
| **MathPile / OpenWebMath** | 수학 특화 web corpus | - |

---

## 1. Mid-Training의 정의 (서베이 합의)

두 서베이 (arxiv 2510.23081, 2510.06826) 모두 mid-training을 "pre-training과 post-training 사이의 다리"로 정의:

- 일반(broad) 사전학습 데이터에서 **specialized·high-quality** 데이터로 의도적 전환
- 보통 **전체 FLOPs/토큰의 5-10%**
- 목표: math, code, reasoning, long-context, multilingual 같은 capability를 enhance하면서 base 분포 보존
- **Continued pre-training**과는 구별: mid-training은 base model의 **전이적/계획된 단계**, continued pretraining은 별도 도메인 적응

---

## 2. 모델별 Mid-Training 데이터 구성 (19개 모델)

### 2.1 명시적으로 "mid-training" 용어 사용

#### OLMo 2 (AI2)
- 출처: https://arxiv.org/abs/2501.00656, https://huggingface.co/datasets/allenai/dolmino-mix-1124
- 2-stage. Stage 2가 **Dolmino Mix 1124** (총 843B 토큰의 *dataset 풀*; 모델은 그 중 일부를 추출해 사용).
- Dolmino는 **50B / 100B / 300B** 세 가지 사전 추출 mix size로 배포됨. 모델별 실제 사용량:
  - **7B**: 50B mix × 3 souped runs = **~150B** 토큰 (model card 명시)
  - **13B**: 100B mix × 3 + 300B mix × 1 souped = **~600B** 토큰
  - **32B**: 13B와 동일 recipe (100B × 3 + 300B × 1) = **~600B** 토큰
- 구성: DCLM 50% + FLAN 11-17% + pes2o 6-19% + Math(GSM8K/MetaMath/TinyGSM-MIND/MathCoder2) 11-21% + Wiki/StackExchange.
- **FLAN을 raw text로 그대로 넣음** (chat template 없음, loss masking 없음).
- LR: cosine → linear → 0.

#### GLM-4.5 (Z.ai)
- 출처: https://arxiv.org/abs/2508.06471
- "Unlike traditional pre-training... these training stages utilize medium-size domain-specific datasets, **including instruction data, denoted as mid-training**."
- 23T pre-train + multi-stage mid-training (repo-level code, synthetic reasoning, long-context/agent 데이터).
- **Instruction 데이터를 mid-training 단계에 명시적으로 통합한 흔치 않은 사례**.

#### Granite 3.0 (IBM)
- 2-stage. Stage 2 = 2T mid-training.
- 비율: Web 5% + Domain 11% + Code 10% + Math 10% + Instruction 10% + Multilingual 5% + Academic 5% + Technical 4%.
- Power scheduler.

#### MiniCPM (OpenBMB)
- 출처: https://arxiv.org/abs/2404.06395
- **WSD scheduler 원조**. Decay phase(전체 10%)에 **UltraChat/SlimOrca/OssInstruct/EvolInstruct/LeetCode/K12 SFT 데이터를 raw text로 mix**.
- Ablation에서 "decay-only pretrain + 별도 SFT"보다 "decay에 SFT mix-in"이 우월.

### 2.2 Annealing / Decay phase로 부르는 모델

#### Llama 3 (Meta)
- 출처: https://arxiv.org/abs/2407.21783
- 3-stage: pretrain → long-context (~800B, 8K→128K 6단계) → **annealing (마지막 40M)**.
- Annealing에서 high-quality math/code/reasoning up-sample, LR linear→0, **Polyak averaging**.
- 효과: 8B 모델 GSM8k +24.0%p, MATH +6.4%p (paper 본문은 "by 24.0% and 6.4%" — 절대 percentage point 기준 해석). 405B에서는 negligible.
- **Instruction/conversation은 annealing에 안 들어감** (전부 post-training).

#### MAP-Neo
- 출처: https://arxiv.org/abs/2405.19327
- "decay phase" 용어 명시. 4.5T fundamental + **778B decay** (74.5% English, 8.5% Chinese, 17% code).
- LR cosine → exponential decay.

#### Kimi K2 (Moonshot)
- 출처: https://arxiv.org/abs/2507.20534
- 15.5T 중 첫 10T constant + 5.5T cosine. **Annealing 400B@4K + 60B@32K + YaRN→128K**.
- MuonClip optimizer.

#### Hunyuan-Large (Tencent)
- 출처: https://arxiv.org/abs/2411.02265
- warmup → gradual decay → **annealing** + 별도 long-context. 7T 중 ~1.5T가 synthetic.
- Long-context: 자연 long-context 25% + 일반 pretrain 75%, 32K→256K, RoPE base 1B.

### 2.3 "Continued / Stage-2" 명칭

#### Apple AFM
- 출처: https://arxiv.org/abs/2407.21075
- 교과서적 3-stage: Core 6.3T → **Continued 1T (web down-weight, math/code/licensed up-weight, 8K)** → Context-lengthening 100B (32K, RoPE 6.3M).
- Instruction은 continued에 직접 안 들어감.

#### Nemotron-4 340B (NVIDIA)
- 출처: https://arxiv.org/abs/2406.11704
- 8T pretrain + **1T continued**. Continued = HQ source up-weight + 소량 QA examples + 부진 영역 up-weight.
- 70/15/15 EN/multi/code.

#### Nemotron-H (NVIDIA)
- 출처: https://arxiv.org/abs/2504.03624
- 4-phase blending. **Phase 4는 학습의 *마지막 380B 토큰 윈도우*** (8B 모델 15T / 56B 모델 20T 중). 원문 §2.2.3: *"The fourth phase is performed for the last 380 billion training tokens."*
- **별도로** 전체 학습 코퍼스에 걸쳐 **230B synthetic SFT-style 토큰**이 분산 주입: 174B math (OpenMathInstruct-2 파이프라인) + 35B code (Genetic Instruct) + 21B general knowledge. 원문 §2.2.2 인용. ⚠ Phase 4 = 380B와 synthetic = 230B는 **서로 다른 축의 숫자**이며 합쳐서 "Phase 4 = 610B"로 읽으면 안 됨.
- SFT-style을 chat template 없이 LM loss로 흡수.

#### Llama-Nemotron Ultra (NVIDIA)
- 출처: https://arxiv.org/abs/2505.00949
- NAS + KD 65B + **88B continued PT on Nemotron-H phase-4** + SFT/RL.
- SFT 단계에서 `detailed thinking on/off` system prompt toggle.

### 2.4 Long-context 중심 mid-training

#### DeepSeek-V3
- 출처: https://arxiv.org/abs/2412.19437
- Pretrain 14.8T + **long-context 2-phase mid-training**: 1000step@32K + 1000step@128K (≈333B), YaRN, LR 2.2e-4→2.2e-5→7.3e-6.

#### Qwen3 (Alibaba)
- 출처: https://arxiv.org/abs/2505.09388
- 3-stage. Stage 2 = ~5T STEM/code/reasoning/synthetic, "**accelerated LR decay**". Stage 3 = long-context (75% 16K-32K + 25% 4K-16K).
- Instruction/CoT는 base에 미포함 (Long-CoT Cold Start는 post-train).

#### Phi-4 (Microsoft)
- 출처: https://arxiv.org/abs/2412.08905
- **midtraining 250B**, 30% newly curated long-context + 70% recall. Context 4K→16K, RoPE base 250K.
- LR을 peak/10 (≈3e-5)로 떨어뜨림. **자연 long-context > 인공 padding**.

#### MiniMax-Text-01
- 출처: https://arxiv.org/abs/2501.08313
- 3-stage context: 4K → ... → 1M. 단계마다 long-sequence up-sample.

#### InternLM2 (Shanghai AI Lab)
- 출처: https://arxiv.org/abs/2403.17297
- 3-phase. Phase 3 = ~24B (1% steps) "capability-specific enhancement" — STEM 65% + special domain 8% + HQ 26%. 7B GSM8k 36→71.

#### Yi-Lightning (01.AI)
- 출처: https://arxiv.org/abs/2412.01253
- "initial → mid-training → **fast-decay** (12.5% of total)" + 별도 64K context 20B 단계.

### 2.5 완전 공개된 mid-training 레시피 (사용자에게 가장 유용)

#### SmolLM2 (HuggingFace)
- 출처: https://arxiv.org/abs/2502.02737
- 4-stage + decay. Stage 4 (10-11T) = **58% web + 24% code + 14% math + 4% Cosmopedia v2**. WSD 사용.

#### SmolLM3 (HuggingFace) — 사용자 use case에 가장 가까움
- 출처: https://huggingface.co/blog/smollm3
- 11.2T pretrain + **mid-training 1 (long-context, 100B)** + **mid-training 2 (reasoning, 140B, 4 epoch)**.
- Reasoning mid-training 데이터: **OpenThoughts3-1.2M + NVIDIA Llama-Nemotron-Post-Training-Dataset-v1.1의 R1 trace subset**.
- 포맷: **ChatML chat template + wrapped packing + loss masking 없음 (전체 토큰 LM loss)**. SFT 단계는 user/tool turn에 mask.
- **HF SFT/reasoning 데이터를 mid-training용으로 쓰는 가장 검증된 recipe**.

### 2.6 비교 표 (요약)

| 모델 | 용어 | Mid-train 토큰 | Web% | Math% | Code% | Instruction | Reasoning trace | Loss mask |
|---|---|---|---|---|---|---|---|---|
| OLMo 2 (7B/13B/32B) | mid-training | 150B / 600B / 600B | 51.9 | 10.8 | 1.7 | **FLAN 11.3%** | 없음 | 없음 |
| Llama 3 405B | annealing | 40M + 800B long-ctx | up | up | up | 없음 | 없음 | N/A |
| Qwen3 | stage 2/3 | 5T + 100s B | ↓ | ↑STEM | ↑ | 없음 | 없음 | N/A |
| DeepSeek-V3 | long-ctx mid | ~333B | 일반 | 일반 | 일반 | 없음 | 없음 | N/A |
| Phi-4 | midtraining | 250B | 70% recall | (30%) | (30%) | 없음 | 없음 | 없음 |
| Nemotron-H | phase 4 | 380B+230B synth | 일반 | +174B | +35B | +21B QA | 없음 | 없음 |
| Apple AFM | continued | 1T+100B | ↓ | ↑ | ↑ | 없음 | 없음 | N/A |
| Yi-Lightning | mid+fast-decay | ~12.5% total | 일반 | synth↑ | synth↑ | "early adaptation" | 없음 | 미공개 |
| MAP-Neo | decay | 778B | 74.5 EN | - | 17 | 없음 | 없음 | N/A |
| Hunyuan-Large | annealing | ~1.5T synth | 일반 | synth↑ | synth↑ | 없음 | 없음 | N/A |
| **GLM-4.5** | **mid-training** | multi | filtered | reasoning↑ | repo-level↑ | **포함** | synth 포함 | 없음 |
| **Granite 3.0** | stage 2 | 2T | 5 | 10 | 10 | **10** | 없음 | 없음 |
| InternLM2 | enhancement | ~24B | 26 | STEM65 | 포함 | 없음 | 없음 | N/A |
| **MiniCPM** | WSD decay | 20B (10%) | + | + | + | **UltraChat/SlimOrca** | 없음 | 없음 |
| SmolLM2 | stage 4/decay | 1T | 58 | 14 | 24 | 0 (+4% Cosmopedia) | 없음 | N/A |
| **SmolLM3** | LC + reasoning | 100B+140B | mix | up | up | reasoning 포함 | **OpenThoughts3+R1** | **없음** |
| Kimi K2 | annealing | 460B | 일반 | 일반 | 일반 | 없음 | 없음 | N/A |
| MiniMax-01 | 3-phase ctx | - | 일반 | 일반 | 일반 | 없음 | 없음 | N/A |
| Nemotron-4 340B | continued | 1T | reweighted | up | up | 소량 QA | 없음 | 없음 |

**굵게** 표시된 모델이 **instruction/conversation 데이터를 mid-training에 명시적으로 통합한 사례**.

### 2.7 한국 LLM의 Mid-training 사례 (사용자 직결)

> 한국어 hybrid reasoning LLM을 from-scratch로 만든다면 가장 직접 참고할 사례들. 비용·데이터 sourcing·hybrid 구현 패턴 모두 한국적 제약(한국어 web 부족, 영어 모델과의 격차) 안에서 작동.

#### Trillion 7B (Trillion Labs, 2025-04) — ★ 비용/규모 reference point
- 출처: https://arxiv.org/abs/2504.15431
- **명시적 2-stage**. Stage 1: 1.8T @ constant LR. **Stage 2 (annealing): 0.2T @ LR 10%로 decay**, 영어 top-20%/한국어 **top-10% 품질 필터**.
- **총 학습 59.4K H100-hours / $148K** — 한국 빌더에게 가장 현실적인 budget reference.
- **Cross-lingual Document Attention (XLDA)**: 한 시퀀스에 다국어 문서를 혼합 packing하되 cross-document attention을 **차단하지 않음** → 영→한 지식 전이를 attention 레벨에서 강제. **Emergence-based proxy modeling**(1.8B/100B로 hyperparameter sweep 후 scale-up). MTP loss.
- 한국어 ≈ 10% (EN:KO:기타 ≈ 8.5:1:0.5).

#### HyperCLOVA X THINK (Naver, 2025-06) — ★ 하이브리드 reasoning 가장 정교
- 출처: https://arxiv.org/abs/2506.22403
- **3-stage curriculum, ~6T 한·영 토큰**. Context 단계적으로 **128K**까지 확장.
- **Reasoning trace 처리**: chat template에 명시적 think 채널 (`<|im_start|>assistant/think ... <|im_end|>`). `force_reasoning` / `skip_reasoning` flag로 hybrid 토글. Prompt에 `"Think for maximum 1024 tokens"` 형태의 **Length Controllability**를 학습.
- **Multi-turn에서 직전 turn의 reasoning을 다음 prompt에 포함하지 않음** (context 절약 정책).
- Post-train pipeline: SFT → RLVR → LC → RLHF+RLVR joint.
- **Peri-LN (Peripheral Layer Normalization) + μP** 조합으로 hyperparameter transfer 안정화. Pruning+KD로 14B를 **Qwen2.5-14B 대비 약 91배 저비용** (68K A100-h)에 학습.

#### EXAONE 4.0 / Deep (LG AI Research, 2025) — ★ Token-less hybrid
- 출처: EXAONE 3.5 https://arxiv.org/abs/2412.04862 / EXAONE Deep https://arxiv.org/abs/2503.12524 / EXAONE 4.0 https://arxiv.org/abs/2507.11407
- **EXAONE 3.5**: 명시적 two-stage. 32B 6.5T / 7.8B 9T / 2.4B 6.5T. **Stage 2 = long-context 강화 + replay 기반 catastrophic forgetting 방지**.
- **EXAONE 4.0**: 32B 14T / 1.2B 12T. Context 4K→32K→128K 두 번의 long-context FT.
- **★ Hybrid reasoning 구현**: special token **없이** Reasoning/Non-Reasoning SFT를 **단일 데이터셋에 1.5:1 비율로 섞어 동시 학습**. 모드 구분은 generation length/temperature로 처리. Qwen3/SmolLM3의 `/think /no_think` 데이터 분리 방식과 정반대의 단순화.
- **EXAONE Deep**: `<thought>...</thought>` XML 태그. 12B tokens (1.6M instances) SFT + 20K DPO + 10K Online RL. **SimPER (DPO 변형) + GRPO 변형**.
- **AGAPO** (Asymmetric Sampling + Global Advantage PO): PPO clipping 제거로 exploratory token 보존, all-incorrect sample에 음의 작은 reward 유지, sequence-level KL penalty. GRPO를 한국식으로 변형한 가장 구체적인 공개 사례.
- 어휘 한·영 ≈ 50:50. 학습 코퍼스 비율은 비공개.

#### Kanana (Kakao, 2025-02) — ★ OLMo-2식 micro-annealing 가장 근접
- 출처: https://arxiv.org/abs/2502.18934, Kanana-2 https://huggingface.co/kakaocorp/kanana-2-30b-a3b-thinking
- **명시적 two-stage staged pre-training**. **Stage 1: 2.7T 토큰 (다양·중품질)**, **Stage 2: 300B 토큰 (고품질 큐레이션)**.
- **"Lightweight annealing experiments"** 로 Stage 2 후보 데이터셋을 사전 검증 → ablation으로 최종 mixture 결정. 한국 LLM 중 **OLMo-2식 microanneal**과 가장 유사.
- **DUS는 staged PT 종료 이후** 적용 (8B→9.8B, 26.8B→32.5B). 각 upscaled 모델에 추가 Stage1 100B + Stage2 100B. 총 컴퓨트 약 11% 절감.
- Pruning + Distillation (Minitron 개선)으로 **2.1B 모델을 0.3T만으로** from-scratch 3T 대비 성능 유지.
- **흥미로운 관찰**: *"영어 데이터 품질 개선이 한국어 벤치마크 점수까지 끌어올린다"* — 한국어 빌더가 한국어 데이터에만 집중하는 함정 경계.
- **Kanana-2 30B-a3b-thinking** (2025-12): 한국 최초 오픈소스 MoE thinking 모델.

#### Mi:dm 2.0 / Mi:dm K 2.5 Pro (KT, 2026-01/03) — ★ 가장 최신
- 출처: Mi:dm 2.0 https://arxiv.org/abs/2601.09066, Mi:dm K 2.5 Pro https://arxiv.org/abs/2603.18788
- **Mi:dm 2.0**: 3-stage. Stage-1 (8B from scratch) → **Stage-2: DUS 8B→11.5B + ultra-high-quality data** → Stage-3 long-context 8K→32K.
- **Mi:dm K 2.5 Pro (32B)**: 3-stage CPT. **Stage 0** (alignment after depth scaling) → **Stage 1** (29% EN / 17% KO / 50% STEM / 9% multilingual) → **Stage 2** (STEM 비중 증가, reasoning 강화). Long-context 4K→32K→64K→**128K** 3-phase.
- **Reasoning trace**: Mi:dm 2.0는 LongCoT 데이터셋 구축 후 **SFT에는 reasoning trace 제외하고 최종 답만** 학습. K 2.5 Pro는 본격적 **channel tokens (analysis / commentary / final)** 도입.
- **★ Multi-turn 정책**: 마지막 reasoning만 유지, 이전 trace는 폐기 (HyperCLOVA X THINK와 동일).
- Pipeline: Reasoning SFT → 모델 merging → Reasoning RL → Fusion SFT → Fusion RL.
- 한국어 데이터 ≈ 85.7% organic + ~14% synthetic. AIHub/NIKL 약 10% 명시.
- **유니크 기법**: **숫자 1-digit unit tokenization** (수학 reasoning 정밀도↑), **Layer predictor 기반 DuS**, **AST 기반 코드 실행성 필터**, **Asynchronous RL** (variable-length reasoning에 유리).

#### KORMo (MLP-Lab, 2025-10) — ★ Synthetic 데이터 비중 검증
- 출처: https://arxiv.org/abs/2510.09426
- 10.8B from-scratch.
- **한국어 부분의 68.74%가 synthetic** — 비영어 fully-open LLM 중 합성 비중 가장 명시적으로 높음. 핵심 주장: *"large-scale pretraining에서 synthetic data가 collapse나 불안정성을 일으키지 않는다"*.
- 한국어 corpus 부족이 bottleneck인 빌더에게 reference로 가장 중요.

#### Solar Pro 2 (Upstage), A.X 4.0 (SK Telecom) — 공개정보 빈약
- Solar: **Depth Up-Scaling (DUS)** 가 원조 — 기존 7B/14B에 layer를 duplicate/concat 후 continued PT. arXiv https://arxiv.org/abs/2312.15166. Solar Pro 2부터 *"Chat Mode" vs "Reasoning Mode"* 사용자 토글 (포맷 미공개).
- A.X 4.0: Qwen2.5 base에 대규모 한국어 CPT (토큰 수 비공개). 공식 기술문서 부재, 학술 reference로 약함.

#### 한국 LLM 종합 관찰 (빌더 시사점)

1. **Mid-training 정의의 한국 수렴**: 거의 모든 신모델이 "Stage 2 = 고품질·도메인 강화 + long-context 확장"으로 수렴. **고전적 LR-annealing**을 명시한 사례는 Trillion 7B (0.2T @ 10% LR + top-품질 필터)뿐. Kanana가 OLMo-2식 microanneal에 가장 가까움.
2. **Hybrid reasoning 양대 구현 패턴**:
   - **Special token + channel**: HyperCLOVA X THINK (`/think` 채널), EXAONE Deep (`<thought>`), Mi:dm K 2.5 Pro (analysis/commentary/final)
   - **Token-less data mixing**: EXAONE 4.0 (1.5:1 비율 단일 학습, 모드 분리 없음)
   - **공통 multi-turn 정책**: 이전 turn의 reasoning trace를 다음 prompt에 포함하지 않음 (context 절약).
3. **한국어 data sourcing 3대 패턴**:
   - **합성 비중 상한**: KORMo (68.74%) > Mi:dm 2.0 (~14%)
   - **AIHub + NIKL** = 한국 기관 데이터 정설
   - **영어 품질이 한국어 점수까지 견인** (Kanana 관찰): 영어 데이터 품질에 투자할 가치
4. **저비용 reference**: Trillion 7B = **2T tokens / 59.4K H100-h / $148K**. HyperCLOVA X THINK 14B = **68K A100-h** (Qwen2.5-14B 대비 91배 저비용). 둘 다 **proxy modeling** 또는 **Peri-LN+μP** 같은 hyperparameter 안정화 기법 활용.
5. **RL 알고리즘 한국식 변형**: EXAONE 4.0 **AGAPO**, EXAONE Deep **SimPER+GRPO 변형**, Mi:dm K 2.5 Pro **Asynchronous RL**. DeepSeek-R1식 vanilla GRPO를 그대로 쓴 한국 모델은 없음 — 모두 안정성·길이·언어 일관성을 위해 변형.
6. **Length Controllability**가 한국 모델 특유의 강조점 (HyperCLOVA X THINK, EXAONE 4.0). 사용자/시스템 프롬프트로 thinking budget 제어를 학습 시점부터 implant.

### 2.8 Mid-training 비용 / 토큰 / 비율 비교 (가용 데이터)

> ⚠ 대부분의 기술보고서가 GPU-hour를 공개하지 않음. 토큰 수와 *비율(% of pretrain)* 위주로 정리. GPU/비용 공개 사례는 ★ 표시.

| 모델 | Pretrain 총량 | Mid-training 토큰 | 비율 | 공개 컴퓨트 | 데이터 mix 핵심 |
|---|---|---|---|---|---|
| **Llama 3 405B** | ~15.6T | annealing 40M + long-ctx 800B | 0.0003% + 5.1% | — | HQ math/code up-sample + Polyak avg |
| **OLMo 2 7B** | ~4T | 150B (50B×3 souped) | 3.8% | — | Dolmino: DCLM 50 + FLAN 11 + math 11-21 |
| **OLMo 2 13B / 32B** | ~5T | 600B (100B×3 + 300B×1 souped) | 12% | — | 동상 |
| **DeepSeek-V3** | 14.8T | ~333B (1000step×2 long-ctx) | 2.2% | — | 일반 + YaRN long-ctx |
| **DeepSeek-V3.1** | V3 base | 32K phase 630B + 128K phase 209B = **839B** | — | — | long-ctx만 (V3 위에) |
| **Qwen3** | ~36T | Stage 2 ~5T + Stage 3 ~수백B | 13-15% | — | STEM/code/reasoning ↑ |
| **Phi-4** | ~10T | 250B | 2.5% | — | 30% long-ctx 신규 + 70% recall |
| **Apple AFM** | 6.3T core | 1T continued + 100B context | 17.5% | — | Web ↓, math/code/licensed ↑ |
| **Nemotron-4 340B** | 8T | 1T continued | 12.5% | — | HQ up-weight + 소량 QA |
| **Nemotron-H 8B** | 15T | 380B (phase 4) | 2.5% | — | + 230B synth SFT 전체 분산 |
| **MAP-Neo** | 4.5T | 778B decay | 17.3% | — | 74.5% EN + 8.5% ZH + 17% code |
| **SmolLM2** | ~11T | Stage 4 1T | 9% | — | 58 web + 24 code + 14 math + 4 Cosmopedia |
| **SmolLM3** | 11.2T | LC 100B + reasoning 140B = **240B** | 2.1% | — | OpenThoughts3 + Llama-Nemotron R1 |
| **Granite 3.0** | 10T | Stage 2 2T | 20% | — | web 5 + domain 11 + code 10 + math 10 + instr 10 |
| **MiniCPM** | varies | decay ~10% of total | 10% | — | UltraChat/SlimOrca mixed |
| **Kimi K2** | 15.5T | 400B@4K + 60B@32K annealing | 3% | — | constant LR 10T + cosine 5.5T |
| **Hunyuan-Large** | 7T | ~1.5T synth + annealing | 21% | — | synthetic 비중 큼 |
| **MiMo-7B** | 25T | Stage 3 (10% synthetic reasoning) | ~10% | — | + MTP loss 0.1 |
| **Llama-Nemotron Ultra** | NAS base | 65B KD + 88B continued | — | — | Nemotron-H phase-4 데이터 재사용 |
| **GLM-4.5** | 23T | multi-stage (정확치 비공개) | — | — | repo-level code + synth reasoning + agent |
| **OctoThinker** | Llama-3 base | 200B stable + decay 3-branch | — | — | short/long/hybrid CoT 분기 |
| **Kanana 8B/30B** | 2.7T | Stage 2 300B + DUS 추가 200B | 11-19% | — | KO web (FastText-filtered) + 영어 HQ |
| **EXAONE 4.0 32B** | 14T | 별도 stage 미명시 (long-ctx 2회 FT) | — | — | 한·영 50:50 vocab |
| **Trillion 7B** ★ | 2T | annealing 0.2T @ 10% LR | 10% | **59.4K H100-h / $148K** | EN top-20% + KO top-10% 필터 |
| **HyperCLOVA X THINK 14B** ★ | ~6T 한·영 | 3-stage curriculum (분량 비공개) | — | **68K A100-h** (Qwen2.5-14B 대비 91×↓) | synthetic Korean 강화 |
| **Mi:dm K 2.5 Pro 32B** | 비공개 | Stage 0+1+2 + LC 3-phase | — | — | 50% STEM + KO/EN/multi |
| **KORMo 10.8B** | 비공개 | (전체 stage 미공개) | — | — | KO 부분의 68.74%가 synthetic |

#### 비용/규모 관찰
- **Mid-training 토큰 비율은 보통 2~20%** (전체 pretrain 대비). 너무 작으면(< 1%) annealing 수준으로 효과 제한적, 너무 크면(> 20%) over-specialization 위험 (Qwen3가 instruction 20%↑ 경고).
- **GPU-hour 공개는 한국 빌더가 가장 적극적** — Trillion 7B와 HyperCLOVA X THINK가 ML 학계 보기 드문 수준의 비용 투명성. 두 사례를 직접 비교 가능한 reference로 삼을 것.
- **Annealing 자체는 매우 저렴**: Llama 3 40M, Kimi K2 460B, Trillion 0.2T — 전체의 0.03~3% 수준. mid-training 효과 대비 cost가 가장 좋은 단계.
- **Long-context extension이 가장 비싼 단계**: DeepSeek-V3.1만 봐도 839B 토큰을 long-ctx에만 투입. **mid-training 전체 budget의 절반 이상**을 long-context에 쓰는 경우 흔함.

---

## 3. Think 토큰(`<think>...</think>`)의 처리

### 3.1 두 갈래

**A. Plain-text marker로 처리 (다수파)** — vocab 변경 없음

| 모델 | 처리 방식 | 출처 |
|---|---|---|
| DeepSeek-R1 | 시스템 프롬프트 템플릿에 `<think>...</think>` 명시, RL의 format reward로 강제 | https://arxiv.org/abs/2501.12948 |
| Qwen3 thinking mode | chat template 내 plain text. `enable_thinking=False`로 빈 thinking block. `/think`, `/no_think` flag도 plain text | https://arxiv.org/abs/2505.09388 |
| OpenThoughts | R1 template 또는 SkyT1 template(`<|begin_of_thought|>...<|end_of_thought|>`) — 성능 동등 | https://arxiv.org/abs/2506.04178 |
| Llama-Nemotron | "explicitly prompting for reasoning steps enclosed in `<think>` tags". vocab 추가 X | https://arxiv.org/abs/2505.00949 |
| OctoThinker | question + `<think>...</think>`를 line break으로 concat한 plain text로 mid-training | https://arxiv.org/abs/2506.20512 |
| OpenMathReasoning(NVIDIA) | `generated_solution` 필드에 `<think>...</think>` XML-like 블록 포함된 단일 문자열 | https://arxiv.org/abs/2504.16891 |
| Nemotron-CrossThink | `<think>...</think>` + `\boxed{...}` 응답 포맷. Reward = `R_acc ∧ R_format` | https://arxiv.org/abs/2504.13941 |

**B. Special token으로 처리 (예외)**

| 모델 | 처리 방식 | 출처 |
|---|---|---|
| Phi-4-reasoning | base model의 unused placeholder token ID를 `<think>`/`</think>` 의미로 재할당. RoPE base frequency 2배→32K | https://arxiv.org/abs/2504.21318 |
| s1/s1.1 | ChatML 기존 special token reuse — `<|im_start|>think ... <|im_end|>` / `<|im_start|>answer`. Budget forcing으로 think 길이 제어 | https://arxiv.org/abs/2501.19393 |
| Quiet-STaR | **신규 학습 가능한** `<|startofthought|>`, `<|endofthought|>` 토큰 추가. em-dash 임베딩으로 init, gradient weight 1e2 | https://arxiv.org/abs/2403.09629 |

**핵심**: 대규모 production 모델 대부분(R1, Qwen3, Llama-Nemotron, OpenThoughts 계열)이 **vocab에 새 token을 추가하지 않는다**. BPE가 `<think>`/`</think>`를 1~3개 sub-token으로 토크나이즈하고, 자주 등장하면서 사실상 학습된 marker가 된다.

### 3.2 Mid-training (post-SFT 아닌)에서 reasoning trace 학습 사례

| 모델/논문 | 어떻게 | 출처 |
|---|---|---|
| **OctoThinker** | "Mid-training Incentivizes RL Scaling". Llama-3 base에 OpenR1-Math 220K(`<think>...</think>` 포함)를 plain text concat 후 standard LM loss. 후속 RL의 scaling 효과 증대 입증 | https://arxiv.org/abs/2506.20512 |
| **NVIDIA Front-Loading Reasoning** | 1T pretrain의 20%를 reasoning Q-A로 채움(OpenThoughts-style + Nemotron-Pretraining-SFT-v1). Standard LM loss. **SFT만으로 복구 불가능한 +19% 이득** | https://arxiv.org/abs/2510.03264 |
| **MiMo-7B** | 25T pretrain의 Stage 3(annealing)에 ~10% synthetic reasoning + 8K→32K. "Synthetic reasoning data는 다회 epoch에도 overfit 안 됨" | https://arxiv.org/abs/2505.07608 |
| **Llama-Nemotron Ultra** | 65B distillation + 88B continued PT, 이후 SFT에서 R1 trace 학습 | https://arxiv.org/abs/2505.00949 |
| **Quiet-STaR** | Mistral 7B에 OpenWebMath/C4로 continued pretraining + REINFORCE로 thought token 효용 학습 | https://arxiv.org/abs/2403.09629 |
| **SmolLM3 reasoning mid-train** | OpenThoughts3 + Llama-Nemotron R1 trace를 ChatML+wrapped packing+LM loss(no mask)로 140B 토큰(4 epoch) | https://huggingface.co/blog/smollm3 |

### 3.3 Loss masking 정책 (Think vs Answer)

거의 모든 사례에서 **think와 answer에 동일 가중치 CE**. 차등 가중치 사례는 보고되지 않음. 차이는 prompt/question 부분 처리에서 발생:

| 모델 | Question | Think | Answer | 비고 |
|---|---|---|---|---|
| s1/s1.1 | mask | 정상 CE | 정상 CE | 명시적으로 prompt masking |
| DeepSeek-R1 (SFT) | 표준 mask | 정상 CE | 정상 CE | RL은 acc+format reward |
| Kimi K1.5 | — | RL | RL | think+answer 통합 시퀀스, length penalty |
| Phi-4-reasoning | — | 정상 CE | 정상 CE | think/answer 동일 |
| OpenThoughts | — | 정상 CE | 정상 CE | 표준 SFT |
| OctoThinker (mid) | — | 정상 CE | 정상 CE | Q+think+answer concat LM loss |
| NVIDIA Front-Loading | — | 정상 CE | 정상 CE | Q-A pretrain inject |
| MiMo Stage 3 | — | 정상 CE | 정상 CE | + MTP loss 0.1 |
| Quiet-STaR | NLL | REINFORCE 별도 | NLL + mixing | 가장 복잡한 케이스 |

**유의사항** (Llama-Nemotron 강조):
- **Sequence-length-dependent token loss averaging** 함정: 평균 시퀀스 길이가 16K-32K로 길어지면 토큰당 effective LR이 작아짐 → reasoning SFT에는 LR을 일반 SFT보다 높임(1e-5~8e-5).
- **Sequence packing**: 큰 데이터셋에서만 적용, 짧은 데이터셋은 packing 생략.

### 3.4 Think trace 길이(distribution shift) 대응

| 기법 | 출처 |
|---|---|
| Reasoning data 도입 시 context length 확장: MiMo 8K→32K, Qwen3 32K, Phi-4-reasoning RoPE base 2배→32K | MiMo, Qwen3, Phi-4-reasoning |
| Higher LR로 long-sequence averaging 보상 | Llama-Nemotron |
| Budget forcing: `<|im_end|>` 강제 삽입(상한), "Wait" 삽입(하한) | s1 |
| RL length penalty: `λ = 0.5 - (len-min)/(max-min)` | Kimi K1.5 |
| Thinking budget 시스템 프롬프트 | Qwen3 |
| Answer-Length Filtered subset (`D_ALF`, >4096 token) | NVIDIA Front-Loading |
| Partial rollouts for RL | Kimi K1.5 |
| Self-reflection 제거 시 성능 -49.1% | OpenThoughts |

---

## 4. Conversation 데이터 변환 5가지 패턴

### 패턴 A — Pure Concatenation (FLAN-style)
```
You are a helpful assistant.

What is the capital of France?

The capital of France is Paris.

What about Germany?

Germany's capital is Berlin.
```
**사용**: OLMo 2 (Dolmino Mix의 FLAN 17B 그대로), FLAN-T5, T0.
**장점**: 가장 단순, tokenizer 변경 없음.
**단점**: role 경계 약함.

### 패턴 B — Plain-text Role Marker (Nemotron-MIND style)
```
Participant 1: What is the capital of France?
Participant 2: The capital of France is Paris.
Participant 1: What about Germany?
Participant 2: Germany's capital is Berlin.
```
**사용**: Nemotron-MIND (DEBATE/PROBLEM-SOLVING/TEACHER-STUDENT 등 7가지), WRAP, Cosmopedia, MAmmoTH2 WebInstruct.
**장점**: tokenizer 변경 불필요, document boundary 자연스러움. **mid-training과 가장 호환**.
**단점**: chat-style special token은 SFT에서 별도 학습 필요.

### 패턴 C — ChatML + Wrapped Packing (SmolLM3/Phi-4 style)
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
...
```
**사용**: SmolLM3, Phi-4 mid-training, OLMoE-Instruct, Llama-Nemotron.
**장점**: SFT와 동일 포맷 → downstream fine-tune 효율↑. Role 정보 보존.
**단점**: special token이 base에 없으면 embedding 초기화 필요.

### 패턴 D — Instruction-Augmented Raw Text (Cheng et al.)
```
<bos> [raw web doc] <eos>
Q: What is the capital of France?
A: The capital of France is Paris.
<eos>
Q: What about Germany?
A: Germany's capital is Berlin.
<eos>
```
**사용**: Microsoft Instruction Pre-Training (synthesizer 200M pair 합성, Llama3-8B≈70B 성능)
출처: https://arxiv.org/abs/2406.14491
**장점**: 자연 분포 보존하며 instruction signal 주입. CPT에 가장 효과적.
**단점**: synthesizer를 따로 학습/운영.

### 패턴 E — Rephrase to Pretraining Styles (WRAP/Nemotron-CC)
**Nemotron-CC의 5가지 prompt**:
1. Diverse QA Pairs (한 문서에서 최대 8 QA)
2. Distill (밀도 높은 요약)
3. Knowledge List (사실/개념/수치 list)
4. Extract Knowledge (Wikipedia 스타일 재작성)
5. Wikipedia-style (low quality용)

**사용**: WRAP, Nemotron-CC (Llama 3.1 8B MMLU +5.6 over DCLM), Phi-4, Kimi K2, Hunyuan-Large.

---

## 5. 사용자가 바로 쓰는 변환 파이프라인

ShareGPT / OpenHermes / Tulu / Magpie / WildChat → mid-training 전용 데이터로 변환하는 단계별 가이드.

### Step 1 — 필터링
- [ ] Language filter (fastText, FineWeb-Edu 방식)
- [ ] Turn count: 3 ≤ turns ≤ 10 (WildChat-1M 기준)
- [ ] User dedup (한 user >100 conv는 program-generated 의심)
- [ ] Benchmark decontamination (10-gram overlap + `SequenceMatcher` ratio>0.5)
- [ ] MinHash dedup
- [ ] FineWeb-Edu classifier score ≥ 3 retain
- [ ] PII/toxicity check (특히 WildChat)

### Step 2 — 포맷 선택 (의사결정 트리)
1. Base 모델에 ChatML special token이 이미 있나? → **Yes: 패턴 C**
2. SFT까지 갈 계획, mid부터 chat 형식 학습? → **Yes: 패턴 C + embedding init**
3. Domain knowledge 주입이 주 목적? → **패턴 B or E**
4. Pretraining 분포에 가장 가깝게? → **패턴 A or D**

### Step 3 — Mix 비율
- Instruction/conversation 데이터: **전체의 1-10%** (안전). 절대 20% 넘기지 말 것(Qwen3 경고).
- OLMo 2 Dolmino식: web 50% + 기타 50%.
- MiniCPM WSD식: stable에 raw 위주, decay phase(~10% tokens)에 SFT mix.
- Llama 3식: 마지막 40-100B에 high-quality + instruction up-sample.
- **Replay data**: 이전 분포 일부 섞기 (catastrophic forgetting 방지, Ibrahim et al. 2403.08763).

### Step 4 — Loss & LR
- **Mid-training: 모든 토큰 LM loss** (절대 mask하지 말 것).
- LR: WSD decay 또는 cosine→0 annealing. Phi-4식 peak LR의 1/10이 안전한 default.
- Re-warming/re-decaying (Ibrahim et al.) — 새 분포 학습 시.

### Step 5 — Packing & Tokenization
- Wrapped packing (mid-training) / BFD packing (SFT).
- EOS로 dialogue 종결.
- Cross-document attention mask: mid-train optional, SFT 권장.
- Context length: base 모델에 맞춤 (4K). Long-context는 별도 stage.

### Step 6 — Reasoning trace 데이터 (OpenThoughts/Open-R1/R1 distilled) 처리
- `<think>...</think>`는 plain text marker로 두기 (vocab 변경 X).
- ChatML wrap + wrapped packing + **mask 없음** (SmolLM3 recipe).
- LR을 일반 mid-training보다 약간 높게 (3e-5~1e-4) — long sequence averaging 보상.
- 다회 epoch도 안전 (MiMo 보고).

### Step 7 — 모델 averaging (선택)
- OLMo 2: 같은 stage-1 checkpoint에서 데이터 순서 다른 3개 학습 후 평균.
- Llama 3: annealing 중 여러 checkpoint Polyak averaging.

### 위험 신호 체크
- Synthetic instruction >10-20% → over-specialization (Qwen3).
- ShareGPT raw 그대로 → turn-confusion → role marker 명시 필요.
- ChatML special token을 mid-train에서 처음 도입 → embedding init/LR 조심.
- Reasoning trace에서 self-reflection 제거 → -49% (OpenThoughts).

---

## 6. 참고문헌 60편

### A. Mid-training / Annealing 서베이·핵심
1. A Survey on LLM Mid-Training — https://arxiv.org/abs/2510.23081
2. Mid-Training of LLMs: A Survey — https://arxiv.org/abs/2510.06826
3. MiniCPM (WSD scheduler) — https://arxiv.org/abs/2404.06395
4. Simple and Scalable Continual Pretraining (Ibrahim et al.) — https://arxiv.org/abs/2403.08763
5. Reuse, Don't Retrain (NVIDIA Parmar) — https://arxiv.org/abs/2407.07263
6. Llama 3 Herd — https://arxiv.org/abs/2407.21783
7. OLMo 2 Furious — https://arxiv.org/abs/2501.00656
8. Dolmino Mix 1124 — https://huggingface.co/datasets/allenai/dolmino-mix-1124
9. OLMoE — https://arxiv.org/abs/2409.02060
10. Apple AFM — https://arxiv.org/abs/2407.21075
11. DCLM / DataComp-LM — https://arxiv.org/abs/2406.11794
12. FineWeb / FineWeb-Edu — https://arxiv.org/abs/2406.17557
13. ProLong — https://arxiv.org/abs/2410.02660
14. Phi-4 — https://arxiv.org/abs/2412.08905
15. Llama-Nemotron — https://arxiv.org/abs/2505.00949
16. Power Scheduler — https://arxiv.org/abs/2408.13359

### B. Instruction-for-Pretraining
17. Instruction Pre-Training (Cheng et al.) — https://arxiv.org/abs/2406.14491
18. WRAP: Rephrasing the Web — https://arxiv.org/abs/2401.16380
19. Nemotron-CC — https://arxiv.org/abs/2412.02595
20. MAmmoTH2 / WebInstruct — https://arxiv.org/abs/2405.03548
21. MIND: Math Informed syNthetic Dialogues — https://arxiv.org/abs/2410.12881
22. Cosmopedia — https://huggingface.co/blog/cosmopedia
23. Phi-1 / Textbooks Are All You Need — https://arxiv.org/abs/2306.11644
24. Magpie — https://arxiv.org/abs/2406.08464
25. Hunyuan-Large — https://arxiv.org/abs/2411.02265
26. Kimi K2 — https://arxiv.org/abs/2507.20534
27. DeepSeek-V3 — https://arxiv.org/abs/2412.19437
28. DeepSeek-R1 — https://arxiv.org/abs/2501.12948
29. Qwen3 — https://arxiv.org/abs/2505.09388
30. Genetic Instruct — https://arxiv.org/abs/2407.21077
31. Arctic-SnowCoder — https://arxiv.org/abs/2409.02326

### C. Reasoning / Think Token
32. OctoThinker (Mid-training for RL Scaling) — https://arxiv.org/abs/2506.20512
33. NVIDIA Front-Loading Reasoning — https://arxiv.org/abs/2510.03264
34. MiMo-7B — https://arxiv.org/abs/2505.07608
35. Quiet-STaR — https://arxiv.org/abs/2403.09629
36. OpenThoughts: Data Recipes for Reasoning — https://arxiv.org/abs/2506.04178
37. s1: Simple Test-time Scaling — https://arxiv.org/abs/2501.19393
38. LIMO: Less Is More for Reasoning — https://arxiv.org/abs/2502.03387
39. Phi-4-reasoning — https://arxiv.org/abs/2504.21318
40. Kimi K1.5 — https://arxiv.org/abs/2501.12599
41. Nemotron-CrossThink — https://arxiv.org/abs/2504.13941
42. OpenMathReasoning / AIMO-2 — https://arxiv.org/abs/2504.16891
43. DeepSeek-Prover-V1.5 — https://arxiv.org/abs/2408.08152

### D. Conversation 데이터셋·SFT 포맷
44. LIMA — https://arxiv.org/abs/2305.11206
45. FLAN Collection — https://arxiv.org/abs/2301.13688
46. Self-Instruct — https://arxiv.org/abs/2212.10560
47. Evol-Instruct / WizardLM — https://arxiv.org/abs/2304.12244
48. Tulu 3 — https://arxiv.org/abs/2411.15124
49. WildChat 1M — https://arxiv.org/abs/2405.01470
50. ChatQA — https://arxiv.org/abs/2401.10225
51. ChatQA-2 — https://arxiv.org/abs/2407.14482
52. SmolLM3 — https://huggingface.co/blog/smollm3

### E. 모델 테크리포트 추가
53. Nemotron-4 340B — https://arxiv.org/abs/2406.11704
54. Nemotron-H — https://arxiv.org/abs/2504.03624
55. MAP-Neo — https://arxiv.org/abs/2405.19327
56. GLM-4.5 — https://arxiv.org/abs/2508.06471
57. InternLM2 — https://arxiv.org/abs/2403.17297
58. Yi-Lightning — https://arxiv.org/abs/2412.01253
59. MiniMax-01 — https://arxiv.org/abs/2501.08313
60. Gemma 2 — https://arxiv.org/abs/2408.00118
61. Gemma 3 — https://arxiv.org/abs/2503.19786
62. SmolLM2 — https://arxiv.org/abs/2502.02737
63. Qwen2.5 — https://arxiv.org/abs/2412.15115

### F. 한국 LLM (§2.7)
64. HyperCLOVA X — https://arxiv.org/abs/2404.01954
65. HyperCLOVA X THINK — https://arxiv.org/abs/2506.22403
66. SOLAR 10.7B (DUS 원조) — https://arxiv.org/abs/2312.15166
67. EXAONE 3.5 — https://arxiv.org/abs/2412.04862
68. EXAONE Deep — https://arxiv.org/abs/2503.12524
69. EXAONE 4.0 — https://arxiv.org/abs/2507.11407
70. Trillion 7B — https://arxiv.org/abs/2504.15431
71. Kanana — https://arxiv.org/abs/2502.18934
72. Mi:dm 2.0 — https://arxiv.org/abs/2601.09066
73. Mi:dm K 2.5 Pro — https://arxiv.org/abs/2603.18788
74. KORMo — https://arxiv.org/abs/2510.09426
75. Polyglot-Ko — https://arxiv.org/abs/2306.02254

---

## 7. 핵심 4줄 (실무 적용 결론)

1. **Mid-training은 거의 모두 "전체 토큰 LM loss"**. SFT의 user-mask와 정반대. 이게 conversation 데이터를 변환할 때 첫 번째 결정.
2. **Think 토큰은 plain text marker 표준** (`<think>...</think>`을 vocab에 안 넣음). Phi-4-reasoning의 placeholder reuse만 예외.
3. **Conversation 변환 5가지 패턴** 중 사용자에게는 **SmolLM3 recipe(ChatML+wrapped packing+no mask)** 가 가장 직접 적용 가능. 단순함이 우선이면 **OLMo 2식 raw concat** 도 검증됨.
4. **Instruction/conversation 비율 1-10%**, 나머지는 HQ web + math/code. WSD decay 또는 cosine→0 annealing의 마지막 5-12.5% 토큰에서 데이터 교체. Replay data로 forgetting 방지.
