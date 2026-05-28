# 추론 모델 데이터 레시피 — 엄선 보고서 (2025~2026 최신작 중심)

> **목적**: "OpenThoughts: Data Recipes for Reasoning Models"를 출발점으로, **깐깐한 기준을 통과한 ~10편**만 골라 논문별 인사이트를 **원문 verbatim 발췌**와 함께 심층 정리한다.
> **이전 광범위 서베이**(26편, 4범주 전부)는 같은 폴더의 `reasoning_data_recipe_papers_report.md`에 있다. 본 문서는 그것을 **최신성·실험강도 기준으로 압축·갱신**한 버전이다.
> **작성일**: 2026-05-28.

---

## 선정 기준 (깐깐하게)

| 기준 | 내용 |
|---|---|
| **① 발표 시기** | 2025년 이후만. 본 보고서는 그중에서도 **2025 H2~2026 신작**을 우선. (H1 논문은 대체 불가한 경우만 date-flag로 포함) |
| **② 모델 최신성** | 연구가 다루는 teacher/base가 현 세대(DeepSeek-R1-0528/V3.1, QwQ-32B, Qwen3, GLM-4.5, gpt-oss-120B, Kimi K2, ProRL-v2 등). 구세대(Qwen2.5 단독, QwQ-Preview, Gemini 2.0) 중심은 감점 |
| **③ 실험 강도** | 대규모 ablation / 통제 실험 / scaling 곡선 필수. 단순 데이터셋 릴리스·블로그 단독·abstract-only 배제 |
| **④ 일반화 가능성** | "무엇이 좋은 데이터를 만드는가"에 대한 전이 가능한 인사이트 |
| **⑤ 중복 제거** | 같은 결이면 최강 1편만 |

**탈락(대표 예)**: gpt-oss model card(데이터 비공개 명시), DeepSeek-V3.2/V3.1(조성 hand-wavy), s1·LIMO·Demystifying·Structure-not-content(2025 H1·구세대 모델 — 이전 서베이에 보존), Distillation Scaling Laws(추론 아닌 LM cross-entropy), ScaleRL(데이터 큐레이션 아닌 compute scaling).

**최종 10편 (발표월 순)**: Qwen3(2505·flag) → OpenThoughts(2506) → Front-Loading(2510→2509)·실제 09 → GLM-4.5(2508) → Nemotron Nano 2(2508) → Merge-of-Thought(2509) → DRIVE(2511) → RLVE(2511) → Data Repetition(2602) → PRISM(2603). (+ 4B 직결 추가참고 §3)

### 발췌 신뢰도 범례
- ✅ **검증**: 본 작성자가 직접(또는 라운드1 PDF 추출로) abstract/full-text에서 문장 단위 verbatim 확인.
- 🟡 **본문기반**: 리서치 에이전트가 full-text HTML에서 확보(검증), abstract엔 없으나 본문 §에 존재.
- ⚠️ **도표기반**: 그림/표 라벨에서 추출 — 수치 인용 시 원문 재확인 권장.

---

## 0. 핵심 요약 (TL;DR)

최신작들은 OpenThoughts의 결론을 **확장·정밀화**한다. 5가지 큰 흐름:

1. **mid-training이 결정적 (사용자 현 단계 직격)** — PRISM(2026-03)은 7개 base 모델 통제실험으로 *"Data composition matters most at mid-training, not RL"* 를 입증. **science 데이터를 mid-training에 넣어야 RL에서 GPQA가 +17~28점** 열리고, RL 믹스 변경은 <2점. mid-training이 가중치의 90% 이상을 재구성, RL은 ~5%만 손본다.
2. **SFT는 "많은 고유 데이터"보다 "작은 셋 다회 반복"** — Data Repetition(2026-02): 고정 update 예산에서 **400샘플 128에폭이 51,200샘플 1에폭을 12~26점 능가**, token accuracy가 포화 신호.
3. **teacher는 단일 oracle이 아니다** — Merge-of-Thought(2025-09): *"different students have different best teachers"*. 다중 teacher의 student variant를 **weight-merge**하면 best-single·naive-union을 능가(Qwen3-14B, ~200 CoT로 R1·o1 능가). OpenThoughts의 "QwQ>R1" 반직관과 같은 결.
4. **하이브리드 추론은 데이터로 만든다 (사용자 목표 직격)** — Qwen3의 `/think` `/no_think` fusion, GLM-4.5의 reasoning/direct 데이터 균형, Nemotron Nano 2의 truncated-trace로 **thinking budget 제어**.
5. **RL은 소량·고난이도·다양성** — DRIVE(2025-11): broad→hard-focus 2단 커리큘럼으로 9K 프롬프트만으로 V3.1급 코드 성능. RLVE(2025-11): **정적 데이터 포화 후엔 compute보다 환경 다양성**(+3.37% vs +0.49%, 3배 compute에도).

---

# Part 1 — 심층 분석 (10편)

---

## 1. PRISM: Demystifying Retention and Interaction in Mid-Training 〔mid-training, 2026 최신〕
**IBM Research·MIT-IBM Watson AI Lab** (Runwal, Agrawal, Roy, Panda) · **2026-03-17** · **arXiv 2603.17074** · https://arxiv.org/abs/2603.17074

**한 줄**: 7개 base 모델·4개 패밀리(Granite/LLaMA/Mistral/Nemotron-H)·3B~24B에 걸친 통제 실험으로, *추론 능력은 RL이 아니라 mid-training 데이터 조성에서 대부분 결정됨*을 기계적으로 입증. **사용자의 4B mid-training에 가장 직접적인 최신 근거.**

**원문 발췌**:
- (Abstract, ✅) *"Through controlled experiments across seven base models spanning four families (Granite, LLaMA, Mistral, Nemotron-H), two architecture types (dense Transformer and attention-Mamba hybrid), and scales from 3B to 24B parameters, we show that a mid-training phase of ∼27B high-quality tokens yields consistent gains of +15 to +40 points on math, +5 to +12 points on code, and +6 to +13 points on science (GPQA-Diamond)..."*
- (Abstract, ✅ 핵심) *"Data composition matters most at mid-training, not RL"*
- (Abstract, ✅) *"including science data during mid-training unlocks +17 to +28 point GPQA-Diamond gains during RL"* (RL 믹스 변경은 <2점)
- (Abstract, ✅) *"mid-training densely restructures over 90% of model weights, while RL makes sparse, front-loaded refinements to approximately 5% of parameters"*
- (Abstract, 🟡) *"RL consistently preserves mid-training's representational geometry (>0.998 CKA) across both dense Transformers and hybrid architectures."*
- (§3 Takeaway, 🟡) *"Mid-training performance is highly sensitive to data composition; carefully tuned mixtures that balance general web and instruction data with domain-specific reasoning sources yield robust retention and consistent gains..."*

**인사이트**:
- 【d】(timing/composition) **조성 레버는 mid-training에 있다.** RL은 mid-training이 만든 표현 기하를 거의 그대로 보존(CKA>0.998)하고 가중치의 5%만 손봄. → "RL에서 고치면 된다"는 통념 반박.
- 【b】 **도메인 의존성**: science를 mid-training에 *반드시* 넣어야 RL에서 GPQA가 열린다(+17~28). 도메인별로 "언제 넣느냐"가 다름.
- 【c】 ~27B 고품질 토큰 + (general web + instruction + domain reasoning) 균형 믹스로 큰 이득. 단일 도메인 몰빵이 아니라 *균형*.
- 패밀리·아키텍처(dense/hybrid Mamba) 불문 일관 — 레시피의 일반성.

**신뢰도**: ✅ abstract 핵심 4개 직접 검증(2026-03, 학습 컷오프 이후 논문이라 별도 확인). 본문 수치는 🟡.

---

## 2. Data Repetition Beats Data Scaling in Long-CoT SFT 〔SFT 사이징, 2026 최신〕
**Qualcomm AI Research·Univ. of Amsterdam** (Kopiczko, Vaze, Blankevoort, Asano) · **2026-02-11** · **arXiv 2602.11149** · https://arxiv.org/abs/2602.11149

**한 줄**: 고정 update 예산에서 *작은 고품질 셋을 여러 에폭 반복*하는 것이 *큰 셋을 1에폭* 도는 것을 능가. token accuracy가 멈출 시점을 알려준다.

**원문 발췌** (✅ abstract 전문 검증):
- *"Counterintuitively, we show that SFT benefits from repetition: under a fixed update budget, training for more epochs on smaller datasets outperforms single-epoch training on larger datasets."*
- *"On AIME'24/25 and GPQA benchmarks, Olmo3-7B trained for 128 epochs on 400 samples outperforms the equivalent 1 epoch on 51200 samples by 12-26 percentage points, with no additional catastrophic forgetting."*
- *"We find that training token accuracy reliably signals when repetition has saturated; improvements from additional epochs plateau at full memorization, a pattern consistent across all settings."*
- *"...scaling epochs with token accuracy as a stopping criterion can replace expensive undirected data scaling."*
- (§2.3, 🟡) *"at a budget of 51,200 updates, Olmo3-7B trained for 32 epochs on 1,600 samples reaches an average 39% accuracy across benchmarks, compared to 17% for a single epoch on 51,200 samples."*

**인사이트**:
- 【d】 **"적은 고유 데이터 × 다회 반복"이 "많은 고유 데이터 × 1회"를 이긴다**(동일 compute). Olmo3-7B/Qwen3 distill로 검증. 30B급 무차별 확장보다 작은 큐레이션 셋 반복이 유리할 수 있음.
- 【c】 **무료 stopping signal**: training token accuracy가 ~100%(완전 암기)에 닿으면 반복 이득 포화 → 데이터 추가를 멈출 신호.
- 【a】 teacher 크기와 샘플 정답성이 반복 이득을 조절(분석축).
- *유의*: 이는 long-CoT **SFT** 한정 결과. "full memorization이 generalization 향상과 동시 발생"을 저자 스스로 open problem으로 제기.

**신뢰도**: ✅ abstract 전문 검증(2026-02, 컷오프 이후 별도 확인). 본문 §2.3 수치 🟡.

---

## 3. Merge-of-Thought Distillation (MoT) 〔teacher 다양성, 2025 H2〕
**Zhanming Shen 외 (Zhejiang Univ. 계열 공개)** · **2025-09-10** · **arXiv 2509.08814** · https://arxiv.org/abs/2509.08814

**한 줄**: 단일 oracle teacher 가정을 깨고, teacher별 SFT 분기를 학습한 뒤 **weight-space merging**으로 통합. ~200 CoT로 Qwen3-14B가 R1·o1을 능가.

**원문 발췌** (✅ abstract 검증):
- *"We revisit teacher selection and observe that different students have different "best teachers," and even for the same student, the best teacher can vary across datasets."*
- *"...Merge-of-Thought Distillation (MoT), a lightweight framework that alternates between teacher-specific supervised fine-tuning branches and weight-space merging of the resulting student variants."*
- *"On competition math benchmarks, using only about 200 CoT samples, applying MoT to a Qwen3-14B student surpasses strong models including Deepseek-R1, Qwen3-32B, and OpenAI-O1, demonstrating substantial gains."*
- (§6.3, 🟡) *"reasoning trajectories distilled from peer-level teachers can still help ... and using all teachers performs best."*
- (§4.1, 🟡) source = BOBA + S1K, 각 200 prompt 샘플.

**인사이트**:
- 【a】 **"최고의 teacher"는 student·데이터셋마다 다르다.** 다중 teacher의 student variant를 merge하면 best-single과 naive-union(단순 합집합 학습)을 모두 능가 — OpenThoughts의 "QwQ가 R1보다 좋은 teacher"와 같은 메시지를 일반화.
- 【a】 **peer-level(동급) teacher도 도움**, 전부 쓰는 게 최선 → monoculture 회피의 강한 근거.
- 【c】 weight-merge가 일종의 consensus filter로 overfitting/forgetting 완화.
- 【d】 ~200 prompt의 극소량 + 다중 teacher 절차로 14B가 frontier 능가.

**신뢰도**: ✅ abstract 검증. 본문 §4.1/§6.3 🟡.

---

## 4. DRIVE: Data Curation Best Practices for RLVR in Competitive Code Generation 〔RL 데이터, 2025 H2〕
**Tencent** (Speed Zhu 외) · **2025-11-09** · **arXiv 2511.06307** · https://arxiv.org/abs/2511.06307

**한 줄**: RLVR을 *데이터 큐레이션* 관점에서 정리한 보기 드문 논문. broad-uniform → hard-focus 2단 커리큘럼(Pre-GRPO)으로 소량 고난이도 프롬프트만으로 V3.1급 코드.

**원문 발췌** (✅ abstract 검증):
- *"...advances are dominated by mathematics (e.g., AIME), with competitive-programming code generation underexplored and data curation receiving less attention than RL algorithm design."*
- *"...first, training on a large, uniformly distributed set of competitive-programming problems using Group Relative Policy Optimization (GRPO) with 8 rollouts per prompt and a relatively short response-generation window (e.g., 32k during SFT and 24k in this stage) to expand entropy and mitigate repetition and truncation;"*
- *"...second, we perform Pre-GRPO: updating on a small, high-quality set of challenging problems with a large rollout budget (64 rollouts per prompt) under a hard-focus curriculum that continuously retains the most difficult instances throughout training."*
- *"We implement our method on Qwen2.5-32B ... comparable to leading systems such as DeepSeek v3.1 and Doubao-1.5-Thinking."*
- (🟡 본문 §3·§4, abstract 미포함) SFT seed는 DeepSeek-R1-0528 distill, 1.27M→470K(5-round arena), stage-1 RL 9K, hard-focus phase는 72→50→25개 최난문제.

**인사이트**:
- 【b】 **커리큘럼 형태가 핵심**: 먼저 broad·uniform로 entropy 확장(반복·truncation 억제) → 이후 가장 어려운 소수(수십 개)만 남기는 hard-focus.
- 【c】 실행 가능·testcase 기반 binary reward + 난문에 큰 rollout 예산(64/prompt)으로 sparse signal 추출.
- 【d】 1.27M→470K→9K로 좁혀도 V3.1급 — RL 프롬프트는 *대부분 잉여*, 소수 정예가 핵심(🟡 본문).
- 【a】 강한 오픈모델(R1-0528) distill로 SFT seed를 먼저 깔고 RL(🟡 본문). base는 Qwen2.5-32B(구세대지만 teacher·비교는 최신).

**신뢰도**: ✅ abstract 검증(two-stage·Pre-GRPO·V3.1 비교). teacher(R1-0528)·1.27M→470K·9K·72/50/25는 🟡(본문, abstract 미포함).

---

## 5. RLVE: Scaling Up RL with Adaptive Verifiable Environments 〔RL 다양성, 2025 H2〕
**UW·Allen AI·UIUC 외** (Zeng 외) · **2025-11-10** · **arXiv 2511.07317** · https://arxiv.org/abs/2511.07317

**한 줄**: 정적 데이터 대신 *난이도를 정책에 맞춰 동적 적응*하는 400개 검증가능 환경(RLVE-Gym). 정적 셋 포화 후엔 **compute보다 환경/도메인 다양성**이 성능을 올린다.

**원문 발췌** (✅ abstract 검증):
- *"RLVE enables each verifiable environment to dynamically adapt its problem difficulty distribution to the policy model's capabilities as training progresses. In contrast, static data distributions often lead to vanishing learning signals when problems are either too easy or too hard for the policy."*
- *"...RLVE-Gym, a large-scale suite of 400 verifiable environments carefully developed through manual environment engineering."*
- *"...environment scaling, i.e., expanding the collection of training environments, consistently improves generalizable reasoning capabilities."*
- *"RLVE with joint training across all 400 environments in RLVE-Gym yields a 3.37% absolute average improvement across six reasoning benchmarks, starting from one of the strongest 1.5B reasoning LMs. By comparison, continuing this LM's original RL training yields only a 0.49% average absolute gain despite using over 3x more compute."*
- (🟡 본문 §5.1) base 후보: ProRL-1.5B-v2, Qwen2.5-7B-Base, R1-Distill-1.5B, DeepScaleR-1.5B.

**인사이트**:
- 【d】(diversity vs compute) **정적 데이터(이미 강한 ProRL)에서 RL을 더 돌리는 것(+0.49%, 3배 compute)보다, 다양한 검증 환경을 더하는 것(+3.37%)이 압도적.** 데이터 다양성이 병목.
- 【b】(difficulty) **정적 분포는 너무 쉽거나 어려우면 학습 신호가 소멸** → 환경이 정책 능력에 맞춰 난이도를 동적 생성(offline 난이도 필터의 생성형 대안).
- ProRL-v2처럼 RL로 포화된 모델조차 다양 환경 데이터로 더 오른다.

**신뢰도**: ✅ abstract 검증(400환경·3.37% vs 0.49%·dynamic difficulty). base 명칭은 🟡(본문).

---

## 6. Qwen3 Technical Report 〔하이브리드 토글 정전(正典), date-flag〕
**Alibaba/Qwen** · **2025-05** (date-flag: H1이나 하이브리드 추론 레시피의 정전·최신 frontier) · **arXiv 2505.09388** · https://arxiv.org/abs/2505.09388

**한 줄**: 4단 post-training(long-CoT cold-start → reasoning RL → **thinking-mode fusion** → general RL)으로 `/think`·`/no_think`를 단일 모델에 통합. 사용자의 하이브리드 추론 목표와 직결.

**원문 발췌** (🟡 리서치 에이전트 full-text HTML 확보):
- (§4.1 query 필터) *"In the query filtering phase, we use Qwen2.5-72B-Instruct to identify and remove queries that are not easily verifiable."* / *"...we exclude queries that Qwen2.5-72B-Instruct can answer correctly without using CoT reasoning."*
- (§4.1 teacher/검증) *"we generate N candidate responses for each remaining query using QwQ-32B"* / *"When QwQ-32B consistently fails to generate correct solutions, human annotators manually assess the accuracy..."*
- (§4.2 RL 데이터 4기준) *"The query-verifier pairs used in the Reasoning RL stage must satisfy the following four criteria: (1) They were not used during the cold-start phase. (2) They are learnable for the cold-start model. (3) They are as challenging as possible. (4) They cover a broad range of sub-domains."*
- (§4.2 규모) *"We ultimately collect a total of 3,995 query-verifier pairs, and employed GRPO..."*
- (§4.3 fusion) *"we introduce /think and /no_think flags ... For non-thinking mode samples, we retain an empty thinking block in the assistant's response."* / *"By default, the model operates in thinking mode..."*

**인사이트**:
- 【하이브리드】 reasoning-on/off를 **별도 모델이 아니라 템플릿 토큰(`/think` `/no_think`)으로 단일 모델에 통합**. non-thinking 샘플은 *빈 thinking 블록*을 유지 — 사용자 hybrid 모델 설계의 정전 패턴.
- 【a】 **검증/필터용 모델(Qwen2.5-72B)과 trace 생성기(QwQ-32B)를 분리**. teacher 실패 시 인간 검수 fallback.
- 【b】 RL 데이터는 *학습가능·최대 난이도·광범위 sub-domain* 만, 단 3,995쌍으로 극소.
- 【d】(timing) 추론은 전부 post-training; small 모델은 4단 파이프라인 대신 **logit distillation**(1/10 GPU시간).

**신뢰도**: 🟡 본문 §4 full-text(2025-05, 공개 문서로 잘 정립). 일부 §번호·distillation 인용은 근사.

---

## 7. GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models 〔frontier mid-training+하이브리드, 2025 H2〕
**Z.ai/Zhipu** · **2025-08** · **arXiv 2508.06471** · https://arxiv.org/abs/2508.06471

**한 줄**: 명시적 토큰 예산의 mid-training 단계 + 도메인 expert 모델 → **self-distillation 통합** + reasoning/direct 데이터 균형으로 하이브리드 모델. 사용자가 선호하는 클린 frontier.

**원문 발췌** (🟡 본문, GLM-4.5는 PDF pdftotext 확보):
- (§2.3 mid-training) *"Synthetic Reasoning Data Training At this stage, we add synthetic reasoning content for math, science, and coding competitions. We collect a large number of questions and answers ... and synthesize reasoning processes with a reasoning model."*
- (§2.3 repo code) *"...we add concatenated code files from the same repository to learn cross-file dependency ... We extend the training sequence length from 4K to 32K"*
- (Figure 3 라벨, ⚠️) mid-training 토큰 예산 = Repo-Code 500B · Synthetic Reasoning 500B · Long-context & Agent 100B (pretrain General 15T + Code&Reasoning Continual 7T).
- (§3.1 하이브리드) *"...we meticulously balanced training data containing full reasoning with data lacking explicit thought processes. This approach allows the model to operate in both the reflective and immediate response modes, thereby creating a hybrid reasoning model."*
- (§3.2 RL 데이터) *"using exclusively expert-verified multiple-choice questions for RL leads to significantly better performance compared to training with mixed-quality or unverified data ... rigorously filtering the RL data pool to include only high-quality, challenging instances is crucial"*
- (§3.3.2 self-distill) *"we apply self-distillation by substituting the original cold-start data with responses generated by the RL-trained model, thus creating a superior SFT model."*

**인사이트**:
- 【d】(timing) 추론을 **mid-training**(합성 reasoning 500B 토큰)에 명시 투입 + 시퀀스 길이 4K→32K→128K 단계 확장, best-fit packing으로 추론 체인 truncation 방지.
- 【하이브리드】 full-reasoning과 no-thought 데이터를 **균형**시켜 reflective/immediate 양 모드 — GLM의 하이브리드 레시피.
- 【a】 도메인 **expert 모델을 따로 학습 후 self-distill**로 한 generalist에 통합(모델이 스스로의 teacher).
- 【c】 RL 데이터는 *expert-verified·hard·filtered* 가 mixed/unverified를 크게 능가 — quality-over-quantity의 명시 ablation(긴장 ①에서 "검증 중요" 편).

**신뢰도**: 🟡 본문 prose(검증). ⚠️ Figure 3 토큰 예산(500B/500B/100B, 15T/7T)은 도표 라벨 — 수치 인용 시 재확인.

---

## 8. NVIDIA Nemotron Nano 2 (Nemotron-H family) 〔R1-0528 teacher·thinking budget, 2025 H2〕
**NVIDIA** · **2025-08~09** · **arXiv 2508.14444** · https://arxiv.org/abs/2508.14444

**한 줄**: DeepSeek-R1-0528 teacher + 다중 teacher 합성, **truncated-reasoning 데이터로 thinking budget 제어**, distillation 믹스 ablation까지 갖춘 최신 frontier 레시피.

**원문 발췌** (🟡 본문 full-text):
- (§3.1 teacher) *"For Math, Science and Coding data, we generate responses using the open-weights DeepSeek-R1-0528 model"*
- (§3 규모/truncation) *"post-training was performed on roughly 90 billion tokens"* / *"About 5% of the data contained deliberately truncated reasoning traces"*
- (§3.2 budget control) *"augmented examples across domains where reasoning traces were abruptly truncated to 1–2k tokens while preserving the final answer"*
- (§3.4) *"Nemotron Nano V2 allows users to specify how many thinking tokens the model may generate before producing the final answer"*
- (§2.2.2 합성 STEM, 다중 teacher) 88.6k seed 질문을 *"Qwen3-30B-A3B and Qwen3-235B-A22B, both with thinking mode enabled, Deepseek-R1, and Deepseek V3"* 로 생성.
- (§2.2.2 난이도 라벨) *"assigned attribute labels for educational quality, educational difficulty, and educational subject to all documents coming from academic data"*
- (Table 11, ⚠️) distillation 믹스 *"70% post-training stage 2 data and 30% pretraining data yields the highest accuracy"*

**인사이트**:
- 【하이브리드】 **5% truncated-trace + 1~2k 절단 데이터**가 thinking budget 제어의 메커니즘 — "어려우면 길게, 쉬우면 짧게"를 데이터로 학습. 사용자 overthinking 문제에 직접적.
- 【a】 다중 teacher(R1-0528 + V3 + Qwen3 2종, thinking on) 합성 — 다양성.
- 【b/c】 학술 문서에 educational quality/difficulty 라벨 부여 후 upsample, 코드 QA는 실제 코드 스니펫 seed + heuristic 사후 필터.
- 【d】(timing) 추론을 pretrain(라벨 STEM) + post-train 모두에; distill은 70% post + 30% pretrain 혼합이 최적(⚠️ Table 11).

**신뢰도**: 🟡 본문 prose(검증). ⚠️ Table 11 70/30은 표 기반.

---

## 9. Front-Loading Reasoning: Synergy between Pretraining and Post-Training Data 〔언제 넣나, 2025 H2〕
**NVIDIA·CMU** (Akter 외) · **2025-09/10** · **arXiv 2510.03264** · https://arxiv.org/abs/2510.03264

**한 줄**: 8B from-scratch 통제 실험 — 추론 데이터를 **pretraining에 front-load**하면 나중 SFT로 회복 불가능한 이득(+19%), "pretrain=다양성 / SFT=품질" 비대칭.

**원문 발췌** (✅ 라운드1 full-text HTML 검증):
- (Abstract) *"Front-loading reasoning data into pretraining is critical (19% average gain), establishing foundational capabilities that cannot be fully replicated by later-stage SFT, even with more data."*
- (Abstract) *"Pretraining benefits most from broad diversity in reasoning patterns (11% average gain), while SFT is more sensitive to data quality (15% average gain with high quality data)."*
- (§5) *"This provides strong evidence that pretraining instills a foundational reasoning capability that cannot be fully replicated by simply scaling the SFT phase"*
- (§2.2) 고품질 SFT 소스로 *"the dataset of Guha et al. (2025), comprising 1.2M carefully curated examples (71% math, 21% code, 8% science)"* ← **OpenThoughts 데이터셋**
- (§2.2 필터) *"retaining examples where the answer length exceeds 4096 tokens, based on the principle that longer responses often correspond to more complex CoT reasoning"*

**인사이트**:
- 【d】(timing) **추론 노출을 SFT까지 미루지 말 것** — pretraining front-load의 이득은 나중에 SFT를 늘려도 되살 수 없다. PRISM(mid-training 결정성)과 상보적: 둘 다 "이른 단계가 결정적".
- 【a/b】 **pretrain/mid = 다양성, SFT = 품질**의 단계별 비대칭 전략.
- 【c】 명시 필터는 주로 답변 길이(>4096) = 복잡도 프록시.

**신뢰도**: ✅ 라운드1 full-text 검증.

---

## 10. OpenThoughts: Data Recipes for Reasoning Models 〔SFT 앵커〕
**Stanford·Bespoke·TRI·UW 외** · **2025-06** · **arXiv 2506.04178** · https://arxiv.org/abs/2506.04178

**한 줄**: 1,000+ ablation으로 SFT 추론 데이터 파이프라인을 역설계. 본 보고서 다른 논문들이 확장하는 *기준점*.

**원문 발췌** (✅ 라운드1 PDF 직접 추출):
- (Key Finding #2) *"Models with better performance are not necessarily better teachers. QwQ-32B is a stronger teacher than DeepSeek-R1, although it scores lower on target reasoning benchmarks."*
- (Key Finding #1) *"Sampling multiple answers per question from a teacher model is an effective technique to increase the size of a data source by at least 16×. The increased dataset scale drives significant performance gains."*
- (Key Finding #4) *"Selecting questions from a small number (top 1 or 2) of high-quality sources leads to better downstream performance compared to optimizing for diversity..."*
- (Key Finding #5) *"Filtering questions by LLM labeled difficulty or LLM response length yields better results than filters typical to pre-training data curation that use embeddings or fastText."*
- (Key Finding #3, 검증 음성결과) *"We experimented with numerous verification and answer filtering methods, and none gave significant performance improvements."*

**인사이트**:
- 【a】 **벤치 강자 ≠ 좋은 teacher**(QwQ>R1). Merge-of-Thought(§3)가 이를 "student마다 best teacher 다름"으로 일반화.
- 【b】 소수 고품질 소스 집중 + LLM-judge 난이도/응답길이 필터.
- 【c】 **정답 필터링이 유의미 이득 없음**(compute-control 시) — 단, GLM-4.5(§7)는 RL 데이터에선 검증이 중요하다고 봄 → SFT vs RL 단계 구분 필요.
- 【d】 16× 재샘플링이 주 스케일 레버.

**신뢰도**: ✅ 라운드1 PDF 검증. (앵커로 유지; 세부는 광범위 서베이 문서 A1 참조.)

---

# Part 2 — 큐레이션 축별 횡단 종합

## 2.A 【a】 Teacher / 생성모델 선택·다양성
- **단일 oracle teacher는 없다.** OpenThoughts(QwQ>R1) → Merge-of-Thought(student·데이터셋마다 best teacher 다름, weight-merge가 best-single·union 능가, peer teacher도 기여) → Nemotron Nano 2(R1-0528+V3+Qwen3 2종 혼합 합성). **다중 teacher 권장, monoculture 회피.**
- **역할 분리**: Qwen3는 검증/필터 모델(Qwen2.5-72B)과 trace 생성기(QwQ-32B)를 분리. teacher 실패 시 인간 fallback.

## 2.B 【b】 질문 소싱·난이도·다양성
- **난이도는 "정책 능력 기준 상대값"**: RLVE(정적 분포는 too-easy/too-hard에서 신호 소멸 → 동적 적응), DRIVE(broad→hardest 수십 개), OpenThoughts/Qwen3("학습가능·최대난이도"만). PRISM은 *도메인별로* 어디 넣느냐가 다름(science=mid-training).
- **소수 정예 소스**(OpenThoughts top-1/2) + **LLM-judge 난이도/응답길이 필터**가 임베딩·fastText보다 우위.

## 2.C 【c】 정답 검증·품질 필터링 — 단계로 갈린다
- **SFT 단계**: OpenThoughts는 *정답 필터링 무의미*. Data Repetition은 *token accuracy*가 품질 신호(포화 감지). → SFT에선 과도한 정답 필터보다 구조·반복이 중요.
- **RL 단계**: GLM-4.5는 *expert-verified·hard·filtered RL 데이터가 mixed/unverified를 크게 능가*. DRIVE는 executable testcase binary reward. → RL에선 검증 필수.
- **요지**: *SFT는 검증 완화 가능, RL은 정확 검증 필수* — 이전 서베이의 긴장 ①을 단계 구분으로 해소.

## 2.D 【d】 스케일·믹스·반복
- **"적은 데이터" 재해석**: Data Repetition(작은 셋 다회 반복 > 큰 셋 1회), DRIVE(9K로 V3.1급), RLVE(다양성>compute), Merge-of-Thought(~200 CoT). → *고유 데이터 양*보다 *반복·다양성·난이도*가 레버.
- **믹스/timing**: PRISM(mid-training 조성이 RL보다 결정적, science 필수), Front-Loading(pretrain front-load는 SFT로 회복 불가, pretrain=다양성/SFT=품질), GLM-4.5·Nemotron(mid+post 양쪽 투입), Nemotron distill 70%post+30%pretrain.

## 2.E 하이브리드 추론 — 데이터로 만든다 (사용자 목표)
- **토글 메커니즘**: Qwen3 `/think`·`/no_think`(non-think는 빈 thinking 블록), GLM-4.5 reasoning/direct 데이터 균형, Nemotron Nano 2 truncated-trace(1~2k 절단)로 **thinking budget 사용자 지정**.
- **시사**: "쉬운 태스크엔 짧게/안 함"을 별도 모델이 아니라 *혼합 데이터 + 템플릿 토큰 + 절단 trace*로 단일 모델에 학습.

---

# Part 3 — 추가 참고 (4B·최신성 직결, 2차 우선순위)

깐깐한 기준의 본선 10편엔 못 들었으나 사용자 4B 빌드에 직접 관련된 신작 (각 1줄, ✅=abstract 확인):

- **Skill-Aware Data Selection** (arXiv 2601.10109, 2026-01, ✅) — *"With only 1,000 training examples selected from a 100K teacher-generated corpus, our method surpasses random SFT baselines by +1.6% on Qwen3-4B and +1.4% on Qwen3-8B"*. **Qwen3-4B = 사용자 스케일**, R1 teacher. 이득은 modest.
- **DASD-4B-Thinking / Distribution-Aligned Sequence Distillation** (arXiv 2601.09088, 2026-01) — Qwen3-4B-Instruct student + **gpt-oss-120B teacher**(클린 라이센스), low-temp→high-temp 2단 distill, 다도메인. *방법론 비중↑*이나 4B+gpt-oss 조합이 사용자 자산과 직매칭.
- **SmolLM3** (HF blog, 2025-07) — **전용 reasoning mid-training 단계**(OpenThoughts3+R1 trace 35B토큰×4에폭) + dual-mode SFT(1B no-think/0.8B think). mid-training 템플릿으로 가장 직접적(단, 블로그 출처).
- **NaturalThoughts** (arXiv 2507.01921, 2025-07, ✅) — Less-is-More 반박; 넓은 STEM 분포는 500K까지 무포화, 난이도+다양성 선택이 sample-efficient.
- **Scaling Behaviors of LLM RL Post-Training** (arXiv 2509.25300, 2025-09) — RL 데이터 scaling law(R²>0.99); 데이터 제약 시 *uniqueness보다 total optimization steps*, 소량 셋 ~25× 재사용 무손실.
- **FLAMES** (arXiv 2508.16514, 2025-08) — 10개 합성 전략 통제 비교; *난이도↑ 에이전트가 최선, 고정 예산에선 정답 신뢰성보다 문제 coverage가 중요*.

(s1·LIMO·OpenCodeReasoning·OpenMathReasoning·Phi-4·AceReason·DAPO·ProRL·DeepSeek-R1 distill·Structure-not-content·Demystifying 등 2025 H1 논문은 같은 폴더 `reasoning_data_recipe_papers_report.md`에 보존.)

---

# Part 4 — 인덱스 & 신뢰도

| # | 논문 | 범주 | 발표 | arXiv | 한 줄 핵심 | 신뢰도 |
|---|---|---|---|---|---|---|
| 1 | PRISM | mid-training | 2026-03 | 2603.17074 | 조성은 mid-training이 결정, science@mid가 RL GPQA +17~28 | ✅abs |
| 2 | Data Repetition Beats Scaling | SFT 사이징 | 2026-02 | 2602.11149 | 400샘플×128에폭 > 51,200×1에폭 (12~26점), token-acc 정지신호 | ✅abs |
| 3 | Merge-of-Thought | teacher 다양성 | 2025-09 | 2509.08814 | student별 best teacher 다름, weight-merge가 union 능가 | ✅abs |
| 4 | DRIVE | RL 데이터 | 2025-11 | 2511.06307 | broad→hard-focus 2단, 9K로 V3.1급 코드 | ✅abs/🟡본문 |
| 5 | RLVE | RL 다양성 | 2025-11 | 2511.07317 | 400 동적난이도 환경, 다양성 +3.37% vs compute +0.49% | ✅abs |
| 6 | Qwen3 | 하이브리드 토글 | 2025-05* | 2505.09388 | /think·/no_think fusion, RL 3,995쌍 | 🟡본문 |
| 7 | GLM-4.5 | mid+하이브리드 | 2025-08 | 2508.06471 | mid-training 토큰예산, expert→self-distill, RL 검증중요 | 🟡본문/⚠️도표 |
| 8 | Nemotron Nano 2 | teacher·budget | 2025-08 | 2508.14444 | R1-0528 teacher, 5% truncated-trace로 thinking budget | 🟡본문/⚠️표 |
| 9 | Front-Loading | timing | 2025-09 | 2510.03264 | pretrain front-load는 SFT로 회복 불가(+19%) | ✅full |
| 10 | OpenThoughts | SFT 앵커 | 2025-06 | 2506.04178 | QwQ>R1 teacher, 검증 무용, 16× 재샘플 | ✅PDF |

\* Qwen3는 2025-05(H1)이나 하이브리드 추론 레시피의 정전으로 date-flag 포함.

**신뢰도 주의**:
- 2026 신작(PRISM·Data Repetition·Skill-Aware·DASD)은 학습 컷오프 이후라 **본 작성자가 arXiv abstract를 직접 재검증**함(존재·제목·저자·날짜·핵심 주장).
- 🟡 본문 인용(Qwen3·GLM-4.5·Nemotron·DRIVE 일부)은 리서치 에이전트가 full-text에서 확보. ⚠️ 도표/표 기반 수치(GLM-4.5 토큰예산, Nemotron 70/30)는 인용 전 원문 재확인 권장.
- DRIVE의 teacher(R1-0528)·1.27M→470K·9K는 abstract엔 없고 본문 §3·§4 기준.
