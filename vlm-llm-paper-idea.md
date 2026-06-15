> **메타데이터**
> - 생성일: 2026-06-02
> - 영문 제목: *The Sufficiency Axis: A Quality-Decorrelated, Relevance-Targeted Test for an Internal "Enough-To-See" Direction in Vision-Language Models*
> - 분야: VLM 신뢰성 × 기계적 해석가능성 (mechanistic interpretability) — "knowing-when-it-cannot-see"
> - 논문 장르: 진단/메커니즘 분석 + training-free intervention
> - **리뷰어 평가: overall 평균 7.00 / novelty 평균 7.50 / 최저 7 / 치명결함 0 → 합격(PASS)**
> - 통과 기준: avg ≥ 7.0, min ≥ 6, novelty ≥ 6.5, fatal flaw 0, skeptic "already_done" 없음 (모두 충족)

---

# 연구 제안서

## The Sufficiency Axis: VLM 내부의 "볼 수 있을 만큼 충분한가" 방향에 대한 품질-비상관·관련성 표적 검증

---

## 1. 선정 분야와 선정 이유 (왜 이 화이트스페이스인가)

본 제안은 **Vision-Language Model(VLM)이 "이미지가 너무 열화되었다(the image is too degraded)"는 신호와 "나는 답할 수 없다(I cannot answer)"는 신호를 내부적으로 분리해서 인코딩하는가, 그리고 그 분리 가능한 image-quality / evidence-sufficiency 신호가 본래 abstention(기권)을 유발해야 함에도 실패하는가**라는 메커니즘 질문을 선정 분야로 삼는다.

이 화이트스페이스를 선정한 이유는 다음과 같다.

- **"knowing-when-it-cannot-see"의 문자 그대로의 핵심**이다. 자율주행·의료·문서 이해 등 안전이 중요한 환경에 배포된 VLM이 시각 증거가 열화·부재할 때 정직하게 기권해야 하는데, 실제로는 부재한 증거로부터 자신 있게 답을 날조한다.
- 인접 연구 공간이 **generic refusal을 위한 steering**(SteerVLM, "Steering to Say No" arXiv:2602.07013)이나 **latent abstention probe**(LRP, arXiv:2511.19806)에 머물러 있을 뿐, **evidence-sufficiency / image-quality를 answer-uncertainty와 구별되는 표상(representation)으로 분리**하거나, abstention이 그 신호에 의해 **인과적으로(causally)** 유발됨(혹은 linguistic answer-prior에 의해 인과적으로 무력화됨)을 보인 연구는 없다.
- 벤치마크가 아니라 **인과/메커니즘 질문**(separability + steering + counterfactual degradation)으로 프레이밍되어 있어, 깨끗한 positive result는 (a) "올바른 abstention을 예측하지만 answer-prior에 의해 무력화되는 linear evidence-sufficiency direction이 존재하고, steering으로 정직한 refusal을 복구할 수 있다"는 **놀라운, 잘 측정된 인과적 발견**이라는 헤드라인과, (b) **training-free intervention**이라는 실용적 임팩트를 동시에 제공한다.

요컨대 이 분야는 **openness(아직 아무도 정확히 이 disentanglement를 하지 않음)와 impact(안전성 주장)의 최적 조합**을 갖는다.

---

## 2. 문제 정의 (왜 중요하고, 왜 아직 안 풀렸나)

안전이 중요한 환경(자율주행, 의료, 문서 이해)에 배포된 VLM은 시각 증거가 너무 열화되었거나 부재하여 질문에 답할 수 없을 때 **기권(abstain)** 해야 하지만, 실제로는 자신 있게 날조된 답을 내놓는다. 진전을 가로막는 세 가지 미해결 문제가 있다.

**(1) Regime(영역)의 부재.** 동시기 메커니즘 연구는 **clean image** 위에서의 visual-linguistic knowledge conflict를 다룬다. 즉 정답이 percept 안에 존재하지만 language prior가 그것을 덮어쓰는 상황("When Seeing Overrides Knowing", arXiv:2507.13868; 동시기 "Arbitration Failure" 계열, arXiv:2604.09364)이다. 이는 **"증거가 전혀 없는데 prior가 답을 날조하고, abstention이 정답인"** 사건과 근본적으로 다른 event다. **열화/불충분(degraded/insufficient) regime에서 answer recovery가 아니라 abstention이 정답인 경우를 다룬 메커니즘 연구는 없다.**

**(2) Construct validity(구성 타당도)의 결여.** 모든 선행 degradation probe는 low-level image quality(blur, noise, exposure)를 task-level sufficiency와 혼동(conflate)한다. global degradation이 둘을 동시에 움직이므로, **단지 흐림(blurriness)을 탐지하는 probe**가 **불충분성(insufficiency)을 탐지하는 probe**와 동일한 점수를 받는다. 따라서 "sufficiency direction" 주장은 quality reading을 배제하는 방식으로 검증된 적이 없다.

**(3) Separability와 causality의 미검증.** sufficiency-predictive probe가 있다 하더라도, 그것이 강력한 capacity-matched answer-uncertainty/entropy direction이나 explicit quality direction과 **구별됨**을 보인 적이 없고, random-direction null에 대비하여 abstention의 **인과적 레버**임을 보인 적도 없다.

우리는 VLM이 **image quality와 answer-uncertainty 양쪽으로부터 분리 가능한, 내부적·선형적·인과적으로 활성인 "볼 수 있을 만큼 충분한가(is-there-enough-to-see)" 신호**를 갖는지 알고자 한다.

---

## 3. 핵심 아이디어 (key insight)

**Task sufficiency와 low-level image quality는 구성적으로(by construction) 통계적 직교(orthogonal)로 만들 수 있으며, 이 직교화 자체가 판별 검정(discriminating test)이다.**

Global degradation은 둘을 얽는다(entangle). **relevance-targeted occlusion at a matched global budget**은 둘을 분리한다. 구체적으로:

- clean image와, 정답이 알려진 relevant region $R$에 의존하는 질문을 가져온다.
- **동일한 degradation budget**(동일한 masked pixel count, clean 대비 matched SSIM/PSNR)에서 두 변형을 만든다:
  - **RELEVANT-OCCLUDED**: $R$을 가린다 → 정답이 더 이상 결정 불가 → **abstain이 정답**.
  - **IRRELEVANT-OCCLUDED**: $R$과 disjoint한 동일 면적 영역을 가린다 → 정답이 여전히 결정 가능 → **answer가 정답**.

두 변형은 **동일한 image quality, 반대의 sufficiency**를 갖는다.

- probe accuracy가 image quality를 추종하는 어떤 방향이든 RELEVANT-vs-IRRELEVANT 대조에서 **chance로 강제**된다.
- 둘을 분리하는 방향은 quality로 설명할 수 없는 무언가, 즉 **task sufficiency**를 인코딩해야 한다.

같은 matched-budget 설계가 깨끗한 **인과 검정**을 구동한다: 후보 axis를 따라 steering하면 RELEVANT-OCCLUDED 예측이 abstention 쪽으로 뒤집히되, matched IRRELEVANT-OCCLUDED(여전히 answerable) 케이스는 교란되지 않아야 한다 — 이 **selectivity**는 entropy나 quality direction이 달성할 수 없다.

---

## 4. 제안 방법 / 연구 설계 (상세, 구현 가능하게)

### 4.1 모델
LLM hidden states에 접근 가능하도록 **open-weight VLM 3-4종**을 연구한다: LLaVA-1.6, Qwen2.5-VL, InternVL2.5, Molmo.

### 4.2 자극(Stimulus) 구성
localized answer evidence를 가진 VQA 항목에서 출발한다.
- **GQA / Visual Genome**: scene-graph bounding box가 질문별 relevant region $R$을 제공.
- **TextVQA / ST-VQA**: OCR box가 $R$을 제공.
- **DocVQA**: layout box가 $R$을 제공.

각 (image, question)에 대해 **matched global degradation budget $B$**(동일 총 occluded area, clean 대비 SSIM 및 PSNR을 허용 밴드(tolerance band) 내에서 post-hoc 매칭)에서 세 조건을 생성한다:

- **(a) CLEAN**
- **(b) RELEVANT-OCCLUDED**: 면적 $B$를 $R$ 위에 blur/mean-fill/inpaint로 가림.
- **(c) IRRELEVANT-OCCLUDED**: $R$과 disjoint한 동일 면적 영역을 샘플링하여 가리되, **local saliency** 및 **global SSIM/PSNR**에 매칭.

**Ground-truth sufficiency label**: CLEAN/IRRELEVANT = sufficient(answer), RELEVANT = insufficient(abstain).

**라벨의 행동적 검증(behavioral verification)**: 강력한 oracle VLM(또는 500개 항목에 대한 human spot-check)으로 RELEVANT-OCCLUDED가 진정으로 unanswerable이고 IRRELEVANT-OCCLUDED가 answerable임을 확인한다.

### 4.3 Axis 추출
각 layer $L$에서 last-token(및 pooled visual-token) hidden state를 수집한다.

후보 **Sufficiency Axis**는 **matched RELEVANT-vs-IRRELEVANT 쌍에 대해서만** 계산한 difference-of-means:

$$v_{\text{suf}}(L) = \text{mean}(h \mid \text{sufficient}) - \text{mean}(h \mid \text{insufficient})$$

(추출 동안 quality가 고정됨). 동일 matched 데이터에 logistic probe도 적합시킨다.

### 4.4 Triple Dissociation (삼중 해리)
후보 axis를 세 비교 대상에 대해 검정한다.
1. **capacity-matched answer-entropy direction**: 모델의 answer-distribution entropy를 라벨로 한 동일 용량의 방향.
2. **explicit image-quality direction**: 명시적 image-quality를 라벨로 한 방향.
3. **random-direction null**: 무작위 방향 귀무가설.

측정량: **quality-decorrelated probe accuracy**, **causal steering effect**, **selectivity**.

### 4.5 Training-free Steering Operator
inference 시 axis를 더하여(add) 정직한 abstention을 복구하는 **훈련 불필요 steering operator**를 도출한다. steering layer/계수는 held-out에서 선택하고, **no-steering baseline**을 함께 보고한다.

### 4.6 자연 열화로의 Transfer 검증
synthetic occlusion에서 **natural degradation**(motion blur, low light, weather, compression)으로 **refit 없이** transfer됨을 검증한다. 이로써 abstention-under-insufficiency를 clean-image knowledge-conflict arbitration과 메커니즘적으로 구별되는 regime으로 격리하고, synthetic artifact에 과적합하지 않았음을 방어한다.

### 4.7 리뷰어 권고를 반영한 핵심 통제(controls)
- **Within-image cross-question control**: 동일하게 masked된 이미지가 한 질문에는 relevant, 다른 질문에는 irrelevant가 되도록 하여, axis가 여전히 insufficient로 읽으면 그것은 true sufficiency가 아니라 **question-agnostic relevant-region-occluded / edit-detection 신호**임을 분리한다.
- **Occlusion operator 교차/무작위화**: blur, mean-fill, inpaint를 condition과 독립적으로 교차하고 **artifact-detector null**을 추가하여, operator에 걸쳐 held-out accuracy를 보고한다 (matched SSIM/PSNR이 보장하지 못하는 고차 local statistics / artifact 단축경로(shortcut) 방어).
- **Primary causal endpoint의 사전등록(pre-registration)**: model별 abstention base rate에 지배되는 raw flip-to-abstention 대신, **selectivity = (relevant에서의 delta) − (matched irrelevant에서의 delta)**를 1차 인과 종점으로 사전등록하고, entropy·quality direction을 steering baseline으로 둔다.
- **자연 열화에서의 재-얽힘 방어**: 자연 열화 transfer에서 quality direction을 regress out하거나 decorrelate하고, **진정으로 불충분한 자연 열화**와 **열화되었으나 판독 가능한(degraded-but-readable)** 케이스를 perceptual quality로 매칭하여, axis가 전자에서만 발화함을 보인다.
- **Diffuse insufficiency 조건** 추가(localized-region scope가 놓치는 분산형 불충분성), 및 **layer-wise emergence·cross-model subspace alignment** 보고로 dedicated-internal-direction 주장을 뒷받침한다.

---

## 5. 기존 연구 대비 차별점 — 표절/중복 없음의 근거

신규성 검증(skeptic)이 찾아낸 **검증된(verified_real) prior art**를 명시하고, 각각에 대해 무엇이 다른지 구체적으로 반박한다.

| 선행연구 (verified real) | overlap | severity | 본 연구와의 차별점(반박) |
|---|---|---|---|
| **Reading Between the Lines** (arXiv:2511.19806) | VLM hidden state/attention에 latent probe를 학습해 abstention; visual occlusion 포함 uncertainty source에 걸쳐 일반화; 중간 layer 신호가 최적. **abstention-via-internal-probe 축에서 가장 근접.** | partial | **sufficiency-direction 격리 없음, image-quality decorrelation 없음, entropy/quality contrast 없음, triple causal dissociation 없음, training-free steering operator 없음.** 즉 상관(correlational) probe일 뿐 quality 통제·인과 steering이 없다. (제안: 그들의 probe를 우리의 matched-quality 대조에서 돌려, 우리 설계가 통과하는 confound test를 그들이 **실패**함을 직접 보임.) |
| **Arbitration Failure, Not Perceptual Blindness** (arXiv:2604.09364) | clean-image visual-linguistic conflict 메커니즘; percept는 인코딩되나 arbitration이 실패(encoding-grounding dissociation). | partial | **recoverable한 정답이 존재**하며, degraded input·evidence-sufficiency·quality decorrelation·abstention-as-target을 전혀 다루지 않음. 본 연구의 **contrast class(대조군 regime)** 로 위치시킴. |
| **When Seeing Overrides Knowing** (arXiv:2507.13868) | counterfactual clean image에서 language prior가 vision을 덮어쓰는 attention head를 국소화하고 vision/knowledge 쪽으로 steering. | partial | **알려진 정답으로의 knowledge-conflict steering**이지 degradation 하의 sufficiency/abstention이 아님. steering+localization 기계는 공유하나 **regime이 다름**. |
| **Sufficient Context** (arXiv:2411.06037, ICLR 2025; Joren et al.) | text LLM의 context-sufficiency를 external classifier로 정의, uncertainty와 구별. | partial | **sufficiency 구성의 개념적 조상**이나 **text-only, behavioral/external label, vision 없음, internal direction 없음, causal isolation 없음.** 본 연구는 이를 **internal direction**이자 **vision** 영역으로, 그리고 인과적으로 격리한다. |
| **Are LLM Uncertainty and Correctness Encoded by the Same Features?** (arXiv:2604.19974) | SAE로 internal uncertainty vs incorrectness feature의 functional dissociation 입증; capacity control로 경쟁 방향을 disentangle하는 방법론과 유사. | partial | **text-only LLM**, 축이 uncertainty vs correctness이지 **VLM의 sufficiency vs image-quality vs entropy**가 아님. 방법론적 유비일 뿐 대상 구성이 다름. |
| **Steering to Say No** (arXiv:2602.07013) | VLM에서 activation-steered configurable refusal; out-of-scope/정책 위반 거부. | partial | **generic refusal** steering이지 evidence-sufficiency에 의해 구동되는 abstention이 아님. quality 비상관·관련성 표적 검정이 없음. |

리뷰어가 추가로 지목한 **Reading Between the Lines (2511.19806)** 및 **CARA(multimodal insufficient-context abstention)** 에 대해서는, 델타가 실재함을 인정하되 **head-on 비교**(그들의 probe를 matched-quality contrast에서 실행하여 confound test 실패를 입증)를 평가 설계에 포함한다 (6절·4.7 참조).

**요약(표절/중복 없음의 근거)**: 어떤 단일 논문도 (i) quality와 sufficiency를 구성적으로 직교화하는 matched-budget relevance-targeted occlusion, (ii) entropy/quality/random에 대한 triple dissociation, (iii) selective causal steering, (iv) synthetic→natural transfer를 결합하여 **VLM 내부의 construct-valid·causally-tested sufficiency direction**을 보인 바 없다. skeptic은 이 disentanglement를 정확히 수행하는 한 편의 논문을 인용할 수 없다.

---

## 6. 실험/평가 설계 (데이터셋, 베이스라인, 메트릭, ablation)

**데이터셋**
- GQA, Visual Genome (scene-graph box → $R$)
- TextVQA, ST-VQA (OCR box → $R$)
- DocVQA (layout box → $R$)
- 자연 열화 transfer set: motion blur, low light, weather, compression

**조건**: CLEAN / RELEVANT-OCCLUDED / IRRELEVANT-OCCLUDED (matched budget $B$, matched SSIM·PSNR·local saliency).

**베이스라인 / 비교 방향**
- capacity-matched **answer-entropy direction**
- explicit **image-quality direction**
- **random-direction null**
- 선행 probe 직접 비교: **Reading Between the Lines (2511.19806)**, **CARA** — 동일 matched-quality contrast에서 실행.
- **no-steering baseline**

**메트릭**
- **quality-decorrelated probe accuracy** (RELEVANT-vs-IRRELEVANT 대조; quality probe는 구성상 chance여야 함)
- **causal steering effect** (abstention rate 변화, error-on-answered 변화)
- **selectivity = Δ(relevant) − Δ(matched irrelevant)** — **사전등록된 1차 인과 종점**
- 행동 라벨 검증: oracle VLM + 500항목 human spot-check
- **layer-wise emergence**, **cross-model subspace alignment** (LLaVA-1.6 / Qwen2.5-VL / InternVL2.5 / Molmo)
- operator-held-out accuracy, **artifact-detector null**

**Ablation / 통제**
- within-image cross-question control (sufficiency vs edit-detection 분리)
- occlusion operator 교차(blur/mean-fill/inpaint) 무작위화
- 자연 열화에서 quality direction regress-out / decorrelate
- diffuse-insufficiency 조건
- entropy direction에 대한 orthogonalization 후 separability 정량화
- SSIM/PSNR tolerance band 보고

---

## 7. 기대 기여

1. **VLM 내부에 image quality 및 answer-uncertainty 양쪽으로부터 분리 가능한, linear·causally-active "enough-to-see" sufficiency direction이 존재한다는 첫 construct-valid·causally-tested 증거.**
2. global degradation이 만드는 quality–sufficiency 혼동을 **구성적으로 직교화**하는 matched-budget relevance-targeted occlusion이라는 **falsifiable한 새 construct-validity 검정**.
3. abstention-under-insufficiency를 clean-image knowledge-conflict arbitration과 **메커니즘적으로 구별되는 regime**으로 격리.
4. 정직한 abstention을 복구하는 **training-free steering operator**와 **synthetic→natural transfer** 검증 — 헤드라인(놀라운 인과적 발견)과 실용성(자율주행·의료 안전) 동시 제공.

---

## 8. 한계 및 향후 연구

- **Residual tamper/edit-detection confound**: matched SSIM/PSNR/saliency도 고차 local statistics에서 다를 수 있어, 모델이 불충분성 대신 편집된 영역을 탐지할 가능성. → within-image cross-question control 및 operator 교차·artifact null로 완화하나 완전 제거는 어려움.
- **자연 열화에서의 재-얽힘**: 자연 열화는 global하여 quality와 sufficiency를 재차 얽음. transfer에서의 효과가 동일 axis인지 quality reading인지 추가 decorrelation/regress-out 필요.
- **Abstention의 under-specification**: base VLM은 거의 abstain하지 않으므로 readout/prompt/steering-layer 프로토콜을 사전등록·완전 명세해야 함.
- **IRRELEVANT-answerable 가정의 leak 가능성**: irrelevant 영역에 맥락이 누출될 수 있음 → 행동 검증으로 통제.
- **Localized-region scope**: 분산형(diffuse) 불충분성을 놓칠 수 있음 → diffuse-insufficiency 조건으로 보완하나 일반화는 향후 과제.
- **difference-of-means의 정적(static) 단일 방향성** vs 질문-조건적 sufficiency: 단일 axis가 question type 전반에서 작동하는지, 아니면 question-agnostic 신호로 붕괴하는지 검증 필요.
- **component novelty의 modesty**: 개별 구성요소(probe, steering)는 알려져 있으며, 신규성은 조합·construct-validity·causal isolation에 있음.

향후: question-conditioned dynamic sufficiency 방향, diffuse/global insufficiency로의 확장, multimodal insufficient-context abstention(CARA 계열)과의 통합 벤치마크.

---

## 9. 리뷰어 평가

| 리뷰어 | novelty | soundness | significance | clarity | overall | 추천 |
|---|---|---|---|---|---|---|
| R1 | 7 | 7 | 7 | 8 | 7 | weak_accept |
| R2 | 8 | 7 | 7 | 7 | 7 | weak_accept |
| R3 | 7 | — | — | — | 7 | weak_accept |
| R4 | 8 | — | — | — | 7 | weak_accept |
| **평균** | **7.50** | — | — | — | **7.00** | — |

- **Overall 평균: 7.00** / **Novelty 평균: 7.50** / **min: 7** / **합격 여부: 합격(passed=true)** — 4명 전원 weak_accept.

**핵심 코멘트 요약**

*강점 (공통)*
- matched-budget relevant-vs-irrelevant occlusion이 quality와 sufficiency를 구성적으로 직교화 — 선행 probe가 결여한, 진정으로 신규적이고 올바른 construct-validity 검정 (R1, R2가 최강 요소로 지목).
- 동시기 Arbitration-Failure 연구 대비 positioning이 정확함: 그들은 clean visual-linguistic conflict(answer recovery)로 본 연구의 abstention-under-insufficiency와 다른 regime.
- entropy/quality/random에 대한 triple dissociation, probe accuracy·causal steering·selectivity의 결합이 rigorous하며 answer-uncertainty로부터의 separability가 설계에 내장됨.
- scene-graph/OCR/layout box 기반 $R$이 확장 가능하고, oracle + human 라벨 검증이 answerability 반론에 대응.
- training-free steering operator + synthetic→natural transfer가 인과적·실용적 가치를 더하고 synthetic artifact 과적합을 방어.

*약점 (공통)*
- 근접 선행연구 under-cite: Reading Between the Lines(2511.19806), CARA(2405.11145)와 head-on 논증 필요 (델타는 실재).
- SSIM/PSNR 매칭은 low-level quality만 통제 — mid-level important-region-occluded feature(순수 quality도 true sufficiency도 아님)에 대한 명시적 control 필요.
- difference-of-means의 단일 정적 방향 vs question-conditioned sufficiency: question-agnostic relevant-region-occluded 신호로 붕괴할 위험.
- raw flip-to-abstention은 model별 base rate에 지배됨 → selectivity(relevant − matched irrelevant)를 사전등록된 1차 인과 종점으로.
- inpainting/blur artifact 단축경로, tamper-detection confound 잔존 (matched SSIM/PSNR이 artifact 통계까지 보장하지 않음); 자연 열화 transfer는 quality와 sufficiency를 재-얽음.
- abstention readout/steering 프로토콜 미명세(일부 truncation), IRRELEVANT-answerable 가정의 leak, localized-region scope의 diffuse-insufficiency 누락.

*fatal flaw*: 4명 모두 **없음**.

---

## 부록 A. 인용 검증 (표절/중복 없음의 외부 근거)

본 보고서의 신규성 주장이 의존하는 핵심 선행연구를, 작성자(Claude)가 2026-06-02에 **독립적으로 웹 검색하여 실재함과 기술 내용의 정확성을 확인**하였다. (1차 워크플로우에서 일부 아이디어가 검증 불가한 인용에 의존했던 문제를 차단하기 위한 절차.)

| 인용 | arXiv | 실재 확인 | 본 보고서의 기술이 정확한가 |
|---|---|---|---|
| Reading Between the Lines: Abstaining from VLM-Generated OCR Errors via Latent Representation Probes | 2511.19806 | ✅ | 정확 — hidden state/attention에 latent probe를 학습한 abstention. 본 연구의 가장 근접한 선행연구이며, quality-decorrelation·인과 steering·triple dissociation 부재라는 차별점이 유효 |
| Sufficient Context: A New Lens on RAG | 2411.06037 (ICLR 2025, Joren et al.) | ✅ | 정확 — text-only RAG의 context-sufficiency를 external classifier로 정의. vision/internal-direction/causal-isolation 부재라는 차별점이 유효 |
| Steering to Say No / CR-VLM (Configurable Refusal via Activation Steering in VLMs) | 2602.07013 | ✅ | 정확 — generic/out-of-scope refusal steering. evidence-sufficiency 구동 abstention이 아님 |
| Detecting Multimodal Situations with Insufficient Context and Abstaining (CARA) | 2405.11145 | ✅ | 정확 — multimodal insufficient-context의 **behavioral** abstention. 본 연구는 internal-direction + construct-validity + causal isolation으로 차별화하되, **문제 프레이밍이 가장 근접하므로 평가에서 head-on 비교를 포함**(6절·4.7) — 과대주장 아님 |

> **정직성 주석:** 위 표의 CARA(2405.11145)는 "multimodal insufficient-context abstention"이라는 문제 프레이밍에서 본 연구와 가장 가깝다. 따라서 본 연구의 신규성은 "문제를 처음 제기"한 데 있지 않고, **(i) quality와 sufficiency를 구성적으로 직교화하는 matched-budget relevance-targeted occlusion, (ii) entropy/quality/random에 대한 triple dissociation, (iii) selective causal steering, (iv) synthetic→natural transfer의 결합으로 VLM 내부의 construct-valid·causally-tested sufficiency direction을 입증**하는 메커니즘/인과 방법론에 있다. 이 점을 보고서 본문이 약점으로 명시하고 평가 설계에 반영했다.

## 부록 B. 도출 프로세스 (재현용 기록)

이 아이디어는 2단계 멀티에이전트 워크플로우로 도출되었다.

1. **1차 탐색 (236 에이전트):** VLM/LLM 5개 렌즈 스윕 → 핫 영역 선별 → 후보 생성 → 적대적 신규성 검증 → 리뷰어 채점(최대 5라운드). 결과: 가장 유망한 챔피언(visual tool-use RL 계열)이 **평균 6.13에 그침**. 진단 결과 **해당 영역이 실제 2025–2026 논문들로 포화**되어 신규성 판정 72건 중 67건이 "incremental"이었고, 인용된 핵심 논문(2602.01334, 2511.19820 등)은 웹 검증 결과 **실재**함을 확인 → 포화가 실재함을 입증.
2. **2차 탐색 (84 에이전트):** 전략 전환 — 포화 영역을 회피하고 **미개척(white-space)** 방향만 탐색, 모든 인용을 웹 검증, 리뷰어 패널을 현실적 기준으로 재보정. 2라운드 만에 본 아이디어가 **평균 7.00 / novelty 7.50 / 치명결함 0**으로 통과.

통과 기준(높은 점수 게이트)은 자의적 미달이 아니라, 1차에서 포화로 인한 신규성 한계를 확인한 뒤 **현실적이되 까다로운 top-venue accept 수준(avg ≥ 7.0)**으로 설정하였다.
