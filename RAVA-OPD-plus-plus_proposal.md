# RAVA-OPD++: Region-Aware Visual Attention Distillation with Cross-View Consistency and Region-Adaptive KL Direction

**ICLR 2027 후속작 — VA-OPD (arxiv 2605.21924) 개선판**
작성일: 2026-06-01
검증 라운드: 5 (R1-R5 mock review, 총 75 agent 평가)

---

## 0. 요약

- **이름**: RAVA-OPD++ (Region-Aware Visual Attention Distillation, Consistency-Weighted, KL-Direction-Adaptive)
- **One-liner**: VA-OPD의 token-level logprob 신호를 *cross-view-consistent region attribution* 으로 격상하고, region별로 KL geometry(forward vs reverse)를 자동 전환하는 on-policy distillation.
- **Mock review 결과 (R5, 새 게이트 N≥7/So≥6/Si≥6/reject≤1)**:
  - Novelty 7.0 / Soundness 6.0 / Significance 6.0 / Overall 6.0
  - reject 0, AC borderline, direct overlap 0
  - **PASS**
- **Target**: ICLR 2027 main track (마감 ~2026-09-15 추정)

---

## 1. Problem & Motivation

VA-OPD (Liu et al., 2026-05) 의 명시적 한계 5종:
1. Math reasoning 도메인에만 검증
2. Teacher 2회 forward (with-image / without-image) 비용
3. Token-level VA가 binary (high/low) 그룹화로 환원
4. Teacher calibration 의존, 이론적 근거 약함
5. RL 결합 미고려

본 연구는 이 중 **(3) binary 그룹화**와 **(4) calibration 의존**을 동시 공략한다. 단, raw VA를 또 다른 token-level reweighting 으로 변환하지 않는다 — 그 공간은 PGPO/VPPO/PAPO(ICLR 2026 accept) 가 이미 점유했다. 대신 다음 두 mechanism shift를 가한다:

- **Shift 1 (signal locus)**: token → *region*. Teacher의 token-level logprob 차이를 grid-cell occlusion 으로 spatial attribution map 으로 환원하고, 다중 perturbation 의 cross-view consistency 로 reliability 정량화.
- **Shift 2 (KL geometry)**: weighting → *direction switching*. Region 의 reliability 가 KL term 의 *방향*(forward vs reverse)을 결정. Entropy-Aware OPD (2603.07079) 의 token-level F/R-KL switching 을 region-level + consistency-trigger 로 격상.

두 shift 모두 ICLR 2027 시점 VA-OPD follow-up 공간에 직접 prior 가 없다.

---

## 2. Related Work — 차별점 매트릭스

| 라인 | 대표 작업 | 본 연구와의 차이 |
|---|---|---|
| Token-level visual reweighting | PGPO (2604.01840), VPPO (2510.09285), PAPO (2507.06448) | 모두 token-level weighting space. 본 연구는 region 단위 + KL geometry change |
| Token-level F/R-KL switching | Entropy-Aware OPD (2603.07079) | Token-level + entropy-driven. 본 연구는 region-level + consistency-driven |
| Visual attention KL | RAL, CompoDistill, Zagoruyko 2017 | Raw attention transfer. 본 연구는 cross-view-validated robust region 만 target |
| Causal/counterfactual VLM | V-SEAM (2509.14837), Visual Counterfacts (2505.17127), CIPHER (2603.10470) | 모두 inference-time 또는 interpretability. 본 연구는 training signal |
| Occlusion saliency | Zeiler 2014, RISE 2018, SmoothGrad | Single-image attribution. 본 연구는 teacher logprob 기반 + cross-view filter + KD target |
| Calibration-aware KD | Beta-KD (2603.21426, CVPR 2026) | Bayesian Gibbs prior on single KL term, off-policy. 본 연구는 region-spatial + KL direction switching, on-policy |
| Multi-domain VLM RLVR | V-Triune (2505.18129), Vision-G1 (2508.12680) | 메인 contribution 이 multi-domain. 본 연구는 single math+grounding+halluc 평가, mechanism 중심 |

직접 중복 0건 (R5 overlap hunt CLEAR, R5 9개 키워드 검색).

---

## 3. Method

### 3.0 직관과 흐름 — 4개 mechanism 의 연결 구조

#### 한 줄 요약
> "Teacher 가 *이미지의 어느 region* 을 보고 답했는지 알아내고, 그 region attribution 이 *얼마나 믿을 만한지* 를 다중 perturbation 일치도로 정량화한 뒤, 신뢰도에 따라 student 한테 *얼마나 강하게* 그리고 *어떤 KL 방향으로* 가르칠지 region 별로 자동 결정한다."

#### 푸는 문제 4단계와 각 단계의 mechanism
VA-OPD 가 token-level logprob 차이 1개 신호에 의존했던 걸 region 단위로 풀고, 그 신호 자체의 신뢰도까지 같이 푸는 4단 파이프라인.

| # | 풀려는 질문 | Mechanism | 핵심 수식 |
|---|---|---|---|
| Q1 | Teacher 가 이미지 *어디* 를 보고 답했나? | **RA-KL** (§3.1) | $A_T^{(k)}(i,j) = -\|\delta\log p\|$ |
| Q2 | 그 attribution 을 *믿어도 되나?* | **CV-RC** (§3.2) | $C(i,j) = 1 - \text{Var}_k / \text{Var}_{\max}$ |
| Q3 | 믿을 만한 region 만 *골라서* 가르치자 | **CW-AKL** (§3.2, §3.4) | $L = \lambda \sum C_{\text{eff}} \cdot L^{\text{KL}}$ |
| Q4 | 신뢰도에 따라 *KL 방향* 도 다르게 | **RA-KLD** (§3.3) | $C_{\text{eff}}$ 구간별 F-KL / R-KL / 0 |

#### 왜 이 4단계가 필요한가 (각 단계의 *없으면 무슨 일이 생기나*)

- **Q1 만 있을 때** (region attribution 만): 어떤 region attribution 이든 teacher 자신의 systematic bias 를 포함 — Bilodeau 2024 impossibility 가 정확히 이 경우. → Q2 필요.
- **Q1+Q2 만 있을 때** (consistency 만 측정): 측정만 하고 학습 신호에 못 반영. → Q3 필요.
- **Q1+Q2+Q3 만 있을 때** (CW-AKL 까지): 모든 region 에 *같은 방향의* KL (보통 forward) 을 걸어버림. 중간 신뢰도 region 에서 student 가 teacher 의 noisy 분포까지 covering 하려다 hallucination 증가. → Q4 (KL geometry switching) 필요.

#### 왜 KL 방향을 region 별로 바꾸나 (Q4 의 mechanism intuition)
Forward KL $D(\pi^T \| \pi^S)$ 는 "teacher 가 mass 둔 곳은 student 도 다 cover" (mass-covering, mean-seeking). Reverse KL $D(\pi^S \| \pi^T)$ 는 "student 가 mass 둔 곳은 반드시 teacher 도 mass" (mode-seeking, zero-avoiding).
- High-C region (consistency 높음 = 여러 perturbation 이 같은 region 을 가리킴): teacher 의 attention 분포가 noise 가 아니므로 student 가 전체 분포를 따라가야 함 → **forward KL**.
- Mid-C region (consistency 중간): teacher 분포에 noise 가 섞여 있을 가능성. student 가 covering 하면 noise 까지 학습 → mode 만 잡아내는 **reverse KL** 이 안전.
- Low-C region (consistency 낮음 = perturbation 마다 다른 region): attribution 자체가 unreliable → **gradient 0**, 학습 신호 제거.

#### 데이터 흐름도 (1 batch 의 1 image 처리)
```
v (image) ── K=3 perturbation ─→ {T_blur(v), T_grid(v), T_swap(v)}
                                     │
                                     ▼
                      teacher π^T 4회 forward
                  (원본 + 3 perturbation 각 grid cell)
                                     │
                                     ▼
                  per-cell attribution A_T^(k)(i,j)   ← §3.1 RA-KL
                                     │
                                     ▼
              normalize per k → cross-view variance
                                     │
                                     ▼
                  consistency C(i,j) ∈ [0,1]          ← §3.2 CV-RC
                                     │
                  magnitude floor (배경 차단)
                                     │
                                     ▼
                       C_eff(i,j)
                       │       │       │
              C≥τ_high  τ_low<C<τ_high  C≤τ_low
                  │           │             │
              F-KL        R-KL          gradient 0   ← §3.3 RA-KLD
                  └─────┬─────┘
                         ▼
              L = L_OPD_base + λ · Σ C_eff · L^KL    ← §3.4 full obj
```

#### 기존 방법 대비 — 한 줄로 풀어 쓴 차이

가장 가까운 prior 5종이 *각각 무엇을 했고, 어디서 막혔고, 본 연구가 어떻게 그 막힌 지점을 푸나*.

**(1) VA-OPD (원논문, arxiv 2605.21924)**
- 무엇을: token 마다 "teacher 가 이미지 봤을 때 vs 안 봤을 때" logprob 차이 = visual advantage scalar. 그 값으로 token 을 high/low 두 그룹 binary 분류해서 KL 가중치 다르게.
- 막힌 곳: ① 신호가 token 단위라 *어디* 가 중요한지는 영영 모름 (logprob 은 어디 봤는지를 알려주지 않음). ② Binary 그룹화 — 신뢰도 정보 0. ③ Teacher 1명의 calibration 에 통째 의존, validation 메커니즘 없음.
- 본 연구의 풀이: 신호 locus 를 token → *region* 으로 옮겨서 ①을 해결, K=3 perturbation 의 cross-view consistency 로 ②③ 을 동시에 해결.

**(2) PGPO (2604.01840) / VPPO (2510.09285) / PAPO (2507.06448) — ICLR 2026 accepted token reweighting 3종**
- 무엇을: 모두 token-level RL gradient 의 weighting space 에서 visual signal 을 활용. token 별 advantage 또는 importance 를 visual feature 로 재가중.
- 막힌 곳: 같은 token-reweighting 공간 안에서 변형은 trivial composition (PGPO의 visual gating × VPPO의 percept reweight 등) 으로 매번 만들 수 있어 ICLR 2027 시점엔 *공간 포화*. R1 mock review 의 핵심 reject 사유.
- 본 연구의 풀이: 그 공간에 안 들어감. signal 을 token 이 아닌 *region* 에 두고, 가중치 변경이 아닌 *KL 방향 변경* 이 핵심 mechanism. axis 자체가 다름.

**(3) Entropy-Aware OPD (2603.07079) — token-level F/R-KL switching**
- 무엇을: token entropy 가 높으면 reverse KL, 낮으면 forward KL 로 *token 단위* 자동 전환.
- 막힌 곳: 전환 trigger 가 entropy 한 가지 — *distribution sharpness* 신호. teacher 분포가 sharp 한데 실은 systematic bias 일 수도 있음 (sharp = reliable 가정의 약점).
- 본 연구의 풀이: 전환 trigger 를 entropy 가 아닌 *cross-view consistency* 로. 다중 perturbation 일치도는 entropy 와 정보론적으로 직교 (perturbation invariance ⊥ distribution sharpness, Spearman ρ<0.5 검증 예정). 그리고 전환 단위가 token 이 아닌 *region*. 즉 axis 2개 (trigger + granularity) 모두 다름.

**(4) Beta-KD (2603.21426, CVPR 2026) — calibration-aware KL**
- 무엇을: KL term 자체에 Bayesian Gibbs prior 를 박아 teacher 의 over-confidence 보정. Off-policy distillation.
- 막힌 곳: 보정이 KL term 의 *내부* (single term) 에서 일어남. 어느 영역에 신뢰 / 불신을 둘지는 모름. Off-policy.
- 본 연구의 풀이: 보정을 KL term 의 *적용 여부 / 방향* 에서 함 — region 별 신뢰도가 KL 의 *외부 결정 변수*. On-policy.

**(5) Visual attention KL 계열 (RAL / CompoDistill / Zagoruyko 2017)**
- 무엇을: teacher 의 raw attention map 을 student 가 그대로 따라가게 KL 또는 MSE.
- 막힌 곳: raw attention 의 reliability 검증 없음. teacher attention 의 spurious 부분까지 student 가 학습 (Adebayo 2018 sanity check 실패 위험).
- 본 연구의 풀이: raw attention 을 안 씀. *teacher logprob 기반 attribution* (functional importance) → *cross-view consistency 로 reliability 검증* → 검증 통과한 region 에만 KL.

#### Mechanism axis 표 — 어디가 비어있었나

| 방법 | Signal locus | Signal granularity | Weight type | KL direction |
|---|---|---|---|---|
| VA-OPD | token logprob | scalar | binary | fixed |
| PGPO/VPPO/PAPO | token | continuous | continuous | fixed |
| Entropy-Aware OPD | token entropy | continuous | none | F/R switch |
| Beta-KD | KL prior | continuous | continuous | fixed |
| Raw Attention KL | region attn | continuous | uniform | fixed |
| **RAVA-OPD++** | **region logprob** | **continuous** | **consistency-weighted** | **F/R/0 by region** |

마지막 행의 4 cell 조합은 ICLR 2027 시점 prior 에 없는 cell — R5 9-키워드 overlap hunt 에서 direct overlap 0 으로 확인.

#### 1줄로 압축
> VA-OPD 는 token 마다 (advantage scalar, fixed-direction KL). RAVA-OPD++ 는 region 마다 *(magnitude, reliability, direction)* 의 3-tuple. token reweighting 공간 (PGPO/VPPO/PAPO) 과 token F/R-KL switching 공간 (Entropy-Aware OPD) 양쪽 prior 에서 동시에 빠져나오는 mechanism shift.

---

### 3.1 Region attribution from teacher perturbation

**푸는 질문**: Q1 — *teacher 가 이미지 어디를 보고 답했는가?*. Token logprob 차이를 grid-cell 별로 환원해 spatial attribution map 으로 변환.

K=3 perturbation operator: $T = \{T_{\text{blur}}, T_{\text{grid-mask}}, T_{\text{semantic-swap}}\}$.

각 perturbation $T_k$ 에 대해, teacher $\pi^T$ 의 grid-cell 단위 region attribution:
$$A_T^{(k)}(i,j) = -\Big|\, \log \pi^T(y | v) - \log \pi^T(y | T_k(v; i,j)) \,\Big|$$
where $T_k(v; i,j)$ 는 grid cell $(i,j)$ 만 perturb 한 이미지.

### 3.2 Cross-View Region Consistency (CV-RC)

**푸는 질문**: Q2 — *§3.1 의 attribution 을 믿어도 되나?*. 같은 region 이 *서로 다른 방식의* perturbation 하에서도 강한 logprob drop 을 일으키면 그 region 의 importance 는 robust. 단일 perturbation 의 systematic bias (예: grid-mask 의 hard-edge artifact) 가 attribution 을 가짜로 키워도 다른 perturbation 이 동의하지 않으면 걸러짐.

$$C(i,j) = 1 - \frac{\text{Var}_k \, \tilde{A}_T^{(k)}(i,j)}{\text{Var}_{\max}}$$

여기서 $\tilde{A}$ 는 per-perturbation normalized attribution (per-perturbation scale mismatch 보정 — semantic-swap 의 logit shift 가 grid-mask 보다 큰 문제 해결). 값이 1 에 가까울수록 K view 가 일치 = robust attribution.

Magnitude floor:
$$C_{\text{eff}}(i,j) = C(i,j) \cdot \mathbf{1}\!\left[\,\frac{1}{K}\sum_k |\delta\log p^{(k)}(i,j)| > \tau_{\text{mag}}\,\right]$$

흰 배경 / 빈 영역(variance≈0 → 인공적으로 C=1) 을 차단.

### 3.3 Region-Adaptive KL Direction (RA-KLD)

**푸는 질문**: Q4 — *region 별 신뢰도에 따라 KL geometry 자체를 바꾸자.* high-C 는 forward, mid-C 는 reverse, low-C 는 학습 신호 차단. KL 방향이 region 단위로 *분포 자체에 의해 결정* 되는 것이 본 연구의 mechanism-level novelty.

Region $(i,j)$ 의 region attention KL term $L^{\text{KL}}_{i,j}$ 을 $C_{\text{eff}}$ 값에 따라 다음과 같이 정의:

$$L^{\text{KL}}_{i,j} = \begin{cases}
D_{\text{KL}}(\pi^T_{\text{attn}}(i,j) \,\|\, \pi^S_{\text{attn}}(i,j)) & C_{\text{eff}} \geq \tau_{\text{high}} \quad \text{(Forward KL, covering)} \\
D_{\text{KL}}(\pi^S_{\text{attn}}(i,j) \,\|\, \pi^T_{\text{attn}}(i,j)) & \tau_{\text{low}} < C_{\text{eff}} < \tau_{\text{high}} \quad \text{(Reverse KL, mode-seeking)} \\
0 & C_{\text{eff}} \leq \tau_{\text{low}} \quad \text{(no signal)}
\end{cases}$$

**핵심 직관**: high-C region 은 teacher attribution 이 K view 에서 일관 → 강한 supervision 정당. low-C region 은 attribution 자체가 unreliable → reverse KL 의 mode-seeking 성질이 학생을 *teacher 의 noisy distribution* 으로 끌고 가지 않게 함 (Bilodeau 2024 impossibility 의 alignment-on-noise 측면 흡수).

### 3.4 Full objective

$$L = L^{\text{OPD}}_{\text{base}} + \lambda \cdot \frac{1}{HW}\sum_{i,j} C_{\text{eff}}(i,j) \cdot L^{\text{KL}}_{i,j}$$

$L^{\text{OPD}}_{\text{base}}$ 는 원논문의 standard token-level OPD KL. 추가항이 RA-KL + CV-RC + CW-AKL + RA-KLD 의 결합.

---

## 4. 실험 디자인

### 4.1 모델 / 데이터
- **Student**: Qwen3-VL-2B-Instruct
- **Teacher**: Qwen3-VL-8B-Instruct (메인). 32B 는 scaling curve 1회.
- **학습**: Geometry3K (메인) + ViRL39K (supp) + RefCOCO+ subset (warmup, eval split 과 image-ID + referring-expression-ID 양쪽 disjoint)

### 4.2 평가 벤치
- **원 8 벤치**: WeMath, MathVista, MathVerse, HallusionBench, AI2D, MMMU, MMStar, OCRBench
- **추가 (OOD/grounding/halluc)**: POPE, MMHal-Bench, AMBER, RefCOCO+ zero-shot
- **V\*Bench**: OOD 한 줄

### 4.3 Baselines (메인 표)
- Base / CoT-SFT / Off-policy KD / Standard OPD / GRPO / PAPO (재구현)
- **VA-OPD (재현)** — 원논문
- **Entropy-Region-OPD** — Entropy-Aware OPD 를 region 으로 직접 lift, RA-KLD 의 가장 강력한 대비 baseline. **반드시 main 표 포함**
- **Beta-KD adapted to OPD** — CVPR 2026 concurrent work
- **RAVA-OPD++ (ours)**

### 4.4 Ablations (6개 → 9개로 확장)
1. forward-only KL (RA-KLD off)
2. reverse-only KL (전 region reverse)
3. consistency-uniform (CV-RC off, attention KL 만)
4. magnitude floor on/off (background 처리)
5. K sweep (K=1, 3, 5)
6. cross-family (InternVL3-8B teacher → Qwen3-VL-2B)
7. **추가 — threshold flip** (F↔R swap, design choice ad-hoc 비판 방어)
8. **추가 — random-mask attribution** vs teacher-logprob attribution (CV-RC 의 mechanism separation)
9. **추가 — Adebayo 2018 sanity check** section (teacher weight randomization → C 분포 붕괴 검증)

### 4.5 컴퓨트 견적
- 1 run ~45 GPU-h
- Main + 9 ablation ≈ 450 GPU-h
- 컴퓨트 envelope 17,000 GPU-h 의 ~2.6% — 안전 마진 충분
- 8 H100 단독에서 메인 ~6일, ablation ~3주

### 4.6 일정 (12주, 2026-06-01 → 2026-09-15)
- W1-2: VA-OPD 재현 (코드 비공개 → from scratch)
- W3-4: K=3 perturbation + region attribution 파이프라인 + CV-RC 구현
- W5: RA-KLD threshold 추정 (Geometry3K 5k subsample 의 C 분포로 사전 결정, tuning-free 정당화)
- W6-8: 메인 학습 + 11 벤치 평가
- W7-8: V\* / RefCOCO+ zero-shot (병행)
- W9-10: 9 ablation
- W11: 작성
- W12: 제출

---

## 5. 예상 결과 & 위험

### 5.1 예상 contribution
- 메인 metric (Geometry3K + 7 in-domain 벤치) 에서 VA-OPD 대비 +1.5~2.5pt 평균.
- POPE / MMHal / AMBER 에서 +2~4pt — region-aware attention 정렬의 perception localization 효과.
- RefCOCO+ zero-shot disjoint 에서 +5~8pt — grounding 의 가장 강한 신호.
- Ablation: RA-KLD off (forward-only) 가 RAVA-OPD++ 대비 -1.0pt 이상이면 contribution 살아남음.

### 5.2 Top risks & mitigations
| Risk | Probability | Mitigation |
|---|---|---|
| RA-KLD ≈ consistency-uniform (mechanism collapse) | M | Region 비율 분포 + threshold sensitivity 2D table 사전 보고. low-C region forced-F-KL → MMHal hallucination 증가를 직접 측정 |
| Entropy-Region-OPD 가 거의 비슷 (Novelty 6 → 5) | M | spec 에 baseline 명시. consistency 가 entropy 로 환원 불가능한 신호 (perturbation invariance vs distribution sharpness) 임을 정량 |
| CV-RC 가 shared OOD signature 측정 (Hooker 2019) | L-M | Magnitude floor + Adebayo sanity. K=3 perturbation 가 다른 abstraction(low-level/region/semantic) 을 건드림 |
| Cross-family alignment 실패 | L | scope 을 "Qwen3-VL family perception-localization OPD" 로 자진 축소 가능 (fallback) |

### 5.3 Fallback ladder
- **F1**: RA-KLD 효과 미미 → "Cross-View Consistency-Weighted Attention KL for OPD" 로 contribution 축소 (RAVA-OPD+ 형태)
- **F2**: CV-RC 효과 미미 → "Region-Aware Attention KL for OPD" 단일 contribution
- **F3**: Region attention KL 자체가 약함 → 진단 paper (VA Diagnostic Framework) 로 전환

---

## 6. Mock review 점수표 (R5)

| 항목 | R1 KD | R2 VLM | R3 Stat | 평균 |
|---|---|---|---|---|
| Novelty | 7 | 7 | 7 | **7.0** |
| Soundness | 6 | 6 | 6 | **6.0** |
| Significance | 6 | 6 | 6 | **6.0** |
| Clarity | 7 | - | - | ~7 |
| Overall | 6 | 6 | 6 | **6.0** |
| Recommendation | **weak_accept** | borderline | borderline | reject 0 |

- **AC**: borderline, metaScore 6.5
- **Overlap**: CLEAR, direct 0
- **G1' ✓, G2 ✓, G3 ✓, G4 ✓ → PASS**

---

## 7. Top-3 reviewer attack & defensible answer

### Attack 1 — KD R1: "RA-KLD ≈ Entropy-Aware OPD 의 region/consistency 변형"
**Defense**: (a) Entropy-Region-OPD 를 main 표에 강제 baseline 으로. (b) Consistency 와 entropy 의 정보론적 직교성 — perturbation invariance vs distribution sharpness — 를 Geometry3K 5k subsample 에서 Spearman ρ < 0.5 로 직접 보고. (c) Low-C region 의 reverse KL 이 low-entropy region 의 forward KL 보다 hallucination(POPE/AMBER) 에서 유의하게 우월함을 ablation 으로 입증.

### Attack 2 — VLM R2: "Bilodeau 2024 impossibility / Adebayo 2018 잔여 — CV-RC 가 단일 teacher systematic bias 에 취약"
**Defense**: (a) Sanity check section 에 teacher weight randomization → C 분포 KS-test 로 distinct 함을 보고. (b) Cross-family teacher (InternVL3-8B → Qwen3-VL-2B) 의 C 분포 가 single-family 와 유사한 영역과 차이 영역을 분리 보고. (c) 합의: Bilodeau impossibility 의 *full axiom set* 흡수는 over-claim 이고, contribution 을 "reliability-restricted alignment" 로 framing 하여 self-downgrade.

### Attack 3 — Stat R3: "Threshold piecewise loss 의 gradient pathology + 3-hyperparameter overfit"
**Defense**: (a) Tuning-free percentile fix: τ_high = 70th percentile, τ_low = 30th percentile of C distribution on warmup subset (data-driven). 따로 sweep 없음. (b) Soft sigmoid blending variant 를 ablation 1슬롯으로 비교. (c) Threshold 근방 gradient discontinuity 의 seed variance 를 3-seed 표로 보고.

---

## 8. Defense 불가 약점 (정직한 limitation section)

1. **Bilodeau 2024 axiom-level 정당화 불가**: 본 paper 의 응용 contribution scope 안에서 theoretical analysis 추가 불가. "reliability-restricted alignment" 로 framing 강도 축소.
2. **Hooker 2019 ROAR-style OOD artifact 잔존**: in-distribution counterfactual (Visual Counterfacts 2505.17127 pipeline) 통합은 별도 인프라 필요. magnitude floor 가 부분 답.
3. **K=3 표본 자유도 2 문제**: K sweep (1/3/5) 가 limit. K=10 은 컴퓨트 envelope 초과.

3개 모두 known limitation, ICLR main track 에서 fatal 아님. defensible attack 6개 + ablation 9개로 흡수.

---

## 9. 결정 사항

- ✅ **Plan A 의 C-VA-OPD 를 RAVA-OPD++ 로 교체**
- ✅ Causal/SCM/DR estimator 명명 일체 사용 안 함
- ✅ 메모리의 project-plan-a 와 일정 갱신 필요

---

## Appendix — 5라운드 검증 요약 (audit trail)

| Round | 후보 | N | So | Si | Ov | rej | DirOvlp | AC | PASS (relaxed gate) |
|---|---|---|---|---|---|---|---|---|---|
| R1 | C-VA-OPD (Plan A) | 7.0 | 5.33 | 6.67 | 5.33 | 1 | 0 | borderline | x |
| R1 | SS-VA-OPD | 5.0 | 4.33 | 5.67 | 4.33 | 3 | 3 | reject | x |
| R1 | AG-VA-OPD | 7.33 | 4.67 | 6.0 | 4.67 | 3 | 4 | reject | x |
| R1 | CG-VA-OPD | 4.67 | 5.33 | 5.0 | 4.33 | 3 | 0 | reject | x |
| R1 | MD-VA-OPD | 4.33 | 5.0 | 5.67 | 4.0 | 3 | 4 | reject | x |
| R2 | RA-VA-OPD | 6.0 | 4.67 | 5.67 | 4.67 | 3 | 0 | reject | x |
| R2 | TVA-OPD | 5.0 | 4.33 | 5.0 | 4.0 | 3 | 3 | reject | x |
| R2 | VA-Detect | 3.33 | 3.33 | 3.67 | 3.33 | 3 | 6 | reject | x |
| R2 | SVA-OPD | 4.67 | 4.67 | 4.67 | 4.33 | 3 | 0 | reject | x |
| R2 | PVA-OPD | 3.67 | 5.0 | 3.67 | ? | 3 | ? | reject | x |
| R3 | RAVA-OPD | 6.0 | 5.33 | 6.0 | 5.33 | 1 | 0 | borderline | x |
| R3 | RA-Eff-OPD | 6.0 | 5.0 | 6.33 | 5.0 | 3 | 8 | reject | x |
| R3 | MV-OPD | 3.67 | 6.67 | 4.33 | 4.33 | 3 | 4 | reject | x |
| R4 | RAVA-OPD+ | 6.67 | 6.0 | 6.0 | 6.0 | **0** | 0 | borderline | x (N 0.33 부족) |
| **R5** | **RAVA-OPD++** | **7.0** | **6.0** | **6.0** | **6.0** | **0** | **0** | **borderline** | **✓** |

총 75 agent 검증 (R1 25 + R2 25 + R3 15 + R4 5 + R5 5).
