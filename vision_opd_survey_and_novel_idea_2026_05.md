# Vision-OPD 전수 조사 + 신규 논문 아이디어 / 실험 방향

작성일: 2026-05-28
앵커 논문: **Vision-OPD** = "Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation" (Yuan, Lou, Yu, Lin, Sun, Han, Lu; ISCAS/UCAS/Xiaohongshu; arXiv:2605.18740, v1 2026-05-18)
조사 규모: 앵커 1편 + 5개 클러스터 85+ 논문(2015 LUPI ~ 2026-05 arXiv), 병렬 리서치 에이전트 + 표적 신규성 검증

> 주제 확정: 사용자 확인에 따라 "vision OPD" = **Vision-OPD (On-Policy self-Distillation for MLLMs)**. (3D Openable Part Detection 아님.)

---

## 0. TL;DR (먼저 읽으세요)

1. **"zoom을 가중치에 내재화해 단일 패스로 미세인식"은 2025-2026의 초혼잡(super-crowded) 전장이다.** Vision-OPD는 그중 "**같은 MLLM의 crop-teacher → full-image-student, on-policy self-distillation(토큰별 JSD), EMA teacher, 라벨/검증기/외부teacher 없음**"이라는 좁은 교집합을 점유한다. 같은 목표(zoom→단일패스)를 동시에 추구하는 경쟁군: **ZwZ(Zooming without Zooming, ICML 2026), DeepSketcher, Latent Visual Reasoning(LVR), Monet, SD-RPN, CropVLM**. 텍스트 쪽 동일 메커니즘(privileged-info self-distillation): **OPSD, π-Distill, GATES, ATESD, RLSD**.

2. **이미 점유되어 새 논문거리가 안 되는 것들**(§4 가드레일 표): on-policy KD(GKD), reverse-KL KD(MiniLLM), JSD/대칭발산 KD(f-DISTILL/GKD), self-distillation(Born-Again/BYOT), EMA teacher anti-collapse(Mean Teacher/MoCo/BYOL/DINO), privileged-info distillation(LUPI/Learning-by-Cheating), region-predict→crop→answer(Visual CoT/Chain-of-Spot/MGPO), RL zoom tool(DeepEyes/Pixel Reasoner/Thyme), training-free zoom search(ZoomEye/DC²), "when/where to zoom" 게이트(AdaFocus/AwaRes/Q-Zoom/AdaTooler-V), region→image distillation 단일패스 내재화(ZwZ), 잠재 시각추론(LVR/Monet/DeepSketcher).

3. **그럼에도 일관되게 비어 있는 핵심 공백**: 내재화(internalization)는 evidence가 **full-image 토큰에 이미 존재하지만 주목 안 된 경우(focus-recoverable)**에만 정당하다. evidence가 **인코더 유효 해상도 미만(resolution-bound)**이면, crop-teacher는 2× 업스케일된 crop으로 그 픽셀을 봤지만 student는 단일 패스에서 **물리적으로 그 정보를 복원할 수 없다.** 그런데 student는 teacher의 자신만만한 답 분포에 맞추도록 학습된다 → **resolution-bound 디테일에서 student는 "본 척하며 자신있게 환각"하도록 학습된다.** 이 구분(recoverable vs resolution-bound)을 실증하고, **자기-증류 잔차 발산(residual/irreducible divergence)을 라벨-free "비내재화성(non-internalizability) 신호"로 삼아 선택적 내재화 + abstain/escalate**하는 연구는 없다.

4. **제안(§5)**: 메인 1개 + 대안 3개.
   - **메인 — RA-OPD (Resolution-Aware On-Policy self-Distillation) + 3지(三枝) 게이트**: ① 내재화-zoom이 resolution-bound 디테일에서 자신있는 환각으로 정확도를 위장함을 실증, ② self-distillation 잔차 KL이 plateau하는(=irreducible privileged gap) 곳을 **라벨-free 신호**로 검출, ③ 단일 패스 안에서 **Answer / 한 번 진짜 Zoom / Abstain** 3지 선택을 학습. 이는 사용자의 텍스트 hybrid-reasoning 철학(쉬우면 빠르게, 어려우면 깊게)의 **시각판**이며, 가장 핫한 방법의 신뢰성 구멍을 이론적 근거(irreducible PI gap) 위에서 메운다.
   - **대안 A — 분산 증거(multi/dispersed-evidence) self-distillation**: 다중 crop teacher → 단일 패스 student (ZwZ가 명시적으로 못 푸는 영역).
   - **대안 B — privileged view를 crop 너머로**: depth/segmentation/OCR-overlay/고해상 재인코딩 teacher를 plain student에 self-distill ("OPD for view-X").
   - **대안 C — 내재화 perception의 calibration**: student가 환각할 때 정확히 unconfident하도록 (faithfulness 보장).

5. **사용자 자산 정렬**: 사용자는 LLM/VLM을 처음부터 학습하고, hybrid reasoning(적응적 연산)과 신뢰성/overthinking에 집중. RA-OPD는 그 철학의 직접 시각 이식이며, label-free라 사용자의 closed-source distill 0% 정책과 충돌 없음(자체 모델 forward만 사용).

> 정직한 경고: 이 분야는 월 단위로 논문이 쏟아진다(2026-02~05에만 경쟁군 다수). 아래 제안도 부분적으로 누군가 건드릴 수 있으니, §6 체크리스트로 착수 직전 arXiv 재검색을 반드시 하라.

---

## 1. 조사 범위 및 방법

5개 클러스터 병렬 조사 + 표적 신규성 검증.
- **C1. 증류 방법론**: on-policy/online/self/privileged-information distillation (LLM & VLM).
- **C2. 미세·고해상 시각인식 문제공간 + 벤치마크**: 왜 MLLM이 작은 것을 못 보나, 고해상 아키텍처, 벤치마크.
- **C3. Thinking-with-images / zoom / crop / agentic**: Vision-OPD가 대체하려는 방법군.
- **C4. RL·self-improvement for VLM + faithfulness/hallucination/reliability**.
- **C5(검증)**: irreducible privileged gap, "언제 zoom할지" 게이트, 내재화 perception의 환각.

용어:
- **Internalization(내재화)**: 추론 시 도구/재인코딩 없이 단일 forward pass로 미세인식. zoom의 이득을 가중치로 흡수.
- **On-policy self-distillation**: 같은 모델을 두 conditioning(privileged teacher vs plain student)으로 인스턴스화, student의 자기 생성 rollout 위에서 토큰별 발산 최소화.
- **Privileged information (PI / LUPI)**: 학습 시에만 주어지고 테스트엔 없는 정보(여기선 evidence crop).

---

## 2. 앵커 정밀 해부 — Vision-OPD

**문제정의**: "regional-to-global perception gap" — 같은 MLLM이 fine-grained 질문에서 **evidence crop을 받으면 full image보다 훨씬 잘 답한다**(GPT-5.4·Gemini-3.5-Flash 포함 18–22%p gap). 저자 주장: 병목은 **local 인식 능력 부족이 아니라 relevant evidence에 집중(focus)하지 못함**.

**방법(OPD)**:
- 같은 MLLM에서 두 정책: **teacher p_T** = evidence crop x′ 조건 (privileged), **student p_S** = full image x 조건.
- student가 on-policy rollout 생성 y~p_S(·|x,q). 두 정책이 같은 y를 평가, 토큰별 **JSD(β=0.5)** 최소화(top-K=100 logits). gradient는 student만.
- 손실: L = E[ (1/|y|) Σ_n JSD_β(p_T(·|x′,q,y_<n) ‖ p_S(·|x,q,y_<n)) ].
- **EMA teacher (α=0.05)가 핵심**: 정규화 없으면 정확도 0.59%로 collapse. EMA가 frozen-init·trust-region보다 우수.
- "dense token-level guidance"라 GRPO/DAPO의 sparse-reward stall 회피.

**데이터**: 6.2K 합성 — 작은 bbox(area ratio<τ) 제안 → 영역만으로 답 가능한 질문 생성 → full image에 bbox overlay + crop 2× resize = x′ → Qwen3.5-397B 다수결 합의>0.75 필터.

**결과**: Vision-OPD-9B 79.68% avg(V*/ZoomBench/HR-Bench/MME-RealWorld), Qwen3.5-397B(77.44%)·GLM-4.6V-106B(71.53%)·GPT-5.4·Gemini-3.5-Flash·agentic SenseNova-MARS(73.05%) 능가, 단일 패스. holdout(MMVP/CV-Bench/MMStar/POPE)에서 망각 없음.

**저자 원문 인용**:
- "internalize the benefit of visual zooming without external teacher models, ground-truth labels, reward verifiers, or inference-time tool use."
- "The model's own crop-conditioned behavior can serve as privileged supervision for improving its full-image behavior."
- "By re-evaluating the same trajectory under a cleaner local view, the teacher's token-level distribution naturally encodes sharper attention to fine-grained visual evidence."

**명시적 한계**: 없음(논문이 한계를 적지 않음). 암묵적: 6.2K 합성데이터 의존, 학습 시 dual forward, evidence-dense task 한정. **신뢰성/환각·해상도 바닥(resolution floor)·언제-못-보는지 분석 전무** ← 본 보고서의 공략 지점.

---

## 3. 문헌 지도 (클러스터별, 원문 인용 포함)

### 3.1 증류 방법론 (C1)

**GKD: On-Policy Distillation of LMs** (Agarwal et al., ICLR 2024) — arXiv:2306.13649. on-policy KD의 일반형. student 자기생성 시퀀스에 dense teacher feedback, divergence-agnostic(forward/reverse KL, JSD(β)). "we do not backpropagate through the student's sampling distribution." → Vision-OPD의 골격. **단, teacher는 고정 외부 대형모델, EMA 없음, self/privileged-view 아님, 텍스트.**

**MiniLLM** (Gu et al., ICLR 2024) — arXiv:2306.08543. reverse KL: "encourages the student to generate samples preferred by the teacher within its own capacities." 외부 teacher, EMA 없음.

**DistiLLM / DistiLLM-2** (Ko et al., ICML 2024/2025) — arXiv:2402.03898, 2503.07067. skew-KL + adaptive off-policy / contrastive. 외부 teacher.

**f-DISTILL** (Wen et al., ACL 2023) — arXiv:2307.15190. seqKD를 f-divergence로; **대칭 손실(JS/TVD)이 비대칭보다 우수** = Vision-OPD의 JSD 선택 근거(즉 JSD는 신규 아님). off-policy.

**Born-Again Networks** (Furlanello et al., ICML 2018) — arXiv:1805.04770 / **BYOT** (Zhang et al., ICCV 2019) — arXiv:1905.08094. self-distillation 원조(동일 구조/내부층). off-policy, privileged-view 없음.

**EMA / mean-teacher 계보**: **Mean Teacher**(Tarvainen & Valpola, NeurIPS 2017, 1703.01780, EMA teacher 원조), **MoCo**(1911.05722), **BYOL**(2006.07733), **DINO**(2104.14294, "self-distillation with no labels" + multi-crop + EMA θ_t←λθ_t+(1−λ)θ_s, centering/sharpening anti-collapse). **DINO가 구조적으로 가장 닮음**(crop 비대칭+EMA+출력발산) — 단 **방향 반대**(DINO: student=local crop, teacher=global; Vision-OPD: teacher=privileged crop, student=full), SSL이지 생성형 next-token 아님.

**LUPI / privileged-info distillation**: Vapnik & Izmailov(JMLR 2015, SVM+), Lopez-Paz et al.(ICLR 2016, "generalized distillation"=Hinton+LUPI 통합), **Learning by Cheating**(Chen et al., CoRL 2019, 1912.12294, privileged teacher→vision student), Lee et al. locomotion(Science Robotics 2020, 2010.11251), asymmetric actor-critic(RSS 2018). → "teacher가 student에 없는 정보를 본다"의 이론적 뿌리.

**텍스트 OPSD 동족(2026, 가장 가까운 메커니즘 쌍)**:
- **OPSD / Self-Distilled Reasoner** (2601.18734): "a single LLM acts as both teacher and student with different contexts ... minimizes the per-token divergence ... over the student's own rollouts." = Vision-OPD의 텍스트 쌍둥이. privileged=정답/CoT 텍스트, EMA 없음.
- **π-Distill** (2602.04942): PI-conditioned teacher + 무조건 student 공동학습 / reverse-KL OPSD 변형. **명시적으로 "does not address multi-modal privileged information (e.g., vision-based PI or cropped image inputs)"** ← Vision-OPD가 정확히 이 미래과제를 채움.
- **RLSD** (2604.03128): 순수 PI-teacher OPSD가 "**severe information leakage and unstable long-term training**" 유발 → RLVR+self-distill 결합. ← Vision-OPD의 EMA가 막으려는 실패모드 문서화.
- **ATESD** (2605.11458): "the teacher always sees the full reference reasoning" → teacher-too-strong → adaptive exposure로 완화. EMA 아닌 대안.
- **GATES** (2602.20574): tutor(privileged document) consensus gating.

**멀티모달 OPD 동족**: VOLD(2510.23497, 외부 텍스트 LLM teacher→VLM, reverse KL), X-OPD(2603.24596, 음성), Video-OPD(2602.02994), Uni-OPD(2605.03677, 다중 외부 teacher), Masters(2512.22238, 대형 teacher 마스킹 + off-policy JSD reward).

### 3.2 미세·고해상 인식 문제공간 + 벤치마크 (C2)

**병목 진단**: **Eyes Wide Shut/MMVP**(Tong et al., CVPR 2024, 2401.06209): "CLIP vision encoders overly focus on high-level semantic understanding, overlooking intricate details"; CLIP↔MLLM 오류 상관 r>0.7. **Do You See Me**(2506.02022): MLLM은 query-relevant 영역에 "around 10%" attention만; 객체가 ~14×14 patch 이하로 작아지면 정확도 붕괴. **HueManity**(2506.03194): fine-tuned ResNet-50(96.5%)이 frontier MLLM(3-33%) 압도.

**고해상 아키텍처**: Monkey(2311.06607), LLaVA-UHD(2403.11703), LLaVA-NeXT/AnyRes, InternVL 1.5(2404.16821, dynamic tiling+pixel-shuffle), Mini-Gemini(2403.18814, dual encoder), S²(2403.13043, multi-scale), DocOwl 1.5(2403.12895), Qwen2-VL(2409.12191, NaViT dynamic res + M-RoPE), NativeRes-LLaVA(2506.12776).

**벤치마크**: **V*Bench/SEAL**(2312.14135, SEAL 75.4% vs GPT-4V 55%), **HR-Bench 4K/8K**(2408.15556), **MME-RealWorld**(2408.13257, "none of them reach 60%"), **MMStar**(2403.20330, vision-indispensable+leakage 지표), **CV-Bench**(2406.16860), **POPE**(2305.10355, 환각), **VER-Bench**(2508.04852, clue ~0.25% area), **ZoomBench**(2602.11858, dual-view "zooming gap" 정량), **RC-Bench**(2506.12776), **FREAK**(2603.19765, 작은/저해상 요소 환각), **FINER**(2603.17662, fine-grained negation).

### 3.3 Thinking-with-images / zoom / agentic (C3)

**foundations**: V*/SEAL(2312.14135, "lack of a visual search mechanism ... hinders ability to focus"). **supervised region-then-answer**: Visual CoT(2403.16999, 438K box-CoT), Chain-of-Spot(2403.12966). **RL zoom/tool**: DeepEyes(2505.14362), Pixel Reasoner(2505.15966), Thyme(2508.11630, code ops), OpenThinkIMG(2505.08617), Chain-of-Focus(2505.15436), Visual-RFT(2503.01785), GRIT(2505.15879), ViGoRL(2505.23678), MGPO(2507.05920, binary reward만으로 grounding emergent), DeepEyesV2(2511.05271), SenseNova-MARS(2512.24330), Pixelis(2603.25091). **training-free zoom**: ZoomEye(2411.16044, tree search), DC²(2408.15556). **survey**: "Thinking with Images"(2506.23918, "external tool → programmatic → intrinsic imagination" 3단계).

**단일패스 내재화 동족(핵심 경쟁군)**:
- **ZwZ / Zooming without Zooming** (2602.11858, ICML 2026) — **최근접 경쟁자**. "transforms zooming from an inference-time tool into a training-time primitive." teacher가 micro-crop에서 VQA 생성 → full image에 bbox overlay(privileged) → RLVR(DAPO) 증류. **off-policy 데이터 증류 + 외부 teacher + RLVR/box**. Vision-OPD 차별점: on-policy self-distill, 라벨/검증기/외부teacher 없음. ZwZ 명시 한계: "may struggle with multiple dispersed regions", info-gain action(web search)은 내재화 불가.
- **DeepSketcher**(2509.25866, latent visual thought SFT), **LVR**(2509.24251, latent visual reasoning+GRPO), **Monet**(2511.21395, latent visual thought dual-supervision SFT), **SD-RPN**(2509.16944, self-distilled RoI).

### 3.4 RL·self-improvement + faithfulness/reliability (C4)

**RLVR perception**: Visual-RFT(2503.01785), Perception-R1(2506.07218), VLM-R1(2504.07615), Ground-R1(2505.20272) — 모두 verifiable reward, abstain/해상도 분석 없음.
**self-reward/label-free**: RLAIF-V(2405.17220, "a 12B model can learn from the feedback of itself"), **Vision-SR1**(2508.19652, perception 충분성 self-check: "given only the question and the visual reasoning, the VLM can give the correct answer"; 저자 자인: 언어 shortcut 보상 가능성 배제 못함), V-Zero(2601.10094).
**"언제 zoom" 게이트**: **AdaFocus**(2603.00171, confidence gate, training-free), **AwaRes/Look Where It Matters**(2603.16932, 학습된 answer-or-crop 정책; "where to look matters as much as whether to look"; 라벨은 low-vs-high-res 답 비교/oracle grounding), **AdaTooler-V**(2512.16918, "blind tool-use" 비판 + Tool Benefit Score), **Q-Zoom**(2604.06912, binary gating net; 자인 "does not address potential hallucinations from ... false-negative decisions"), AdaptVision(2512.03794), CropVLM(2511.19820, 항상 1 crop, where만 결정), Reinforced Hesitation(2511.11500, ternary abstain reward, **LLM-only**).

**faithfulness/reliability(핵심 증거)**:
- **Linking Perception, Confidence and Accuracy** (2603.12149): "**confidence remains surprisingly stable despite severe perception degradation**" / "Does the model know when they do not know?" ← 환각이 자신만만한 이유.
- **Tone Matters / Ghost-100** (2601.06460): by-construction 정보부재 이미지 → "the queried information is absent by construction, [so] any specific answer ... constitutes a hallucination"; 모델이 "honest refusal → fully structured but fabricated card number"로 슬라이드.
- **MLLMs Know Where to Look** (2502.17422, ICLR 2025): "they consistently know where to look, even when they provide the wrong answer" → 병목은 **읽기(reading)**지 위치찾기 아님. (단, region 내 인식이 진짜 한계라고도 시사 — recognition-vs-focus 긴장.)
- **Seeing but Not Believing** (2510.17771): "deep layers often lock onto the correct evidence even when the final answer is wrong" (내부 perception ≠ 출력).
- **Perceptual Limitation** (2402.07384): object size/quality가 독립적으로 정확도 저하(해상도 바닥).
- **MM-AQA** (2604.14799): "models abstain when ... evidence is absent, but attempt reconciliation with degraded ... evidence"(=구조적 명시성만 추적, 진짜 정보가용성 아님); abstention-aware **학습은 미완**.
- **CompoDistill** (2510.12184): "visual attention misalignment ... low performance of existing KD"; 답 증류 ≠ perception 증류.
- **MED / What Does Vision Tool-Use RL Really Learn?** (2602.01334): "**Over 70% of learning progress stems from intrinsic capability, independent of tool access**"; tool 기여율 0.22–0.30 → zoom 도구의 측정 이득이 작고, 가장 어려운(가장 작은 디테일) 케이스가 가장 덜 교정됨.

### 3.5 신규성 검증 (C5) — irreducible PI gap
정책 증류 문헌(검색 확인): "There is an irreducible mutual information gap that the student can never eliminate ... when the teacher has access to privileged information that the student does not." / "Under conditions with privileged information, the divergence drops briefly but then plateaus ... suggesting the existence of an irreducible gap." (state aliasing, 2505.09546 등). **VLM 미세인식 self-distillation에는 미적용** → 메인 아이디어의 이론적 핵심을 그대로 빌려올 수 있고, 적용은 신규.

---

## 4. "이미 점유된 아이디어" 마스터 리스트 (표절 가드레일)

| # | 점유된 아이디어 | 대표 논문 |
|---|---|---|
| 1 | on-policy KD(자기생성 rollout에 teacher feedback, sampling 미역전파) | GKD 2306.13649 |
| 2 | reverse-KL / 대칭 JSD KD 선택(대칭>비대칭) | MiniLLM 2306.08543, f-DISTILL 2307.15190 |
| 3 | self-distillation(동일 모델/구조 teacher=student) | Born-Again 1805.04770, BYOT |
| 4 | EMA teacher anti-collapse(centering/sharpening 포함) | Mean Teacher, MoCo, BYOL, DINO 2104.14294 |
| 5 | privileged-info → 제한-입력 student 증류 후 student 단독 배포 | LUPI(JMLR2015), Learning-by-Cheating 1912.12294 |
| 6 | **같은 모델 privileged-teacher on-policy self-distillation(텍스트)** | OPSD 2601.18734, π-Distill 2602.04942, GATES 2602.20574 |
| 7 | **crop-teacher→full-image-student, zoom을 단일패스로 내재화(VLM)** | **Vision-OPD 2605.18740, ZwZ 2602.11858**, SD-RPN 2509.16944 |
| 8 | 잠재 시각추론으로 도구 없이 "thinking with images" | LVR 2509.24251, Monet 2511.21395, DeepSketcher 2509.25866 |
| 9 | region-predict→crop→re-answer (supervised/RL) | Visual CoT 2403.16999, Chain-of-Spot, MGPO 2507.05920 |
| 10 | RL로 zoom/crop/code 도구 학습(agentic, multi-pass) | DeepEyes, Pixel Reasoner, Thyme, DeepEyesV2 |
| 11 | training-free zoom search | ZoomEye 2411.16044, DC² 2408.15556 |
| 12 | "언제/어디 zoom" 학습/휴리스틱 게이트(재인코딩) | AdaFocus 2603.00171, AwaRes 2603.16932, Q-Zoom 2604.06912 |
| 13 | self-reward/label-free VLM 개선 + perception 충분성 self-check | RLAIF-V 2405.17220, Vision-SR1 2508.19652 |
| 14 | ternary abstain reward(정답/기권/오답) | Reinforced Hesitation 2511.11500 (**LLM-only**) |
| 15 | irreducible PI gap / KL plateau 이론 | 정책증류 문헌(2505.09546 등) (**VLM 미적용**) |
| 16 | 미세/저해상 환각 벤치마크 | FREAK 2603.19765, FINER 2603.17662, Ghost-100 2601.06460 |

---

## 5. 신규 아이디어 + 실험 방향

### 5.1 메인 — RA-OPD: Resolution-Aware On-Policy self-Distillation + 3지 게이트

#### 한 줄 요약
> Vision-OPD/ZwZ류 내재화는 **"focus만 하면 된다"**를 보편 가정한다. 우리는 이를 두 체제로 쪼갠다: evidence가 full-image 토큰에 **존재(focus-recoverable)**하면 내재화는 정당하지만, evidence가 **인코더 해상도 미만(resolution-bound)**이면 crop-teacher가 본 픽셀을 student가 물리적으로 복원 못 하므로 **내재화는 student에게 "자신있는 환각"을 가르친다.** RA-OPD는 ① 이 환각을 실증하고, ② **self-distillation 잔차 KL의 plateau(=irreducible privileged gap)를 라벨-free "비내재화성" 신호**로 검출해, ③ 단일 패스 안에서 **Answer / 진짜 Zoom 1회 / Abstain** 3지를 학습한다.

#### 왜 사용자에게 맞나
이것은 사용자의 텍스트 hybrid-reasoning 철학(쉬우면 빠르게, 어려우면 깊게, 불가능하면 멈춤)의 **시각판**이다. focus-recoverable→System-1 단일패스, resolution-bound-but-resolvable→System-2 진짜 zoom 1회, irresolvable→abstain. label-free라 closed-source distill 0% 정책과 충돌 없음. 가장 핫한 방법(Vision-OPD/ZwZ)의 검증되지 않은 신뢰성 구멍을 정면으로 메운다.

#### 핵심 가설
- **H1 (실증)**: Vision-OPD/ZwZ류 내재화 모델은 resolution-bound subset에서 (a) in-distribution 정확도는 오르나(=암기), (b) holdout 일반화는 실패하고, (c) **confidence가 정확도와 decoupled되어 자신있게 환각**한다. 즉 보고된 "single-glance 정확도"가 환각을 가린다.
- **H2 (신호)**: on-policy self-distillation 학습 중, focus-recoverable 예제는 teacher-student KL이 0으로 수렴하지만 resolution-bound 예제는 KL이 **높은 값에서 plateau(irreducible gap)**한다. 이 잔차 KL은 GT 라벨 없이 "이 디테일은 내재화 불가"를 가리킨다.
- **H3 (방법)**: 잔차 KL 신호로 게이트를 학습하면, 단일-패스-우선 + 필요시 1회 zoom + 불가시 abstain의 3지 정책이 (정확도, FLOPs, **환각율/calibration**) 다목적에서 균일 내재화(Vision-OPD)와 always-zoom(AwaRes/ZoomEye) 모두를 Pareto-지배한다.

#### 기존 최근접 이웃과의 차별화 (표절 방어)
| 최근접 이웃 | 그들이 한 것 | 우리가 다른 점 |
|---|---|---|
| **Vision-OPD / ZwZ** | zoom을 **균일 내재화**, "focus not recognition" 보편 가정, 단일패스 | recoverable vs resolution-bound 구분; 균일 내재화가 후자에서 **자신있는 환각**을 유발함을 실증; **선택적 내재화 + abstain/zoom** 추가 |
| **AdaFocus/AwaRes/Q-Zoom** | "언제/어디 zoom" 게이트, 실제 재인코딩, miscalibrated softmax 또는 oracle/저-고해상 비교 라벨, zoom-or-answer 2지 | **라벨-free 잔차-KL 신호**(oracle·GT 불요), **abstain 3지**, 내재화 valid 여부 자체를 판정 |
| **Vision-SR1** | 생성된 perception-text만으로 답 가능한지 self-check | 우리는 **잔차 KL(전송 불가 privileged 시각정보)** 신호로 해상도 바닥을 직접 겨냥; 언어 shortcut 보상 위험 회피 |
| **MM-AQA / Reinforced Hesitation** | abstention(전자: 학습 미완, 후자: LLM-only) | abstention을 **시각 해상도 바닥 + 증류 gap**에 연결, VLM에서 학습 |
| **정책증류 irreducible-gap 이론** | full-state→partial-obs RL에서 KL plateau=irreducible gap 증명 | 이론을 **VLM 미세인식 self-distillation**에 이식 + plateau를 **행동가능한 per-instance 게이트**로 사용(최초) |

> 핵심 신규성 = **"self-distillation 잔차 KL = 전송 불가 privileged 시각정보의 라벨-free 측정치"라는 신호 + 그에 기반한 선택적 내재화/abstain.** 그리고 **"내재화-zoom이 resolution-bound 디테일을 자신있게 환각한다"는 최초 실증.**

#### 방법 (구체)
1. **백본**: Qwen3.5-VL-4B(주). Vision-OPD를 충실 재현(crop-teacher, on-policy JSD, EMA α=0.05)해 기준선 확보.
2. **데이터 2분할**: 합성 시 각 예제를 **(a) focus-recoverable** (crop 영역이 full-image 인코더에서 ≥1 패치로 표현됨) vs **(b) resolution-bound** (crop 대상이 full image에서 인코더 패치(예: 14×14px·28×28px) 미만)으로 라벨. Perceptual-Limitation(2402.07384)·Do-You-See-Me의 패치-임계와 Ghost-100식 by-construction 부재 케이스 활용.
3. **잔차 KL 신호(H2)**: 학습 동안 예제별 teacher-student JSD 궤적 기록. 마지막 K step 평균 잔차를 per-instance "내재화 난이도" score로. (예측엔 score를 hidden state로 회귀하는 경량 head — Vision-SR1식 GT 불요.)
4. **3지 게이트(H3)**: student rollout에 special action 토큰 — `<answer>` / `<zoom:box>` / `<abstain>`. 학습: 잔차-KL이 낮으면 answer, 중간이면 zoom(진짜 crop 1회 재인코딩 후 답), 높으면(plateau, by-construction 부재) abstain. ternary 보상(Reinforced Hesitation식: +정답, 0 abstain when irresolvable, −자신있는 오답)을 GRPO로. **단 보상의 abstain 타당성은 GT가 아니라 잔차-KL+저-고해상 일치불가로 라벨-free 도출**(차별점).
5. **calibration(H1 대응)**: student confidence가 resolution-bound에서 낮아지도록 calibration 항(clean vs sub-patch-blur 입력 간 confidence shift 보상, Linking-Perception식) 추가.

#### 실험 설계
- **평가**: V*Bench, HR-Bench 4K/8K, MME-RealWorld, ZoomBench(+dual-view gap), **FREAK/FINER/Ghost-100(환각)**, holdout(MMVP/CV-Bench/MMStar/POPE). **recoverable vs resolution-bound 분해 필수.**
- **베이스라인**: vanilla, Vision-OPD(재현), ZwZ, AwaRes(always answer-or-crop), ZoomEye(always tree-zoom), AdaFocus(confidence gate), abstain 없는 RA-OPD ablation.
- **핵심 지표**:
  - **Hallucination-under-internalization**: resolution-bound에서 자신있는 오답율(=정답이 by-construction 불가능한데 구체적 답을 낸 비율). Vision-OPD가 vanilla보다 **악화**되는지(H1 핵심).
  - **Calibration**: resolution-bound subset의 ECE / confidence-accuracy 상관. (Linking-Perception처럼 "perception 저하 시 confidence가 떨어지는가".)
  - **3지 정책 품질**: answer/zoom/abstain 정확 분배율(잔차-KL ground-truth 대비), abstain한 것이 실제 irresolvable인 precision/recall.
  - **정확도-FLOPs Pareto**: 동일 평균 패스/zoom 비용에서 정확도; always-zoom 대비 zoom 호출 절감.
- **Ablation**: (a) 잔차-KL 신호 vs softmax-confidence 게이트, (b) abstain 유무, (c) 잔차-KL plateau가 정말 resolution-bound와 상관하는지(인과: 입력 해상도 조작), (d) EMA α가 잔차-KL plateau 높이에 주는 영향(정보 누출↔gap).

#### 예상 기여
- **C1(놀라움·실증)**: "내재화-zoom은 focus-recoverable에선 진짜 perception을 배우지만 resolution-bound에선 자신있는 환각을 배운다" — Vision-OPD/ZwZ류 전체의 보편 가정에 대한 정량 반례 + 환각이 single-glance 정확도에 가려짐을 폭로.
- **C2(신호)**: self-distillation **잔차 KL = 전송 불가 privileged 시각정보의 라벨-free 측정치** (정책증류 이론을 VLM에 이식, per-instance 게이트화 최초).
- **C3(방법)**: 단일패스 안 Answer/Zoom/Abstain 3지 — 사용자 hybrid-reasoning의 시각판, calibration 보장.

#### 리스크와 대응
- **R1: resolution-bound에서도 내재화가 환각 안 하고 그냥 틀리기만 하면?** → 그래도 "confidence-accuracy decoupling"(H1)이 측정되면 기여 성립. 사전 파일럿(Ghost-100 + Vision-OPD 재현)으로 환각율 먼저 측정해 go/no-go.
- **R2: 잔차-KL이 resolution-bound와 약상관이면** → 게이트 신호를 잔차-KL + 저-고해상 답 불일치 + perception-sufficiency(Vision-SR1) 앙상블로 보강.
- **R3: ZwZ/AwaRes와 "또 다른 zoom 게이트"로 보일 위험** → C1(환각 실증)과 "라벨-free 잔차-KL 신호 + abstain"을 전면에. ablation (a),(b)로 차별성 시연.

---

### 5.2 대안 A — 분산 증거(multi/dispersed-evidence) on-policy self-distillation
ZwZ가 명시적으로 "may struggle with multiple dispersed regions". **다중 crop teacher**(여러 evidence 영역을 동시에 본 teacher) → 단일 패스 student self-distillation. 차별화: 기존 내재화는 단일 dominant RoI 가정. 신규: 여러 흩어진 단서를 한 패스에 통합 내재화 + (메인과 결합 시) 영역별 잔차-KL로 어느 단서가 내재화 가능한지 판정. 리스크: 다중 crop teacher 합성 비용, 영역 attribution.

### 5.3 대안 B — privileged view를 crop 너머로 ("OPD for view-X")
on-policy self-distillation의 privileged teacher 입력을 crop이 아닌 **depth map / segmentation / OCR overlay / 고해상 재인코딩**으로. 차별화: π-Distill이 "vision-based PI"를 미래과제로 남겼고, Vision-OPD는 crop만 함. 신규: "어떤 privileged view가 어떤 task에서 내재화 가능한가"의 체계적 지도 + view별 잔차-KL 비교. 리스크: 확장성 있으나 깊이는 메인보다 얕음.

### 5.4 대안 C — 내재화 perception의 calibration/faithfulness
메인의 H1만 떼어내, "내재화 모델이 환각할 때 정확히 unconfident하도록" 만드는 calibration 기법 단독 논문. 차별화: 기존 calibration(Variational VQA, MMBoundary)은 일반 VQA; 우리는 내재화-zoom 특화 + Ghost-100식 by-construction 부재로 faithfulness 직접 측정. 메인의 부분집합이라 메인 우선.

---

## 6. 착수 전 표절 회피 체크리스트
1. **arXiv 재검색**(착수 직전): "internalized zoom hallucination", "resolution-bound vs recoverable detail VLM", "residual KL privileged information gate VLM", "abstain when cannot see fine detail", "selective internalization distillation". 월 단위 신규 주의.
2. **Vision-OPD GitHub(github.com/VisionOPD/Vision-OPD) + ZwZ 정독**: 둘이 calibration/abstain/해상도 분석을 추가하지 않았는지 v-latest 확인.
3. **AwaRes/Q-Zoom/AdaFocus의 게이트 신호**가 우리 "잔차-KL 라벨-free 신호"와 다른지(그들은 softmax/oracle/저-고해상 비교) 명시.
4. **정책증류 irreducible-gap 이론(2505.09546 등)** 인용 — 우리는 이론을 VLM 미세인식에 이식하고 게이트화한 것임을 positioning.
5. **Reinforced Hesitation(LLM-only) / MM-AQA(학습 미완)** 인용 — 우리는 abstention을 VLM 해상도 바닥 + 증류 gap에 연결.
6. closed-source distill 0% 정책 준수: teacher=자체 모델의 crop-conditioned 정책(외부 teacher·GT 라벨 불요).

---

## 7. 권장 진행 순서
1. **파일럿(2~3주)**: Vision-OPD 재현(4B) + Ghost-100/FREAK로 **resolution-bound 환각율 측정**. vanilla 대비 악화되면(H1 확인) 메인 확정.
2. 확인 시: 학습 중 per-instance 잔차-KL 로깅 → recoverable/resolution-bound와의 상관(H2) 검증 → 3지 게이트 + calibration 구현(H3) → Pareto·환각율·calibration 평가.
3. H1 약하면: 대안 A(분산 증거) 또는 B(privileged view 확장)로 피벗.

---

## 부록: 참고문헌 (arXiv)
- 앵커: Vision-OPD 2605.18740
- 증류: GKD 2306.13649 · MiniLLM 2306.08543 · DistiLLM 2402.03898 / 2503.07067 · f-DISTILL 2307.15190 · Born-Again 1805.04770 · BYOT 1905.08094 · Mean-Teacher 1703.01780 · MoCo 1911.05722 · BYOL 2006.07733 · DINO 2104.14294 · LUPI(JMLR2015) · Generalized-Distillation 1511.03643 · Learning-by-Cheating 1912.12294 · locomotion 2010.11251
- OPSD 동족: OPSD 2601.18734 · π-Distill 2602.04942 · RLSD 2604.03128 · ATESD 2605.11458 · GATES 2602.20574 · VOLD 2510.23497 · X-OPD 2603.24596 · Uni-OPD 2605.03677 · Masters 2512.22238
- 문제·벤치: MMVP/Eyes-Wide-Shut 2401.06209 · Do-You-See-Me 2506.02022 · HueManity 2506.03194 · Monkey 2311.06607 · LLaVA-UHD 2403.11703 · InternVL1.5 2404.16821 · Mini-Gemini 2403.18814 · S² 2403.13043 · DocOwl1.5 2403.12895 · Qwen2-VL 2409.12191 · NativeRes 2506.12776 · V* 2312.14135 · HR-Bench 2408.15556 · MME-RealWorld 2408.13257 · MMStar 2403.20330 · CV-Bench 2406.16860 · POPE 2305.10355 · VER-Bench 2508.04852 · ZoomBench/ZwZ 2602.11858 · RC-Bench 2506.12776
- thinking-with-images: Visual-CoT 2403.16999 · Chain-of-Spot 2403.12966 · DeepEyes 2505.14362 · Pixel-Reasoner 2505.15966 · Thyme 2508.11630 · OpenThinkIMG 2505.08617 · Chain-of-Focus 2505.15436 · Visual-RFT 2503.01785 · GRIT 2505.15879 · ViGoRL 2505.23678 · MGPO 2507.05920 · DeepEyesV2 2511.05271 · SenseNova-MARS 2512.24330 · Pixelis 2603.25091 · ZoomEye 2411.16044 · DC² 2408.15556 · survey 2506.23918
- 내재화/잠재: DeepSketcher 2509.25866 · LVR 2509.24251 · Monet 2511.21395 · SD-RPN 2509.16944
- RL/self-improve: Perception-R1 2506.07218 · VLM-R1 2504.07615 · Ground-R1 2505.20272 · RLAIF-V 2405.17220 · Vision-SR1 2508.19652 · V-Zero 2601.10094
- 게이트/abstain: AdaFocus 2603.00171 · AwaRes 2603.16932 · AdaTooler-V 2512.16918 · Q-Zoom 2604.06912 · AdaptVision 2512.03794 · CropVLM 2511.19820 · Reinforced-Hesitation 2511.11500
- faithfulness: Linking-Perception-Confidence 2603.12149 · Ghost-100/Tone-Matters 2601.06460 · MLLMs-Know-Where-to-Look 2502.17422 · Seeing-but-Not-Believing 2510.17771 · Perceptual-Limitation 2402.07384 · MM-AQA 2604.14799 · CompoDistill 2510.12184 · MED 2602.01334 · FREAK 2603.19765 · FINER 2603.17662
- 이론: irreducible PI gap(정책증류) 2505.09546

> 검증 주의: 2026 preprint(2601.\*~2605.\*)는 제목·초록·arXiv ID를 검색·일부 정독으로 확인했으나 전문 미정독 다수. 인용 전 원문 재확인. ViGoR(2402.06118, reward modeling) vs ViGoRL(2505.23678, grounded RL zoom)는 다른 논문 — 혼동 주의. AwaRes = "Look Where It Matters"(2603.16932) 동일 논문.
