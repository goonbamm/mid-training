# 4B LLM Mid-Training 데이터셋 큐레이션 최종 보고서

**작성일**: 2026-05-26
**대상**: 4B hybrid reasoning 모델 mid-training
**토크나이저**: Gemma4 기준 (영문 ≈ 3.8 char/token, 코드 ≈ 3.0, JSON ≈ 3.3, 한국어 ≈ 1.8)
**목표**: 일반 추론 30B 토큰 + 지시 수행 추론 30B 토큰 + **GenUI 특화 ~10B 토큰**(2026-05-28 신규 도메인, +증액 → 총 ~70B)

---

## 요약 (TL;DR)

- **검토 규모**: HuggingFace 데이터셋 1,000+ 인지, 920+ 직접 카드 검토(WebFetch 라이센스/생성모델 확인 포함).
- **확정 정책 (Scenario A, 보수적)**:
  - 라이센스 100% Apache-2.0 / MIT / CC-BY-4.0 / ODC-BY / BSD / CC0
  - Closed-source distill (Claude/GPT-5.x/Gemini/Grok) **0%**
  - Llama 출력 합성 **거부** (Llama 3.1+ 라이센스 변경에도 모델명 prefix 제약 회피)
  - GPT-4o/GPT-4 출력은 비중 ≤ 5%, 보조 용도만
- **Teacher 모델 분포**: DeepSeek-R1 ~35% / Qwen3·QwQ ~25% / gpt-oss-120B ~20% / GLM-5.1 ~8% / Kimi K2.5 ~8% / Human ~4%
- **GenUI 특화 도메인 (2026-05-28 신규, +증액 ~10B → 총 ~70B)**: 코드 생성 + 구조화 렌더링, 텍스트 전용, thinking 포함. 공개 클린 데이터 ~700-800M(WebSight·WebGen·Nemotron-structured·Tessa-T1·Tesslate)이 한계 → **자체 합성(컴포넌트 트리·reasoning UI)이 필수**. 상세 §4.5
- **Hybrid reasoning toggle 학습**의 3대 자산:
  1. [Jackrong/GLM-5.1-Reasoning-1M-Cleaned](https://huggingface.co/datasets/Jackrong/GLM-5.1-Reasoning-1M-Cleaned) — GLM-5.1의 명시적 `<think>` wrapping
  2. [nvidia/Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) AoPS subset — gpt-oss-120B의 6가지 reasoning mode (high/med/low × TIR on/off)
  3. [Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b](https://huggingface.co/datasets/Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b) — 4B 학생 모델 정렬용 (DASD-4B-Thinking 재현)

---

## 1. 큐레이션 정책

### 1.1 라이센스
| 분류 | 라이센스 |
|---|---|
| **허용** | Apache-2.0, MIT, BSD, CC0, CC-BY-4.0, ODC-BY, Gemma Terms, NVIDIA Open License (commercial 가능 명시) |
| **거부** | CC-BY-NC*, CC-BY-SA(weight 전염), CC-BY-ND, GPL-2.0, research-only, Llama Community License 출력 합성 |
| **주의** | GPT-4/Claude/Gemini 출력 (ToS 회색지대) — 비중 ≤ 5% |

### 1.2 모델 출처별 안전성
| Teacher 모델 | 사용 가능성 | 비고 |
|---|---|---|
| DeepSeek-R1 / R1-0528 | 안전 | MIT |
| Qwen2.5 / Qwen3 / QwQ | 안전 | Apache-2.0 |
| **GLM-5.1 / 4.5 / 4.6** | 안전 | MIT, commercial+secondary development 명시 허용 |
| **gpt-oss-120B / 20B** | 안전 | Apache-2.0, proprietary GPT와 달리 출력 제한 없음 |
| **Kimi K2 / K2.5** | 안전 | Modified MIT (1억 MAU 또는 $20M MRR 초과 시 UI attribution) |
| Mistral / Mixtral | 안전 | Apache-2.0 |
| Gemma 2/3 | 안전 | Gemma Terms (prohibited uses 외 OK) |
| GPT-4 / GPT-4o / GPT-5.x | 주의 | OpenAI ToS 회색지대, 비중 제한 |
| Claude (3, 4) | 주의 | Anthropic Usage Policy 회색지대 |
| **Llama 2/3/3.1/3.3** | **거부** | Llama 3.1+ 라이센스 변경에도 모델명 "Llama" prefix 의무 |

### 1.3 모델 다양성 정책
**단일 teacher ≤ 50% 캡** (R1-monoculture는 reflection 패턴 과적합 및 brittle behavior 보고).

확정 분포 (60B 전체):
| Teacher | 비중 | 토큰 |
|---|---|---|
| DeepSeek-R1 / R1-0528 / R1-Distill (Qwen base만) | 35% | ~21B |
| Qwen3-235B-Thinking / QwQ-32B / Qwen2.5 | 25% | ~15B |
| gpt-oss-120B / 20B | 20% | ~12B |
| GLM-5.1 | 8% | ~5B |
| Kimi K2.5 | 8% | ~5B |
| Mixtral / DeepSeek-V3 / Human | 4% | ~2B |

### 1.4 Decontamination (필수)
다음 평가셋과 8-gram + fuzzy 0.8 매칭으로 학습 데이터에서 차단:
- **수학**: AIME 2024/2025, MATH-500, OlympiadBench-math, GSM8K test, AMC23
- **코드**: HumanEval+, MBPP+, BigCodeBench, LiveCodeBench v5, CodeContests test, APPS test
- **과학**: GPQA-Diamond, MMLU-Pro, TheoremQA, LAB-Bench, MedQA-USMLE
- **일반**: BBH, ZebraLogic, ARC-Challenge, MuSR
- **Agent**: BFCL v3 test, ToolHop, GAIA, tau2-bench
- **IF**: IFEval test, IFBench test, FollowBench, ComplexBench test
- **GenUI/Web** (2026-05-28 추가): WebDev Arena ([lmarena-ai/webdev-arena-preference-10k](https://huggingface.co/datasets/lmarena-ai/webdev-arena-preference-10k) — **최우선**, 실제 프롬프트+모델 코드), Design2Code ([SALT-NLP/Design2Code](https://huggingface.co/datasets/SALT-NLP/Design2Code) + Design2Code-HARD + -hf), Sketch2Code ([SALT-NLP/Sketch2Code](https://huggingface.co/datasets/SALT-NLP/Sketch2Code)), Web2Code eval split([MBZUAI/Web2Code](https://huggingface.co/datasets/MBZUAI/Web2Code) 5,990 + 흡수된 Pix2Code), WebCode2M([xcodemind/webcode2m](https://huggingface.co/datasets/xcodemind/webcode2m), fuzzy), WebGen-Bench(eval 101 instruction), DesignBench(WebPAI/DesignBench), WebBench·Interaction2Code(GitHub 코드 타깃), 2025-26 신규(WebCoderBench/IWR-Bench/WebGen-V — HF 공개 시)
  - 이미지 기반 벤치마크라도 **HTML/코드 타깃이 학습에 누출**될 수 있으므로 코드 타깃 기준 8-gram + fuzzy 매칭 차단.

도구: [datatrove](https://github.com/huggingface/datatrove) MinHash + open-r1 decontamination 스크립트.

---

## 2. 토큰 분배 (60B + GenUI ~10B = ~70B)

> **2026-05-28**: 기존 60B(일반추론 30B + IF 30B)에 **GenUI 특화 ~10B 신규 도메인 +증액**. 상세는 §4.5.

### GenUI 특화 ~10B (신규)
| Sub-domain | 토큰 | 비중 |
|---|---|---|
| 프론트엔드 코드 생성 (instruction→code) | 4B | 40% |
| UI reasoning (think→code, hybrid toggle) | 3B | 30% |
| 구조화 UI / 컴포넌트 트리 (자체 합성 중심) | 2B | 20% |
| SVG / 차트 / data-viz spec | 1B | 10% |

### 일반 추론 30B
| Sub-domain | 토큰 | 비중 | 근거 |
|---|---|---|---|
| 수학 | 8B | 26.7% | AIME/MATH/GSM/Olympiad 평가 신호 강함, 정답 명확 |
| 코드 | 7B | 23.3% | reasoning + programmatic 사고 양성 |
| 과학/STEM | 5B | 16.7% | GPQA/MMLU-Pro 매칭, 의학·물리·화학·생물 |
| 논리/일반 long-CoT | 7B | 23.3% | 인문/사회/법/경제/철학/상식 + hybrid toggle |
| Agent / Function-calling (BFCL) | 3B | 10.0% | BFCL 향상 목표 — 도메인 풀 작아 비중 ↓ |

### 지시 수행 추론 30B
| 분류 | 토큰 | 비중 | 근거 |
|---|---|---|---|
| Verifiable (코드/regex 검증) | 18B | 60% | IFBench/IFEval 직접 효과, RLVR-ready |
| Semi-verifiable (LLM-judge) | 7.5B | 25% | 자연어 제약 일반화 |
| Unverifiable (자연 대화 IF) | 4.5B | 15% | rule-checking model 화 방지 |

---

## 3. 일반 추론 30B 후보 데이터셋

### 3.1 수학 (8B)
| 데이터셋 | 라이센스 | 추정 토큰 | 생성 모델 | 우선순위 |
|---|---|---|---|---|
| [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | CC-BY-4.0 | 3.5B (cap) | R1 + QwQ | S |
| [nvidia/Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) **AoPS subset만** | CC-BY-4.0 | 3.0B | gpt-oss-120B 6모드 | **S+** |
| [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) default split | Apache-2.0 | 0.4B | DeepSeek-R1 | S |
| [open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) math 850K | Apache-2.0 | 0.5B | QwQ-32B | S |
| [nvidia/AceReason-1.1-SFT](https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT) math 2.67M | CC-BY-4.0 | 0.5B (subsample) | DeepSeek-R1 | S |
| [AI-MO/NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) | Apache-2.0 | 0.2B | Human + 검증 | S |
| [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) math subset | Apache-2.0 | 0.2B | DeepSeek-R1 | A |
| [nvidia/OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) | NVIDIA Open License | 0.1B | Mixtral-8x7B | A |
| [hkust-nlp/dart-math-uniform](https://huggingface.co/datasets/hkust-nlp/dart-math-uniform) + [zwhe99/DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) | MIT | 0.3B | DeepSeekMath-RL + R1 | A |
| LIMO + s1K-1.1 + Bespoke-Stratos-17k | Apache/MIT | 0.05B | R1 + Human | A |
| [HAERAE-HUB/HRM8K](https://huggingface.co/datasets/HAERAE-HUB/HRM8K) | MIT | 0.003B | 한국어 원본 | A (KR) |
| [TIGER-Lab/WebInstructSub](https://huggingface.co/datasets/TIGER-Lab/WebInstructSub) (Math StackExchange만) | Apache-2.0 | 0.2B | Human (CC 크롤) | A |
| [KbsdJames/Omni-MATH](https://huggingface.co/datasets/KbsdJames/Omni-MATH) + OlympiadBench math | Apache-2.0 | 0.04B | Human + R1 | A |

**도메인 합계**: ~9.0B → dedup(NuminaMath 시드 공유) 후 8B

### 3.2 코드 (7B)
| 데이터셋 | 라이센스 | 추정 토큰 | 생성 모델 | 우선순위 |
|---|---|---|---|---|
| [nvidia/OpenCodeReasoning-2](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2) | CC-BY-4.0 | 1.5B (60% sample) | R1 + QwQ critique | S |
| [ianncity/KIMI-K2.5-1000000x](https://huggingface.co/datasets/ianncity/KIMI-K2.5-1000000x) (code 50% 부분) | Apache-2.0 | 2.5B | Kimi K2.5 high | **S+** |
| [nvidia/Nemotron-Competitive-Programming-v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1) | CC-BY-4.0 | 0.7B (1/3 sample) | R1-0528 | S |
| [nvidia/OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) v1 | CC-BY-4.0 | 0.3B (dedup 후) | DeepSeek-R1 | S |
| [open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) code 250K | Apache-2.0 | 0.4B | QwQ-32B | S |
| [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | CC-BY-4.0 | 0.7B | Mixtral + Qwen2.5 | S |
| [nvidia/Llama-Nemotron-PT](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) code (Llama-3.3 행 제외) | CC-BY-4.0 | 0.8B | R1 + Qwen2.5 | S (필터 필수) |
| [open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots) + Mixture-of-Thoughts code | Apache-2.0 | 0.3B | DeepSeek-R1 | S |
| [zake7749/Qwen3-Coder-Next-Open-Code-SFT](https://huggingface.co/datasets/zake7749/Qwen3-Coder-Next-Open-Code-SFT) (100% test pass) | CC-BY-4.0 | 0.15B | Qwen3-Coder-Next | A |
| [Jackrong/qwen3-coder-480b-distill-mini](https://huggingface.co/datasets/Jackrong/qwen3-coder-480b-distill-mini) | Apache-2.0 | 0.05B | Qwen3-Coder-480B | A |
| BAAI/TACO + codeparrot/apps + deepmind/code_contests (Human) | Apache/MIT/CC-BY | 0.2B | Human | A |
| [hkust-nlp/CodeIO-PyEdu-Reasoning](https://huggingface.co/datasets/hkust-nlp/CodeIO-PyEdu-Reasoning) | ODC-BY | 0.2B | Mixed | A |
| OpenCoder-LLM (opc-sft-stage1/2 multilingual) | MIT | 0.15B | Magicoder/Evol | A |
| [internlm/SWE-Fixer-Train-110K](https://huggingface.co/datasets/internlm/SWE-Fixer-Train-110K) + SWE-rebench | Apache-2.0 / CC-BY-4.0 | 0.2B | Composite | A |
| [bigcode/commitpackft](https://huggingface.co/datasets/bigcode/commitpackft) (permissive 필터) | per-sample | 0.1B | Human commits | A |

**도메인 합계**: ~8.3B → question_id dedup 후 7B

### 3.3 과학/STEM (5B)
| 데이터셋 | 라이센스 | 추정 토큰 | 생성 모델 | 우선순위 |
|---|---|---|---|---|
| [nvidia/Nemotron-Pretraining-Specialized-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1) STEM-SFT | CC-BY-4.0 | 1.2B | R1-0528 + Qwen2.5 | S |
| Nemotron-Pretraining-Specialized-v1 RQA + InfiniByte | CC-BY-4.0 | 1.0B | Qwen3-235B-Thinking + QwQ + gpt-oss-120B | S |
| [nvidia/OpenScience](https://huggingface.co/datasets/nvidia/OpenScience) | CC-BY-4.0 | 0.8B (subsample) | Qwen2.5/3 생성, R1 응답 | S |
| [nvidia/Nemotron-Science-v1](https://huggingface.co/datasets/nvidia/Nemotron-Science-v1) | CC-BY-4.0 | 0.4B | gpt-oss-120B | **S** |
| [Jackrong/Natural-Reasoning-gpt-oss-120B-S1](https://huggingface.co/datasets/Jackrong/Natural-Reasoning-gpt-oss-120B-S1) | Apache-2.0 | 0.4B | gpt-oss-120B-high | A |
| [OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B](https://huggingface.co/datasets/OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B) | Apache-2.0 | 0.4B | gpt-oss-120B | A |
| [open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) science 100K | Apache-2.0 | 0.3B | QwQ-32B | S |
| [cognitivecomputations/dolphin-r1](https://huggingface.co/datasets/cognitivecomputations/dolphin-r1) reasoning splits | Apache-2.0 | 0.2B | R1 + Gemini Flash | A |
| FreedomIntelligence/Medical-R1-Distill + UCSC-VLAA/MedReason | Apache-2.0 | 0.08B | DeepSeek-R1 + KG | A |
| PubMedQA + medmcqa + PhysReason + OlympiadBench(physics) | MIT/Apache/CC-BY | 0.2B | Human | A |

**도메인 합계**: ~5.0B

### 3.4 논리 / 일반 long-CoT (7B)
| 데이터셋 | 라이센스 | 추정 토큰 | 생성 모델 | 우선순위 |
|---|---|---|---|---|
| [Jackrong/GLM-5.1-Reasoning-1M-Cleaned](https://huggingface.co/datasets/Jackrong/GLM-5.1-Reasoning-1M-Cleaned) | Apache-2.0 | 2.5B | GLM-5.1 (`<think>` toggle) | **S+** |
| [nvidia/Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) chat + STEM 필터 | CC-BY-4.0 | 1.0B | R1-0528 + Qwen3 | S |
| [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) toggle | CC-BY-4.0 | 1.0B | R1-0528 + Qwen3 multilingual | **S+** (hybrid toggle) |
| [nvidia/Llama-Nemotron-PT](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) science+chat (Llama 행 제외) | CC-BY-4.0 | 0.5B | R1 + Qwen2.5 | S (필터 필수) |
| [open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) + OpenThoughts-114k puzzle | Apache-2.0 | 0.7B | QwQ + R1 | S |
| [nvidia/OpenScienceReasoning-2](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2) 인문/법/경제 subset | CC-BY-4.0 | 0.3B | R1-0528 | S |
| [GeneralReasoning/GeneralThought-430K](https://huggingface.co/datasets/GeneralReasoning/GeneralThought-430K) (natolambert filtered) | MIT | 0.3B | R1/QwQ/OpenThoughts mix | A |
| [nvidia/Nemotron-CrossThink](https://huggingface.co/datasets/nvidia/Nemotron-CrossThink) | CC-BY-4.0 | 0.2B | Qwen2.5 | A |
| [KOREAson/YiSang-HighQuality](https://huggingface.co/datasets/KOREAson/YiSang-HighQuality) | Apache-2.0 | 0.4B | Qwen3-32B | S (KR) |
| commonsense_qa + social_i_qa + self-CoT augment | MIT/CC-BY | 0.1B | Human + self-CoT | B |

**도메인 합계**: ~7.0B

### 3.5 Agent / Function-calling (3B, BFCL 향상)
| 데이터셋 | 라이센스 | 추정 토큰 | 생성 모델 | BFCL 매칭 | 우선순위 |
|---|---|---|---|---|---|
| [Agent-Ark/Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) | Apache-2.0 | 1.5B (cap) | Qwen3-32B + Kimi-K2 + GPT-OSS-120B | 전 카테고리 + MCP | **S+** |
| [nvidia/Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) tool-calling split | CC-BY-4.0 | 0.25B | R1-0528 + Qwen3-235B | single + multi-turn + multi-step | S+ |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Apache-2.0 | 0.15B | Multi-agent self-evolution | multi-turn, 26K API | S+ |
| [interstellarninja/hermes_reasoning_tool_use](https://huggingface.co/datasets/interstellarninja/hermes_reasoning_tool_use) | Apache-2.0 | 0.2B | Hermes + Atropos RL | reasoning + tool (hybrid) | **S+** |
| [Nanbeige/ToolMind](https://huggingface.co/datasets/Nanbeige/ToolMind) | Apache-2.0 | 0.15B | Multi-agent (Qwen3) | multi-turn, agentic | S |
| [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | CC-BY-4.0 | 0.05B | DeepSeek-V2 + Mixtral | simple/multiple/parallel | S |
| [MadeAgents/xlam-irrelevance-7.5k](https://huggingface.co/datasets/MadeAgents/xlam-irrelevance-7.5k) + [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) | CC-BY-4.0 | 0.05B | Composite | **irrelevance 전담** | S |
| [argilla/Synth-APIGen-v0.1](https://huggingface.co/datasets/argilla/Synth-APIGen-v0.1) (Qwen 17.7k만, Llama 제외) | Apache-2.0 | 0.03B | Qwen2.5-72B | simple/multiple/irrelevance | S |
| [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Apache-2.0 | 0.05B | Hermes + Glaive subset | single + multi-turn structured | S |
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) + [internlm/Agent-FLAN](https://huggingface.co/datasets/internlm/Agent-FLAN) | Apache-2.0 | 0.2B | Glaive + ChatGPT + GPT-4 | simple + ReAct trace | A |
| [ibm-research/nestful](https://huggingface.co/datasets/ibm-research/nestful) | Apache-2.0 | 0.002B | MathQA + StarCoder2 | nested function calls | A |
| **자체 합성** Java/JS BFCL (Qwen2.5-Coder-32B로 5-10k씩) | 자체 | 0.05B | Qwen2.5-Coder-32B | Java/JS BFCL AST | **S** (BFCL 공백 메우기) |

**도메인 합계**: ~2.7B → Toucan oversample 또는 자체 합성 확대로 3B

---

## 4. 지시 수행 추론 30B 후보 데이터셋

### Tier S+ (Verifiable, IFBench 직격) — oversample 20-40×
| 데이터셋 | 라이센스 | 원본 토큰 | 생성 모델 | 분류 |
|---|---|---|---|---|
| [nvidia/Nemotron-RL-instruction_following](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following) | ODC-BY | 15M | WildChat + Open-Instruct 30+ constraint | Verifiable |
| [THU-KEG/VerInstruct](https://huggingface.co/datasets/THU-KEG/VerInstruct) | Apache-2.0 | 12M | Qwen2.5-72B + QwQ | Hard + soft (LLM-judge) |
| [allenai/tulu-3-sft-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) + [sherryy expanded](https://huggingface.co/datasets/sherryy/tulu-3-sft-personas-instruction-following-expanded) | ODC-BY | 30M | GPT-4o + persona | Verifiable (IFEval) |
| [allenai/RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval) + [Dolci-RL-Zero-IF](https://huggingface.co/datasets/allenai/Dolci-RL-Zero-IF-7B) | ODC-BY | 7M | Tulu2 + IFEval 24 constraints | Verifiable (RLVR) |
| [allenai/Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) Precise IF (136K subset만) | ODC-BY | 50M | OLMo3 + Nemotron persona | Verifiable |
| [HuggingFaceTB/smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) multi-turn-reasoning-if-think | Apache-2.0 | 12M | Qwen3-235B + Qwen3-32B think | Multi-turn IF + reasoning |
| [Magpie-Align/Magpie-Qwen2.5-Pro-300K-Filtered](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2.5-Pro-300K-Filtered) | Apache-2.0 | 600M | Qwen2.5-72B | Semi-verifiable |

### Tier S (Verifiable / Semi)
| 데이터셋 | 라이센스 | 원본 토큰 | 생성 모델 | 분류 |
|---|---|---|---|---|
| [nvidia/Nemotron-RL-instruction_following-structured_outputs](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs) | CC-BY-4.0 | 10M | NVIDIA schema-driven | Verifiable JSON |
| [allenai/tulu-3-IF-augmented-on-policy-70b](https://huggingface.co/datasets/allenai/tulu-3-IF-augmented-on-policy-70b) (chosen만) | ODC-BY | 25M | Tulu3-70B + Gemma-2 + InternLM2.5 | Verifiable |
| [nvidia/Llama-Nemotron-PT](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) IF (R1+Qwen 필터) | CC-BY-4.0 | 20M | R1 + Qwen2.5 | Verifiable + reasoning |
| [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) 영어 chat IF | CC-BY-4.0 | 5B | R1-0528 + Qwen3 | Semi/Unverifiable multilingual |
| [Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1) | Apache-2.0 | 2B | Qwen2.5-72B | Semi-verifiable |
| [Magpie-Align/Magpie-Reasoning-V2-250K-CoT-QwQ](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-QwQ) | Apache-2.0 | 500M | Qwen2-72B + QwQ-Preview | Semi (reasoning + IF) |
| Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b IF subset | CC-BY-4.0 | 1B | gpt-oss-120B-high | Verifiable + reasoning |

### Tier A (보조)
| 데이터셋 | 라이센스 | 원본 토큰 | 생성 모델 | 분류 |
|---|---|---|---|---|
| [ConiferLM/Conifer](https://huggingface.co/datasets/ConiferLM/Conifer) | Apache-2.0 | 8M | GPT-4 from ShareGPT (주의) | Multi-turn complex |
| [nvidia/Daring-Anteater](https://huggingface.co/datasets/nvidia/Daring-Anteater) IF subset | CC-BY-4.0 | 2M | Mixtral + NVIDIA | Verifiable precise + JSON |
| [facebook/Multi-IF](https://huggingface.co/datasets/facebook/Multi-IF) | CC-BY-4.0 | 3M | LLM + Human | Multi-turn + 8 언어 |
| [THU-KEG/IF-Verifier-Data](https://huggingface.co/datasets/THU-KEG/IF-Verifier-Data) | Apache-2.0 | 5M | Qwen2.5-72B | Verifiable RM 학습용 |
| [TIGER-Lab/AceCode-87K](https://huggingface.co/datasets/TIGER-Lab/AceCode-87K) | MIT | 200M | GPT-4o-mini | Verifiable code IF |
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) + [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | Apache-2.0 | 100M | Human (65 lang) | Unverifiable |
| HuggingFaceTB/smoltalk2 Smol-Magpie-multilingual (Llama 제외) | Apache-2.0 부분 | 1B | Qwen3-32B | Semi-verifiable |

**원본 합계**: ~12.7B → S+ 20-40× / S 10-20× / A 1-5× oversampling으로 **30B 달성**

---

## 4.5 GenUI 특화 도메인 (신규 도메인 · 토큰 증액, 2026-05-28 추가)

> **결정 (2026-05-28 사용자)**: GenUI 특화 모델이므로 GenUI를 **신규 top-level 도메인**으로 신설하고 총 예산을 **+증액**. 범위는 **(1) 프론트엔드 코드 생성 + (2) 구조화 UI 렌더링 둘 다**, **텍스트 전용**(vision/screenshot 입력 제외), **thinking 트레이스 포함**(hybrid toggle 정렬 — 쉬운 UI는 짧게/없이, 복잡한 UI는 길게 추론). 라이센스·생성모델·decontamination 정책은 §1 그대로 적용.

### 4.5.0 핵심 발견 (먼저 읽을 것)

1. **진짜 "generative-UI" 공개 데이터는 사실상 없음.** 런타임 렌더링용 JSON 컴포넌트 트리 / AI-SDK·RSC generative-UI / tool→component 형태의 공개 클린 데이터셋은 검색 결과 부재(대부분 vision UI-detection으로 귀결). → **이 하위 카테고리는 자체 합성이 유일한 현실적 경로.**
2. **추론+UI 데이터의 최대 자산(Tesslate 계열)은 라이센스는 깨끗(apache-2.0)하나 teacher(생성) 모델이 카드에 비공개.** 정책상 "생성모델 verbatim 기록 + Llama/closed-source 거부"를 엄격 적용하려면 **teacher 확인 전까지 조건부**. (단 `Tesslate/Tessa-T1-Dataset`만 "based on Qwen2.5-Coder" 명시 — 유일한 무결점 S.)
3. **즉시 채택 가능한 무결점 클린 자산**: `HuggingFaceM4/WebSight`(Mistral-7B + DeepSeek-Coder-33B), `luzimu/WebGen-Bench_train_data`(DeepSeek-V3), `nvidia/Nemotron-Instruction-Following-Chat-v1` structured_outputs(gpt-oss-120B + Qwen3), `Tesslate/Tessa-T1-Dataset`(Qwen2.5-Coder).
4. **클린 instruction/reasoning 공개 데이터는 현실적으로 ~700-800M에 그침.** 나머지는 web 코드 pretrain corpus(permissive 필터, the-stack-v2/stack-edu/webcode2m) + 자체 합성 + 고품질셋 oversampling으로 채움.

### 4.5.1 GenUI 토큰 예산 (제안 ~10B, 총 60B → ~70B)

| Sub-domain | 토큰 | 구성 | 근거 |
|---|---|---|---|
| **A. 프론트엔드 코드 생성** (instruction→code) | 4B | WebSight text-side(cap 0.5B) + WebGen-Instruct + Tessa-T1/Tesslate direct + kalinkov + permissive web corpus(the-stack-v2/stack-edu HTML·CSS·JS·Vue·Svelte cap ~3B) | React/HTML/CSS/Tailwind 직접 생성 능력 |
| **B. UI reasoning** (think→code, hybrid toggle) | 3B | Tesslate reasoning 클러스터 ~400M(teacher 검증 후) oversample 4-5× + 자체 합성 reasoning UI(gpt-oss-120B/Qwen3/GLM-4.6) ~1B | "쉬운 UI 짧게 / 복잡한 UI 길게" toggle 학습의 핵심 |
| **C. 구조화 UI / 컴포넌트 트리** (generative-UI) | 2B | Nemotron structured_outputs(~3M) + xLAM tool→widget 보조 + **자체 합성 JSON 컴포넌트 트리 ~1.8B**(공개 데이터 부재) | AI-SDK/RSC 스타일 구조화 렌더링 |
| **D. SVG / 차트 / data-viz spec** | 1B | svg-stack(라이센스 확정 시) + VisCode Vega-Lite·Mermaid·SVG 서브셋(코드만) + Text2Vis | SVG·차트·시각화 spec 생성 |

> **현실 체크**: A~D 중 공개 클린 instruction/reasoning은 ~700-800M. 나머지 ~9B는 (i) permissive web 코드 corpus(instruction 없는 pretrain성), (ii) 자체 합성(B의 reasoning, C의 컴포넌트 트리, 한국어), (iii) 고품질셋 oversampling으로 충당. GenUI 비중을 더 키우려면 자체 합성 규모를 늘려야 함.

### 4.5.2 A. 프론트엔드 코드 생성 후보

| 데이터셋 | 라이센스 | 추정 토큰 | 생성 모델 | reasoning | 우선순위 |
|---|---|---|---|---|---|
| [Tesslate/Tessa-T1-Dataset](https://huggingface.co/datasets/Tesslate/Tessa-T1-Dataset) | apache-2.0 | ~7.7M | **Qwen2.5-Coder (명시)** | Yes | **S** (React+Tailwind 직격, 유일 무결점) |
| [HuggingFaceM4/WebSight](https://huggingface.co/datasets/HuggingFaceM4/WebSight) text-side | CC-BY-4.0 | 960M 전량 → **cap 0.5B** | Mistral-7B(아이디어) + DeepSeek-Coder-33B(HTML) | No | **S** (대량 + 클린, 비전 불필요) |
| [luzimu/WebGen-Bench_train_data](https://huggingface.co/datasets/luzimu/WebGen-Bench_train_data) (WebGen-Instruct) | MIT | ~5-15M | **DeepSeek-V3** | No(trajectory) | **S** (단 동명 eval과 분리·decontam 필수) |
| [Tesslate/UIGEN-T2](https://huggingface.co/datasets/Tesslate/UIGEN-T2) | apache-2.0 | ~110-165M | **비공개(teacher)** | Yes | **S조건부** (teacher 확인 시 S) |
| [Tesslate/Next.js-Dataset](https://huggingface.co/datasets/Tesslate/Next.js-Dataset) | apache-2.0 | ~116-130M | **비공개** | Yes | A조건부 (Next.js/SSR 추론) |
| [kalinkov/tailwindcss_components](https://huggingface.co/datasets/kalinkov/tailwindcss_components) | apache-2.0 | ~2M | LLM 생성(추정) | No | A (Tailwind 컴포넌트, 소량) |
| [cfahlgren1/react-code-instructions](https://huggingface.co/datasets/cfahlgren1/react-code-instructions) | MIT | ~150M(전체)/~75M(클린) | **혼합: Llama-3.1-70B/405B + DeepSeek-V3 + Qwen2.5-Coder-32B** | No | A (**`model` 컬럼으로 Llama 행 제외 필수**, DeepSeek/Qwen만) |
| [HuggingFaceTB/stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu) JS/TS | the-stack-v2 준용(permissive) | JS 11B + TS 3B(StarCoder tok) → web 선별·cap | 인간 코드(Llama는 필터링용일 뿐, 출력 아님 → brand 무관) | No | A (고품질 web 코드 backbone) |
| [bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2) HTML/CSS/JS/Vue/Svelte | other(permissive only) | 수십~수백 B → **강한 cap** | 인간 크롤 코드 | No | A (permissive 서브셋만, attribution, SWH 동의) |
| [codeparrot/github-code](https://huggingface.co/datasets/codeparrot/github-code) (web, permissive 필터) | other(`licenses=` mit/apache/bsd만) | cap 필수 | 인간 GitHub 코드 | No | B (copyleft 제외, the-stack-v2와 중복 가능) |

### 4.5.3 B. UI reasoning (think→code, hybrid toggle 핵심)

| 데이터셋 | 라이센스 | 추정 토큰 | 생성 모델 | reasoning 구조 | 우선순위 |
|---|---|---|---|---|---|
| [Tesslate/UIGEN-T3-Dataset-Extended-Reasoning](https://huggingface.co/datasets/Tesslate/UIGEN-T3-Dataset-Extended-Reasoning) | apache-2.0 | ~106M | **비공개(teacher)** | **2단(Pre+Post) reasoning** — long-think 학습 최적 | **S조건부** |
| [Tesslate/UIGEN-T2](https://huggingface.co/datasets/Tesslate/UIGEN-T2) | apache-2.0 | ~110-165M | 비공개 | prompt/reasoning/response | **S조건부** |
| [Tesslate/UIGEN-T1.5-Dataset](https://huggingface.co/datasets/Tesslate/UIGEN-T1.5-Dataset) = [smirki/UIGEN-T1.1-TAILWIND](https://huggingface.co/datasets/smirki/UIGEN-T1.1-TAILWIND) | apache-2.0 | ~7-8M(중복) | 비공개 | reasoning | B조건부 |
| [smirki/UI_Reasoning_Dataset](https://huggingface.co/datasets/smirki/UI_Reasoning_Dataset) | MIT | ~3.6M | 비공개 | reasoning | B조건부 |
| **자체 합성** reasoning UI (gpt-oss-120B / Qwen3 / GLM-4.6) | 자체 | ~1B | gpt-oss-120B/Qwen3/GLM-4.6 | `<think>` 레이아웃/접근성/상태관리 추론 → 코드 | **S** (teacher 검증 회피 + toggle 라벨 직접 제어) |

> **Tesslate 클러스터 합계 ~300-400M(raw).** teacher 검증 시 oversample하여 B의 핵심. 미검증 시 보수적으로 제외하고 자체 합성으로 대체.

### 4.5.4 C. 구조화 UI / 컴포넌트 트리 (generative-UI)

| 데이터셋 | 라이센스 | 추정 토큰 | 생성 모델 | reasoning | 우선순위 |
|---|---|---|---|---|---|
| [nvidia/Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1) structured_outputs 서브셋 | CC-BY-4.0 | ~3M | **gpt-oss-120B + Qwen3-235B-Thinking/Instruct** | on/off 라벨 | **S** (유일 클린 JSON-schema 제약 출력) |
| [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | CC-BY-4.0 | ~9M | DeepSeek-V2 + Mixtral-8x22B | No | B (tool→widget 구조 보조) |
| [osunlp/Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) text-only | CC-BY-4.0 | 소형(action repr) | 인간 어노테이션 | No | B (DOM 이해 보조, UI emit 아님) |
| **자체 합성** JSON 컴포넌트 트리 / RSC spec | 자체 | ~1.8B | gpt-oss-120B/Qwen3/GLM-4.6 | `<think>` → 컴포넌트 트리 | **S** (공개 데이터 부재 → 필수 합성) |

### 4.5.5 D. SVG / 차트 / data-viz spec

| 데이터셋 | 라이센스 | 추정 토큰 | 생성 모델 | 우선순위 |
|---|---|---|---|---|
| [starvector/svg-stack](https://huggingface.co/datasets/starvector/svg-stack) | **카드 공란**(GitHub Apache-2.0) — 확정 전 보류 | 2.28M행 text-only SVG | 인간/크롤 | A조건부 (라이센스 확정 시) |
| [TIGER-Lab/VisCode-Multi-679K](https://huggingface.co/datasets/TIGER-Lab/VisCode-Multi-679K) Vega-Lite·Mermaid·SVG 서브셋 | apache-2.0 | 코드만 추출 시 수십M | 코드=오픈소스 스크랩 / **지시문=GPT-4.1(플래그)** | A (코드 출력 위주, 지시문 대량 사용 회피) |
| [mizanurr/Text2Vis](https://huggingface.co/datasets/mizanurr/Text2Vis) | MIT | ~1M | 비공개(arXiv 2507.19969) | A (matplotlib, **난이도 라벨**이 hybrid 라우팅에 유용) |

### 4.5.6 거부 목록 (GenUI)

| 데이터셋 | 라이센스 | 거부 사유 |
|---|---|---|
| [MBZUAI/Web2Code](https://huggingface.co/datasets/MBZUAI/Web2Code) | research-only / CC-BY-4.0 NC | **비상업 + GPT-3.5/GPT-4 생성** (이중 거부). 단 eval split은 decontam 대상 |
| [smirki/React-Reasoning](https://huggingface.co/datasets/smirki/React-Reasoning) | — | **claude-3-5-sonnet + o1 생성** (closed-source distill) |
| [iamdyeus/ui-instruct-4k](https://huggingface.co/datasets/iamdyeus/ui-instruct-4k) | apache-2.0 | **Claude Opus 4.6 + GPT-5 distill** + 라이브러리 역공학 리스크 |
| [xingxm/SVGX-SFT-1M](https://huggingface.co/datasets/xingxm/SVGX-SFT-1M) | CC-BY-NC-4.0 | **비상업** |
| [dataunitylab/json-schema](https://huggingface.co/datasets/dataunitylab/json-schema) | unknown | 라이센스 불명(허용목록 외) |
| [paraloq/json_data_extraction](https://huggingface.co/datasets/paraloq/json_data_extraction) | apache-2.0 | **Gemini-Pro 생성** + 추출 태스크(UI 아님) |
| `mrtoy/mobile-ui-design`, `YashJain/UI-Elements-Detection`, `zhourax977/VEGA`, `Tesslate/UIGEN-T3-14B*`(모델) | 다양 | vision/UI-detection 또는 모델 research-only |

### 4.5.7 자체 합성 계획 (GenUI 필수 투자)

> **상세 계획서**: `genui_synthesis_plan.md` (2026-05-28) — 출력형식 표준화(3 캐논 + `genui/v1` 트리), 중간 GPU teacher 선정, 시드→생성→실행기반 검증 캐스케이드(8단계)→reasoning trace→dedup/decontam→self-training, 토큰 예산 매핑, Phase 롤아웃.

공개 클린 데이터로는 ~700-800M에 그치므로, GenUI 도메인의 다수는 자체 합성으로 채움:
- **컴포넌트 트리 / RSC spec (C, ~1.8B)**: gpt-oss-120B/Qwen3/GLM-4.6로 "자연어 요구 → `<think>` 설계 → JSON 컴포넌트 트리/props" 생성. 진짜 generative-UI 공개 데이터 부재를 메움.
- **reasoning UI (B, ~1B)**: 동일 teacher로 레이아웃/접근성(a11y)/반응형/상태관리 추론 → 코드. **명시적 `reasoning: on/off` 라벨**로 hybrid toggle 직접 제어 (Tesslate teacher 미검증 리스크 회피).
- **한국어 UI 보너스**: 기성 한국어 GenUI 데이터는 부재. Qwen/DeepSeek/GLM/gpt-oss로 한국어 프롬프트→UI 코드 소량 합성.
- **Java/Kotlin·SwiftUI 등 비-web UI**(선택): 필요 시 동일 파이프라인.

### 4.5.8 GenUI teacher 다양성

WebSight(Mistral+DeepSeek-Coder-33B) / WebGen(DeepSeek-V3) / Nemotron-structured(gpt-oss-120B+Qwen3) / Tessa-T1(Qwen2.5-Coder) / 자체합성(gpt-oss-120B·Qwen3·GLM-4.6)으로 분산 → §1.3 단일 teacher ≤50% 캡 준수. Tesslate(비공개)는 검증 후에만 카운트.

---

## 5. 거대 데이터셋 Split 라우팅

같은 데이터셋이 여러 도메인에 걸치는 경우 split-level로 분배:

| 데이터셋 (총 토큰) | 도메인별 분배 |
|---|---|
| [nvidia/Nemotron-Post-Training-Dataset-v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) (~25B) | 수학 0.5 + 코드 0.5 + 과학 0.5 + 논리 1.0 + Agent 0.25 + IF 일부 |
| [nvidia/Llama-Nemotron-PT](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) (~16B, Llama 행 제외) | 수학 0.5 + 코드 0.8 + 과학 0.3 + 논리 0.5 + IF 0.4 |
| [open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) (~6B) | 수학 0.5 + 코드 0.4 + 과학 0.3 + 논리 0.3(puzzle) |
| [nvidia/AceReason-1.1-SFT](https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT) (~30B) | 수학 0.5 + 코드 0.5 |
| [allenai/Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) (~5B+) | IF Precise subset 2.0B만 |
| [HuggingFaceTB/smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) (~12B) | IF 1.4B (multi-turn-if-think + Smol-Magpie 영어 일부, Llama-3.1-405B split 제외) |

**처리 원칙**: 한 번 다운로드 후 도메인 메타데이터로 분류, cross-domain 중복 카운팅 방지.

---

## 6. 중복 처리 파이프라인

### 6.1 시드 단위 중복 (NuminaMath, CodeContests 등)
같은 문제에 다른 모델 traces이 분산되어 있음. **30-50% problem-level 중복** 예상.

### 6.2 Dedup 파이프라인 (datatrove 기반)
1. 각 데이터셋 다운로드 → question_id / source URL 메타데이터 추출
2. MinHash 13-gram, num_perm=128, Jaccard 0.7 threshold
3. Cross-dataset dedup (한 question이 여러 데이터셋에 나타나면 highest-quality 1개만)
4. Eval set 8-gram 매칭 차단 (§1.4 평가셋 목록)
5. 도메인별 토큰 카운트 재산출 → 최종 mix ratio

---

## 7. Hybrid Reasoning 학습 정렬 (사용자 목표)

다음 자산이 **thinking/non-thinking toggle 학습의 핵심**:

| 자산 | 역할 |
|---|---|
| [Jackrong/GLM-5.1-Reasoning-1M-Cleaned](https://huggingface.co/datasets/Jackrong/GLM-5.1-Reasoning-1M-Cleaned) | GLM-5.1의 명시적 `<think>` wrapping — thinking mode 학습용 (논리 7B 중 2.5B) |
| [nvidia/Nemotron-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-Math-v2) AoPS subset | gpt-oss-120B의 6가지 reasoning mode (high/med/low × TIR on/off) — thinking 길이 세밀 제어 |
| [Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b](https://huggingface.co/datasets/Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b) | DASD-4B-Thinking 재현용 — 4B 학생 capacity gap 해결 |
| [nvidia/Llama-Nemotron-PT](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset) + [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) | "reasoning on/off" toggle 라벨 — 동일 prompt에 두 모드 응답 페어 |
| [interstellarninja/hermes_reasoning_tool_use](https://huggingface.co/datasets/interstellarninja/hermes_reasoning_tool_use) | reasoning + tool 결합 — Agent와 reasoning을 잇는 다리 |
| [HuggingFaceTB/smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) multi-turn-reasoning-if-think | reasoning + IF 결합 |
| OpenCodeInstruct (non-thinking) + OpenCodeReasoning-2 (long-CoT) | 코드 thinking/non-thinking balance |
| **GenUI**: Tesslate UIGEN-T3-Extended(2단 reasoning, teacher 검증 후) + 자체 합성 reasoning UI(`reasoning: on/off` 라벨) | UI 생성 toggle — 단순 컴포넌트는 직접 생성, 복잡 레이아웃/상태관리는 long-think (§4.5.3) |

**Hybrid 학습 권장 mix**: thinking 70% / non-thinking 30%. 명시적 `reasoning: on/off` 라벨링하여 학습. GenUI도 동일 — 단순 UI(버튼/카드)는 non-thinking, 복잡 UI(대시보드/폼 검증/반응형)는 thinking.

---

## 8. 실행 액션 아이템

### 8.1 다운로드 우선순위 (각 도메인 S 등급만)
- **수학**: OpenMathReasoning, Nemotron-Math-v2 AoPS, OpenR1-Math-220k, OpenThoughts3-1.2M math, AceReason-1.1-SFT math, NuminaMath-1.5, HRM8K
- **코드**: OpenCodeReasoning-2, Kimi K2.5 code 부분, Nemotron-CP-v1, OpenCodeReasoning v1, OpenCodeInstruct, Llama-Nemotron-PT code(필터)
- **과학**: Nemotron-Pretraining-Specialized-v1, OpenScience, Nemotron-Science-v1, Natural-Reasoning gpt-oss-120B, Medical gpt-oss-120B, OpenThoughts3 science
- **논리**: GLM-5.1-Reasoning-1M-Cleaned, Nemotron-Post-Training-v1/v2 toggle, Llama-Nemotron-PT(필터), OpenThoughts3 puzzle, YiSang-HQ
- **Agent**: Toucan-1.5M, Nemotron tool-calling split, ToolACE, hermes_reasoning_tool_use
- **IF**: Nemotron-RL-IF, VerInstruct, Tulu3-personas-IF, RLVR-IFeval, Dolci-Instruct-SFT Precise, smoltalk2 multi-turn-if-think, Magpie-Qwen2.5-Pro-300K
- **GenUI**: Tessa-T1, WebSight(text-side, cap 0.5B), WebGen-Instruct(eval 분리), Nemotron structured_outputs, VisCode UI 서브셋 — 그리고 자체 합성(C/B)

### 8.2 Llama 출력 필터링 (필수)
- `nvidia/Llama-Nemotron-PT`: `generator == "Llama-3.3-70B-Instruct"` 행 ~420k 제외
- `argilla/Synth-APIGen-v0.1`: Llama-3.1 부분 25.7k 제외, Qwen 17.7k만 사용
- `HuggingFaceTB/smoltalk2`: Smol-Magpie-Ultra (Llama-3.1-405B) split 제외
- `cfahlgren1/react-code-instructions`: `model` 컬럼으로 Llama-3.1-70B/405B 행 제외, DeepSeek-V3 + Qwen2.5-Coder-32B 행만 사용

### 8.8 GenUI 도메인 (신규, §4.5)
- **Tesslate/smirki teacher 검증**: ~300-400M 토큰의 채택 여부를 가르는 단일 최대 변수. GitHub/HF discussion으로 teacher 모델 문의 → Llama/closed-source면 거부, Qwen/DeepSeek/GLM/gpt-oss면 S 승격. 검증 전까지 자체 합성으로 대체.
- **자체 합성(필수)**: 컴포넌트 트리/RSC spec ~1.8B + reasoning UI ~1B + 한국어 UI 소량. gpt-oss-120B/Qwen3/GLM-4.6로 `<think>`→코드/JSON, 명시적 `reasoning: on/off` 라벨.
- **`starvector/svg-stack` 라이센스 확정**: 카드 공란 → 작성자 확인 후 채택(SVG D 도메인).
- **`TIGER-Lab/VisCode-Multi-679K`**: `language` 컬럼으로 Vega-Lite/SVG/Mermaid/HTML만 추출, GPT-4.1 지시문은 대량 사용 회피(코드 출력 위주 또는 Qwen/GLM 재라벨).
- **web pretrain corpus(the-stack-v2/stack-edu/github-code)**: permissive 라이센스만 필터 + SoftwareHeritage 동의 + attribution, 강한 cap(web 총 ~3B).
- **GenUI decontam**: §1.4 GenUI/Web 평가셋(WebDev Arena 최우선) 차단.

### 8.3 Decontamination (§1.4 평가셋 목록 적용)
datatrove + open-r1 decontamination 스크립트로 8-gram + fuzzy 0.8 매칭 차단.

### 8.4 자체 합성
**Java/JavaScript BFCL 카테고리**는 공개 학습 데이터가 거의 없음. Qwen2.5-Coder-32B로 각 5-10k씩 자체 합성 (5-10시간 작업).

### 8.5 단일 chat template 정규화
모든 function calling/IF/reasoning 데이터를 하나의 template으로 통일 (Hermes XML 또는 OpenAI tool_calls JSON 권장).

### 8.6 Reasoning on/off 라벨링
hybrid reasoning 학습용 명시적 토큰 추가:
- `<|thinking_on|>` / `<|thinking_off|>` special tokens
- 또는 `<think>` 태그 통일

### 8.7 최종 토큰 측정
datatrove + Gemma4 토크나이저로 실제 토큰 측정. 본 보고서 추정과 ±15% 이내 일치 예상.

---

## 9. 최종 모델 라이센스 분포

| 라이센스 | 비중 | 비고 |
|---|---|---|
| CC-BY-4.0 (NVIDIA Nemotron 계열) | ~50% | Attribution 의무 |
| Apache-2.0 | ~35% | 가장 자유 |
| MIT | ~8% | |
| ODC-BY (Tulu/Dolci) | ~5% | |
| NVIDIA Open License | ~1% | 상업 가능 |
| 기타 (Gemma Terms, CDLA-Permissive) | ~1% | |

**최종 모델 라이센스 권고**: 상업 사용 가능, **attribution 의무 명시** (NVIDIA Nemotron 데이터 사용 사실 표기).

> **GenUI ~10B 추가 영향 (2026-05-28)**: WebSight·Nemotron-structured·VisCode(CC-BY-4.0) + Tessa/자체합성(Apache-2.0) + WebGen(MIT) 조합으로 위 분포를 크게 바꾸지 않음. 단 the-stack-v2/github-code permissive 코드를 쓰면 **파일별 원 라이센스 attribution**과 SoftwareHeritage 동의 의무가 추가됨.

---

## 부록 A. 거부 데이터셋 (핵심 65개+)

### A.1 CC-BY-NC / NC-SA / NC-ND (비상업)
KodCode 전 시리즈, AM-DeepSeek-R1-Distilled 전 시리즈, facebook/natural_reasoning, MegaScience, TextbookReasoning, SCP-116K, WebInstructFull, AceMath-Instruct-Training-Data, ServiceNow R1-Distill-SFT, Camel-AI 전 시리즈, ChemPile 전 시리즈, GAIR/lima, no_robots, BAAI/Infinity-Instruct, PersonaHub, allenai/sciq, ai2_arc, ScienceQA, mgsm, Salesforce/APIGen-MT-5k, **xingxm/SVGX-SFT-1M(GenUI, CC-BY-NC)**, **MBZUAI/Web2Code(GenUI, NC + GPT 생성)**

### A.2 Llama 출력 합성
argilla/magpie-ultra-v1.0/v0.1, FinePersonas, Llama-3-Magpie-Pro 시리즈, Magpie-Llama-3.1-Pro 시리즈, Magpie-Llama-3.3-Pro, Magpie-Reasoning-V2-Deepseek-R1-Llama-70B, HuggingFaceTB/smoltalk v1, smol-smoltalk, MathGenie/MathCode-Pile, DeepHermes-3 데이터, Reflection-Dataset-v2

### A.3 Eval contamination
gorilla-llm/Berkeley-Function-Calling-Leaderboard (BFCL test), google/IFEval, allenai/IFBench_test, TIGER-Lab/MMLU-Pro, TheoremQA, Idavidrein/gpqa, math-ai/aime24, MATH-500, OlympiadBench(eval), lab-bench, ChemBench, livecodebench, BigCodeBench, UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled, tau2-bench-data

### A.4 Closed-source distill (Scenario A 결정에 따라 거부)
lordx64/reasoning-distill-opus-4-7-max-sft, Kassadin88/Claude-Distills, Crownelius/Opus-4.6-Reasoning-3300x, TeichAI/Claude-Sonnet-4.6-Reasoning-1100x, Nettoov/Gpt-5.4-Xhigh-Reasoning-2000x, TeichAI/gemini-3-pro-preview-high-reasoning-1000x, TeichAI/grok-code-fast-1-1000x, **smirki/React-Reasoning(GenUI, claude-3-5-sonnet+o1)**, **iamdyeus/ui-instruct-4k(GenUI, Claude Opus 4.6+GPT-5)**, **paraloq/json_data_extraction(GenUI, Gemini-Pro)**

### A.5 기타
hkust-nlp/agentboard (GPL-2.0), OpenHermes-2.5 (GPT-4 출력 + 라이센스 혼재), ShareGPT 전 시리즈 (OpenAI ToS)

---

## 부록 B. 대안 시나리오 (사용자 정책 변경 시)

### Scenario B (중도, closed-source 1-3B)
Scenario A + Grok distill 약간:
- `TeichAI/grok-code-fast-1-1000x` 등 Grok distill 1-2B (xAI ToS는 output 소유권을 user에게 부여)

### Scenario C (공격적, closed-source 19B = 32%)
- 기존 큐레이션 22B + Claude Opus 4.7 8B + Kassadin Claude-Distills 3B + Claude Sonnet 4.6 critical thinking 2B + Grok 2B + GPT-5.4 math 1B + Gemini code 1B
- Claude 단일 출처 ≤ 25% 캡 준수
- 법적 회색지대, enforcement 선례 0건이나 매출/규모 확대 시 indirect 위험

전환 시 부록 A.4의 데이터셋을 후보로 재추가.

---

## 부록 C. 라이센스 정책 상세

### C.1 Llama Community License 변경 사항
- **Llama 3.0**: "improve any other LLM" 금지 — 거부
- **Llama 3.1+**: 위 조항 삭제. attribution 의무 ("Built with Llama" 표시, 모델명 "Llama" prefix)
- **본 프로젝트 결정**: 모델명 "Llama" prefix가 Gemma4 베이스와 브랜딩 충돌 — **전체 Llama 출력 거부 유지**

### C.2 Closed-source ToS 해석
- **OpenAI / Anthropic / Google**: API ToS의 "competing model 학습 금지"는 직접 API 호출자(데이터셋 생성자)에게 적용. HF에서 다운로드하는 downstream user는 ToS 비당사자
- **enforcement 선례**: HF 공개 데이터셋에 대한 OpenAI/Anthropic/Google 소송 0건. 다만 매출/규모 확대 시 indirect 위험 상승
- **본 프로젝트 결정**: 회색지대 회피, **closed-source distill 0%**

### C.3 GPT-4o 출력 데이터셋 (예외)
다음은 GPT-4o 출력이지만 비중이 작고 검증 효과 큼:
- allenai/tulu-3-sft-personas-instruction-following (IFEval taxonomy 직접)
- allenai/Dolci-Instruct-SFT (IFBench paper 직계)
- TIGER-Lab/AceCode-87K (verifiable code IF)

→ IF 도메인 18B verifiable 중 ~2B 정도. 전체 60B 대비 3.3%.

---

## 부록 D. 사용자가 검토되지 않았다고 지적한 모델 처리

| 사용자 지적 | 최종 처리 |
|---|---|
| **GLM 5.1** | N1로 채택, 논리 도메인 7B 중 **2.5B (35.7%)** — hybrid reasoning toggle 학습의 핵심 자산 |
| **Claude Opus 4.7** | Scenario A 결정에 따라 **거부**. 향후 정책 변경 시 부록 B Scenario C로 즉시 전환 가능 |
| **gpt-oss-120B/20B** | 채택, 총 ~9B+ (Nemotron-Math-v2, Nemotron-Science-v1, Superior-Reasoning, Natural-Reasoning, Medical) — Apache-2.0 무제한 |
| **Kimi K2/K2.5** | 채택, **5B** (KIMI-K2.5-1000000x) — Modified MIT 사실상 무제한 |

---

## 부록 E. 참고 자료

### E.1 핵심 데이터셋 컬렉션
- [NVIDIA Nemotron Collection](https://huggingface.co/nvidia)
- [Allen AI Tulu / Dolci](https://huggingface.co/allenai)
- [Open-Thoughts](https://huggingface.co/open-thoughts)
- [Open-R1](https://huggingface.co/open-r1)
- [PrimeIntellect SYNTHETIC](https://huggingface.co/PrimeIntellect)
- [Agent-Ark Toucan](https://huggingface.co/Agent-Ark)
- [THU-KEG VerInstruct](https://huggingface.co/THU-KEG)

### E.2 핵심 paper
- [IFBench (2025)](https://arxiv.org/abs/2507.02833) — verifiable IF benchmark
- [Tulu 3 / Olmo 3 reports](https://arxiv.org/abs/2512.13961) — RLVR IF 방법론
- [VerIF paper](https://arxiv.org/abs/2506.09942)
- [Conifer paper](https://arxiv.org/abs/2404.02823)
- [Multi-IF paper](https://arxiv.org/abs/2410.15553)

### E.3 라이센스 문서
- [Anthropic Usage Policy](https://www.anthropic.com/legal/aup)
- [OpenAI Business Terms](https://openai.com/policies/may-2025-business-terms/)
- [Gemini API Terms](https://ai.google.dev/gemini-api/terms)
- [Llama 3.3 License](https://www.llama.com/llama3_3/license/)

### E.4 도구
- [datatrove](https://github.com/huggingface/datatrove) — 토큰 측정 + dedup + decontamination
- [open-r1 decontamination scripts](https://github.com/huggingface/open-r1)
