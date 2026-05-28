# GenUI 합성 데이터 생성 계획서 (4B mid-training용)

**작성일**: 2026-05-28
**대상**: 4B GenUI-특화 hybrid reasoning 모델, mid-training
**연관 문서**: `mid_training_dataset_report_v2.md` §4.5 (GenUI 도메인), `adaptive_reasoning_length_report.md` (hybrid thinking)
**토크나이저**: Gemma4 기준 (코드 ≈ 3.0 char/token, JSON ≈ 3.3, 한국어 ≈ 1.8)

---

## 0. 배경 · 목표 · 제약

### 왜 합성하는가
GenUI 도메인의 공개 클린 instruction/reasoning 데이터는 현실적으로 **~700-800M 토큰**에 그치고, 특히 **진짜 generative-UI(JSON 컴포넌트 트리 / AI-SDK·RSC / tool→component)는 공개 데이터가 사실상 0**이다. §4.5 예산의 다음 부분을 자체 합성으로 채워야 한다:

| §4.5 sub-domain | 합성 필요량 | 비고 |
|---|---|---|
| B. UI reasoning (think→code) | ~1B | Tesslate teacher 미검증 리스크 회피 + toggle 라벨 직접 제어 |
| C. 구조화 UI / 컴포넌트 트리 | ~1.8B | 공개 데이터 부재 → 전량 합성 |
| 한국어 UI 보너스 | ~0.1-0.2B | 기성 데이터 부재 |
| A·D 보강 (선택) | 가변 | 공개셋 부족분 보충 |
| **합성 목표(승인 토큰)** | **~3B+** | accept 후 기준 |

### 핵심 통찰 (설계 철학)
GenUI에서 "정답"은 **렌더링 가능한 아티팩트**다. 일반 텍스트 추론과 달리 **컴파일·렌더·인터랙션 테스트라는 객관적 reward 신호를 거의 무료로** 얻는다. 따라서:

> **instruction을 잘 만드는 것보다 verification 파이프라인이 압도적인 품질 레버다.** (Apple UICoder가 4B급에서 폐쇄 모델에 근접한 결정적 이유.)

또한 trace의 옳고 그름조차 "그 trace가 만든 코드가 렌더/테스트를 통과하는가"로 검증 가능 → **환각 추론을 객관 신호로 거를 수 있다**(텍스트 추론 도메인 대비 큰 이점). 이는 사용자의 hybrid/overthinking 연구 방향과도 직접 맞물린다.

### 확정 제약 (2026-05-28 사용자 결정)
- **출력 형식 = 범용(여러 형식 혼합)** → §1의 3개 캐논 포맷으로 표준화.
- **합성 인프라 = 중간 GPU (~30-70B teacher)** → 대형 teacher(gpt-oss-120B/Qwen3-235B) 대신 **dense ≤32B**로 운영. throughput이 병목 → 싼 정적 필터로 대량 제거 + self-training으로 yield 개선이 필수(§6, §9, §10).
- **검증 = 실행 기반(권장)** → 컴파일+headless 렌더(Playwright)+e2e 테스트가 1차 reward, VLM-judge는 캐스케이드 말단에만.
- **thinking 포함(hybrid toggle)** → 쉬운 UI는 think 생략/짧게, 복잡한 UI는 long think. 난이도 라벨을 합성 단계에서 직접 생성.
- **license**: teacher는 OPEN/clean만 — gpt-oss-20B(Apache-2.0), Qwen3, Qwen2.5-Coder, DeepSeek-R1/V3, QwQ, GLM, Kimi, Mistral. **Llama·closed-source(GPT/Claude/Gemini/Grok) 출력 일절 배제.** 샘플 단위로 teacher 모델·버전 로깅(추후 라이선스 감사 대비).

---

## 1. 출력 형식 표준화 (범용 혼합)

실제 GenUI 시스템 조사 결과, 다음 **3개 캐논 포맷**으로 표준화하면 Vercel AI SDK·Airbnb식 SDUI·JSON Forms·craft.js·Vega-Lite/Mermaid를 모두 한 표현으로 커버한다.

1. **Tool-call envelope (OpenAI 규격)** — "어떤 위젯을, 어떤 props로" 결정 레이어. `{name, arguments(JSON)}`. AI SDK 6 / OpenAI 양쪽 호환. 위젯형 GenUI(날씨/주식/폼)의 기본 골격.
2. **통합 컴포넌트 트리 JSON** — `{type, props, children, actions, a11y}` 5필드. SDUI·craft.js·Plasmic·JSON Forms를 흡수하는 구조화 출력 타깃.
3. **도메인 선언 스펙 passthrough** — 차트=Vega-Lite JSON, 다이어그램=Mermaid 텍스트를 트리의 leaf `props.spec`에 통째 임베드(재발명 금지). NL→spec 페어로도 별도 합성.

> **React/Tailwind 직접 코드**는 4번째 타깃이 아니라 **트리에서 결정론적으로 컴파일되는 "참조 정답"**으로 둔다. 동일 의미를 `(NL 프롬프트) + (트리 JSON) + (코드)` **트리플로 페어링**하면 모델이 코드 모드 ↔ 구조화 모드 간 일관성을 학습한다.

### 1.1 통합 컴포넌트 트리 스키마 (`genui/v1`)

```json
{
  "$schema": "genui/v1",
  "root": {
    "type": "Card",
    "props": { "variant": "elevated", "className": "rounded-2xl p-4" },
    "a11y": { "role": "group", "aria-label": "Weather for Seoul" },
    "children": [
      { "type": "Text",
        "props": { "as": "h2", "text": "Seoul", "className": "text-lg font-semibold" } },
      { "type": "Chart",
        "props": {
          "library": "vega-lite",
          "spec": {
            "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
            "data": { "values": [{"d":"Mon","t":18},{"d":"Tue","t":21}] },
            "mark": "line",
            "encoding": {
              "x": {"field":"d","type":"ordinal"},
              "y": {"field":"t","type":"quantitative"}
            }
          }
        } },
      { "type": "Button",
        "props": { "label": "Refresh", "className": "focus-visible:ring-2" },
        "actions": [{ "on": "click", "tool": "getWeather", "args": { "location": "Seoul" } }] }
    ]
  }
}
```

**합성 시 강제할 규칙**:
- 노드 = `type`(레지스트리 키) + `props`(타입화) + `children`(배열) + 선택 `actions`(tool-call로 환원) + 선택 `a11y`(role/aria).
- 차트/다이어그램은 재정의하지 말고 `type:"Chart"/"Diagram"`의 `props.spec`에 Vega-Lite/Mermaid 원문 임베드.
- `actions`는 tool-call 형태(`tool` + `args`)로 정규화 → 인터랙션과 GenUI가 동일 어휘 공유.
- Tailwind 클래스는 `props.className`에 **모바일 퍼스트**, 접근성은 `a11y`에 분리 → 코드 컴파일 시 시맨틱 태그/ARIA 자동 주입.
- 기본값으로 baking할 것: 시맨틱 HTML(`<button>/<nav>/<main>/<form>+<label for>`), 핵심 ARIA(`aria-label`/`aria-live="polite"`/장식 아이콘 `aria-hidden`), WCAG AA 대비, 키보드 포커스(`focus-visible:ring-2`).

### 1.2 학습 레코드 포맷 (JSONL, thinking on/off 라벨 포함)

```json
{
  "id": "genui-syn-000123",
  "domain": "genui",
  "subdomain": "component_tree",            // code | component_tree | tool_widget | chart_spec | diagram
  "format": "tree+code",                     // code | tree | tool_call | spec
  "lang": "en",                              // en | ko
  "difficulty": 4,                            // 1-5 (Evol 진화 단계 = 난이도 프록시)
  "reasoning": "on",                         // on | off  (hybrid toggle 명시 라벨)
  "messages": [
    { "role": "user", "content": "<NL 요구사항>" },
    { "role": "assistant",
      "content": "<think>레이아웃/접근성/상태관리 추론...</think>\n<트리 JSON 또는 코드>" }
  ],
  "artifacts": {                              // verification 산출물 (학습엔 미포함, 메타)
    "tree": { /* genui/v1 */ },
    "code": "<compiled React/Tailwind>",
    "screenshot": "sha256:...",
    "tests_passed": 6, "tests_total": 6
  },
  "verify": { "build": true, "render": true, "axe_violations": 0, "visual_score": 0.83 },
  "teacher": { "idea": "Mistral-Small-3", "impl": "Qwen2.5-Coder-32B", "trace": "QwQ-32B" },
  "license_trace": ["the-stack-v2:permissive", "shadcn-ui:MIT"]
}
```

- `reasoning:"off"`이면 assistant content의 `<think>`를 비우거나(`<think></think>`) 생략. easy(난이도 1-2)는 대부분 off, hard(4-5)는 on + long.
- `artifacts`/`verify`/`teacher`/`license_trace`는 학습 입력엔 넣지 않되 데이터 감사·재현·decontam에 사용.

---

## 2. Teacher 모델 선정 (중간 GPU ~30-70B 맞춤)

대형 teacher를 못 쓰므로 **dense ≤32B**로 역할 분담. (모두 license-clean. MoE 대형(GLM-4.5-Air 106B 등)은 메모리 초과로 제외.)

| 역할 | 모델 (예시) | 라이센스 | 비고 |
|---|---|---|---|
| 아이디어/다양성 (작고 쌈) | Mistral-Small-3 (~24B), Qwen3-14B | Apache-2.0 | WebSight식 "아이디어 10개" 생성 담당 |
| 코드 구현 (핵심 품질) | **Qwen2.5-Coder-32B**, gpt-oss-20B | Apache-2.0 | HTML/React/Tailwind 구현 |
| 일반/Magpie aligned | Qwen3-32B, Mistral-Small-3 | Apache-2.0 | UI-편향 system prompt로 instruction 추출 |
| long reasoning trace | **QwQ-32B**, DeepSeek-R1-Distill-Qwen-32B, Qwen3-32B(thinking) | Apache-2.0 / MIT | hard 샘플 `<think>` 생성 |
| VLM judge (말단) | Qwen2.5-VL-32B, Qwen3-VL | Apache-2.0 | reference 없을 때 시각 채점 |

**다양성 캡(§1.3 정책 준수)**: 단일 teacher ≤50%. 구현은 Qwen2.5-Coder-32B가 다수겠지만 gpt-oss-20B·Qwen3-32B로 분산. trace는 QwQ/R1-Distill 혼합. 모든 샘플에 `teacher` 필드 기록.

**중간 GPU 운영 메모**: 32B dense를 vLLM/SGLang으로 서빙(2×80GB 또는 4×48GB, 필요 시 AWQ/GPTQ 4bit). throughput이 병목이므로 (a) easy/no-think 샘플은 출력이 짧아 싸다 → 비중을 높여 평균 비용↓, (b) long-CoT는 hard에만, (c) self-training으로 채굴률을 올려 재생성 라운드의 효율 개선.

---

## 3. Stage 0 — 시드 코퍼스 수집

OSS-Instruct/Flame은 **실제 코드 스니펫**을 시드로 써야 현실성·다양성이 확보된다.

- **the-stack-v2** HTML/CSS/JSX/TSX/Vue/Svelte 슬라이스 (permissive 라이센스만 필터, SoftwareHeritage 동의, attribution).
- **shadcn/ui · Radix · MUI** GitHub 리포 + **shadcn `components.json`/`registry.json` 스키마**(컴포넌트/의존성/props가 플랫 파일로 노출 → 구조화 트리 시딩의 1순위).
- 스니펫은 **256~512 LoC**로 절단(teacher 컨텍스트 비용 통제).
- 한국어: 한국어 UI 요구사항 시드(쇼핑/금융/공공 서비스 화면 등) 소량 수기 작성 + Qwen3-32B로 한국어 프롬프트 확장.

---

## 4. Stage 1 — Instruction / Seed 생성

5개 기법을 **병렬·상보적**으로 운용한다(단일 기법은 분포 편향).

| 기법 | 무엇 / UI 적용 | 산출물 | 출처 |
|---|---|---|---|
| **OSS-Instruct** (Magicoder) | 실제 스니펫 → "영감받은 (task, 코드)" 생성. LLM 편향을 실제 코드로 깨 현실성↑ | 프론트엔드 (요구, 코드) 쌍 | [2312.02120](https://arxiv.org/abs/2312.02120) |
| **Magpie** | aligned 모델에 user 턴 직전까지만 넣어 query 자가 추출. seed 불필요, 가장 쌈. **UI 엔지니어 system prompt 고정**으로 UI-편향 | UI instruction 대량 | [2406.08464](https://arxiv.org/abs/2406.08464) |
| **WebSight 2단계** | Mistral-Small(아이디어 다양성) → Qwen2.5-Coder(HTML+Tailwind **단일파일**). Lorem 금지, 이미지=Unsplash dynamic API placeholder, Playwright 렌더 | self-contained 페이지 | [2403.09029](https://arxiv.org/html/2403.09029v1) |
| **Flame 3종** (React 특화) | Evolution(기능/스타일 변형) · Waterfall(요구→설계→구현 누적) · Additive(실제 컴포넌트에 상태/a11y/API 점진 추가). import 인라이닝+mock 자동생성+reflective agent 자가수정으로 **렌더 가능화** | React 컴포넌트 | [2503.01619](https://arxiv.org/html/2503.01619v1) |
| **Evol-Instruct** | instruction을 단계적 진화(제약/심화/입력복잡화). **진화 단계 수 = 난이도 라벨**(§7 직결) | 난이도 사다리 | [2306.08568](https://arxiv.org/pdf/2306.08568) |

**역할 배분**: WebSight식 = HTML/Tailwind 페이지(A·D 보강), Flame = React 컴포넌트(A·B), OSS-Instruct = 라이브러리 기반 현실성, Magpie = 저비용 instruction 대량, Evol = 난이도 분포 + hard 샘플 생성.

---

## 5. Stage 2 — 구조화 UI / 컴포넌트 트리 합성 (C, ~1.8B 전량 합성)

공개 데이터가 없으므로 **실제 라이브러리 부트스트랩 + round-trip 일관성**으로 만든다.

1. **라이브러리 시딩**: shadcn/Radix/MUI 컴포넌트 카탈로그(props 타입·variant·a11y 계약)를 시드로 OSS-Instruct/Flame-Additive 적용 → 실제 prop 스키마가 트리에 반영.
2. **code → tree**: 실제 JSX/HTML을 AST/DOM 파싱해 `genui/v1` 트리로 정규화.
3. **tree → code**: 트리에서 React/Tailwind 코드 재생성.
4. **round-trip 일관성 게이트(강한 무료 검증)**: 재생성 코드의 렌더 스크린샷이 원본과 일치(Design2Code block-match/SSIM) **AND** 재파싱 트리가 동일 → **양방향 일관 샘플만 채택**. 라벨 정합성 자동 보장.
5. **tool→component**: 컴포넌트+props 스키마에서 tool 정의 생성 → teacher가 "NL 요구 → tool-call(`name`+`args`) → 대상 컴포넌트" 데이터 생성.
6. **schema-constrained decoding**: JSON 트리 출력은 JSON-Schema grammar로 강제 디코딩 → 파싱 실패율을 구조적으로 0에 수렴(teacher 생성·4B 학습/추론 모두).
7. **차트/다이어그램**: NL → Vega-Lite JSON / Mermaid 텍스트 페어를 별도 합성 후 트리 leaf로 임베드(D 도메인과 공유).

---

## 6. Stage 3 — 검증 캐스케이드 (실행 기반, 최대 품질 레버)

**싼 → 비싼 순 8단계.** 앞단이 대부분을 reject하여 비싼 VLM-judge는 소수에만 도달(중간 GPU 비용 통제의 핵심).

| # | 단계 | 도구 | reject 기준 | 비용 |
|---|---|---|---|---|
| 1 | 컴파일/빌드 게이트 | `tsc`, `vite build`/`next build` | 컴파일/빌드 실패 | CPU, ~무료 |
| 2 | 정적 검사 | ESLint, Prettier, `eslint-plugin-jsx-a11y` | lint 에러 / 포맷 불안정 / 정적 a11y 위반 | CPU, ~무료 |
| 3 | headless 렌더 | Playwright/Puppeteer | 콘솔 에러·예외·흰 화면·0px 콘텐츠·404 | 수백ms~수초 |
| 4 | 동적 접근성 | axe-core 주입 | WCAG 위반 임계 초과(또는 a11y 라벨 부여) | 수십ms |
| 5 | e2e 인터랙션 테스트 | Playwright + AceCoder식 합성 테스트 | pass-rate < τ ("버튼 클릭→모달", "폼 제출→검증") | 초 단위 |
| 6 | 시각 검증 | reference 有: CLIP + **Design2Code(Block-Match/Text/Position/Color)** + CW-SSIM / reference 無: **VLM-judge** | 점수 < τ | CLIP 저렴, VLM 비쌈 |
| 7 | dedup | 정규화 MinHash+LSH → 렌더 스크린샷 임베딩 DBSCAN(eps≈0.25) | 근중복 | 중간 |
| 8 | Self-Refine 재투입 | 에러/렌더 diff feedback → teacher 1~2회 수정 | 통과 시 데이터화 + **수정 추론을 trace로 재활용** | teacher 1-2회 |

**Accept gate**: `build OK ∧ lint clean ∧ render error-free ∧ axe ≤ thr ∧ e2e pass ≥ τ ∧ (시각 ≥ τ) ∧ trace 검증 통과 ∧ non-dup ∧ non-contaminated`.

**임계값 주의**: UICoder 원문 값(텍스트 CLIP ≥0.35, 시각 ≥0.75, DBSCAN eps 0.25, 상위 0.5%, 초기 채굴률 0.4%)은 **SwiftUI/모바일 기준** → HTML/React에선 재튜닝 필요. 풀페이지 웹 스크린샷은 CLIP 신호가 약하므로 **Design2Code element-matching을 주력**으로, CLIP은 보조로.

핵심 레퍼런스: UICoder [2406.07739](https://arxiv.org/html/2406.07739v1), AceCoder [2502.01718](https://arxiv.org/abs/2502.01718), Design2Code [2403.03163](https://arxiv.org/abs/2403.03163), Self-Refine [2303.17651](https://arxiv.org/pdf/2303.17651).

---

## 7. Stage 4 — Reasoning trace 합성 + 길이 제어 (hybrid toggle)

사용자 핵심 목표(쉬운 건 짧게/없이, 어려운 건 길게)를 **데이터 단계에서 직접 라벨링**한다.

### 7.1 think/no_think 페어 생성
- 한 UI task당 (a) long reasoning teacher(QwQ-32B/R1-Distill-32B)로 **`<think>...</think>` 포함 long trace**, (b) non-thinking 모드로 **`<think></think>` 빈/직답**을 생성 → **paired (think, no_think)** (HiPO [2509.23967](https://arxiv.org/pdf/2509.23967), Qwen3 thinking-fusion [2505.09388](https://arxiv.org/abs/2505.09388)).

### 7.2 난이도 라벨 (easy=skip-think 학습)
난이도 프록시 다중화:
- **Evol 진화 단계 수**(§4): 1-2단계=easy → `reasoning:off`/짧은 think, 4-5단계(반응형+상태+a11y+성능)=hard → `reasoning:on` long.
- **§6 첫 통과 시도의 pass@1 / 컴파일 성공률**: 한 번에 성공하면 easy.
- 코드 LoC / 컴포넌트 수 / teacher 자가평가 난이도.
- **trivial UI(버튼/카드)는 think 비우기 샘플을 충분히 섞어** "생각 건너뛰기"를 명시 강화.

### 7.3 길이 제어 (budget)
- AdaCtrl [2505.18822](https://arxiv.org/pdf/2505.18822): long/short 혼합 cold-start + 난이도-aware 길이 budget. Budget Guidance [2506.13752](https://arxiv.org/pdf/2506.13752): "think within N tokens" 프롬프트로 길이 분포 다양화 → 4B가 budget 조절을 학습.

### 7.4 trace 검증 (환각 추론 제거)
- DeepSeek-R1식 rejection [2501.12948](https://arxiv.org/html/2501.12948v1): **trace가 만든 코드가 §6를 통과한 trace만 채택.** GenUI는 trace 품질을 렌더/테스트 통과로 객관 검증 가능 → 텍스트 추론 대비 큰 이점.
- 비용: long-CoT는 토큰 폭증 → hard에만. easy를 짧게/없게 하여 평균 비용↓ (hybrid 학습 목표 = 데이터 비용 절감 동시 달성).

---

## 8. Stage 5 — Dedup & Decontamination (GenUI 특화)

### 8.1 내부 dedup
- **정규화 후 MinHash+LSH** (13-gram, num_perm=128). HTML/JSX는 boilerplate(공통 head, Tailwind reset, import) 때문에 표면 n-gram이 과겹쳐 **거짓 중복** 다발 → 정규화 필수(주석/공백/import/Unsplash 시드 URL 제거, 클래스 순서 정렬).
- **렌더 스크린샷 임베딩 DBSCAN dedup**: 코드는 달라도 보이는 게 같으면 중복 처리(UICoder식). 웹 UI의 진짜 다양성 척도는 시각 임베딩.

### 8.2 eval decontamination (§1.4 + GenUI 평가셋)
대상: **WebDev Arena(최우선)**, Design2Code(+HARD/-hf), Sketch2Code, Web2Code eval, WebCode2M, WebGen-Bench, DesignBench.
3계층 적용:
1. **n-gram(13-gram) lexical** — 빠름, 1차 대량 제거.
2. **임베딩 semantic near-dup** — paraphrase/translation 누수 차단.
3. **LLM-decontaminator** — rephrase 누수.
4. **시각 decontam(GenUI 고유)**: Design2Code/WebDev Arena **스크린샷을 CLIP 임베딩**해 합성 샘플 렌더와 시각 유사도 임계 이상이면 제거(코드는 달라도 같은 페이지 재현 = 누수).

출처: contamination survey [2502.14425](https://arxiv.org/html/2502.14425v2), rephrased samples [2311.04850](https://arxiv.org/pdf/2311.04850).

---

## 9. Stage 6 — 반복 Self-Training (UICoder식 yield 개선)

중간 GPU에서 초기 채굴률이 낮으므로 **생성→필터→학습→재생성을 4-5 라운드** 반복:
1. R0: teacher들로 Stage 1-5 → 1차 정제 데이터 → 4B SFT.
2. R1+: 개선된 4B(또는 협업 teacher)로 Stage 1-3 재생성 → 통과율 상승 → 누적.
3. preference 단계(선택): 같은 입력 N=10 샘플 → "컴파일 가능 > 불가능, 가능끼리는 시각점수순" 자동 랭킹 → DPO 페어(추후 RL 단계용).

UICoder는 이 반복으로 초기 0.4% 채굴률을 라운드마다 끌어올렸다. mid-training 데이터는 SFT 분량만 쓰고, preference/RL 페어는 별도 후속 단계 자산으로 보관.

---

## 10. 토큰 예산 매핑 & 단계적 롤아웃

### 10.1 합성 산출 → §4.5 예산 매핑

| §4.5 항목 | 합성 목표(accept) | 주 생성 기법 | 형식 |
|---|---|---|---|
| C. 컴포넌트 트리 (전량 합성) | ~1.8B | 라이브러리 부트스트랩 + round-trip(§5) | tree+code, tool_call |
| B. UI reasoning (think→code) | ~1.0B | Evol hard + R1/QwQ trace(§7), Flame Waterfall | code/tree + `<think>` |
| 한국어 UI 보너스 | ~0.15B | 한국어 시드 + Qwen3-32B(§3) | code/tree (ko) |
| A·D 보강 (선택) | 가변 | WebSight식, NL→Vega-Lite/Mermaid | code, spec |
| **합계(accept)** | **~3B+** | | |

### 10.2 thinking 비율
도메인 전체 thinking 70% / non-thinking 30% 기조(§7) 유지하되, **GenUI는 easy 컴포넌트 비중이 높아 non-thinking을 약간 더(예: 60/40)** 배치 — "단순 UI엔 생각 안 함"을 강하게 학습. 명시적 `reasoning: on/off` 라벨.

### 10.3 중간 GPU throughput 현실 체크
- accept율을 보수적으로 **10-20%**로 보면, 3B accept를 위해 **~15-30B raw 생성**(+ long-CoT 인플레이션) 필요.
- 32B dense를 vLLM로 멀티 GPU 서빙 시 집계 수천 tok/s 가정 → **수 주~1-2개월 규모**(라운드 병렬·easy 비중↑·정적 게이트 선제거로 단축).
- **단계적 롤아웃**:
  - **Phase 1 (PoC, ~2주)**: WebSight식 HTML/Tailwind + 정적 게이트(컴파일/lint/렌더)만으로 ~200-300M 빠르게 확보. 파이프라인·임계값 튜닝.
  - **Phase 2 (~1개월)**: Flame React + 컴포넌트 트리 round-trip + e2e 테스트 + reasoning trace(hard) 추가 → ~1.5B.
  - **Phase 3**: self-training 2-3 라운드 + 한국어 + 시각 decontam 정밀화 → ~3B+ 달성, preference 페어 부산물 적재.

---

## 11. 인프라 / 툴링 스택

| 레이어 | 도구 |
|---|---|
| teacher 서빙 | vLLM / SGLang (AWQ·GPTQ 4bit 옵션), 멀티 GPU |
| 파이프라인 오케스트레이션 | datatrove(토큰측정·MinHash·decontam) + 커스텀 DAG (참고: SyGra [2508.15432](https://arxiv.org/pdf/2508.15432)) |
| 빌드/정적 | Node + tsc + Vite/Next + ESLint + Prettier + eslint-plugin-jsx-a11y |
| 렌더/테스트 | Playwright(스크린샷·콘솔·e2e) + axe-core, 컨테이너 풀 병렬화 |
| 시각/임베딩 | CLIP, CW-SSIM, Design2Code 메트릭 구현(JV 매칭/CIEDE2000), DBSCAN |
| 구조화 강제 | JSON-Schema grammar (constrained decoding) |
| VLM judge | Qwen2.5-VL-32B / Qwen3-VL |
| 라벨/감사 | 샘플별 teacher·license·verify 로그 JSONL |

---

## 12. 리스크 & 완화

| 리스크 | 완화 |
|---|---|
| 중간 GPU throughput 병목 | 정적 게이트 선제거, easy/no-think 비중↑(출력 짧음), self-training으로 채굴률↑, Phase 롤아웃 |
| 합성 분포 편향(LLM 동질화) | OSS-Instruct(실제 스니펫 시딩) + Magpie + 다수 teacher 혼합 + 시각 임베딩 dedup으로 다양성 강제 |
| 환각 추론(trace 오염) | trace→코드가 §6 통과한 것만 채택(R1 rejection) |
| eval 누수 | n-gram+임베딩+LLM-decontam **+ 시각 decontam**(GenUI 고유) |
| 라이센스 감사 | teacher OPEN-only 강제 + 샘플 단위 teacher/버전 로깅 + permissive 시드만 |
| 컴포넌트 트리 라벨 부정합 | code↔tree round-trip 양방향 일관 샘플만 채택 + schema-constrained decoding |

### 액션 아이템 (착수 순)
1. `genui/v1` 스키마 + 컴포넌트 레지스트리(shadcn/Radix 기반) 확정, 학습 레코드 JSONL 스펙 동결.
2. teacher 서빙(Qwen2.5-Coder-32B/Qwen3-32B/QwQ-32B/Mistral-Small) + Playwright/axe/eslint-a11y 검증 워커 구축.
3. Phase 1 PoC: WebSight식 HTML/Tailwind + 정적+렌더 게이트로 ~200-300M, 임계값 튜닝.
4. Flame React + 컴포넌트 트리 round-trip + e2e 테스트 + Evol 난이도 라벨 + reasoning trace(hard).
5. dedup/decontam(시각 포함) + self-training 2-3 라운드 → ~3B+, preference 페어 적재.

---

## 부록 A. 프롬프트 템플릿 스케치

**OSS-Instruct (구현 teacher)**:
```
다음은 실제 오픈소스 프런트엔드 코드 스니펫이다:
---
{snippet}
---
이 코드에서 영감을 받아, 현실적인 UI 개발 요구사항 1개와 그에 대한
완성도 높은 {React+Tailwind | HTML+Tailwind 단일파일} 구현을 작성하라.
- Lorem Ipsum 금지, 실제적인 문구 사용
- 시맨틱 HTML + label 연결 + 핵심 ARIA + 모바일퍼스트 Tailwind
- 이미지는 https://source.unsplash.com/random/{W}x{H}/?{keyword}
출력: ## 요구사항 / ## 코드
```

**Magpie (UI-편향, aligned teacher)**: system="너는 시니어 프런트엔드/디자인시스템 엔지니어다." 후 user 턴 직전까지만 주고 query 자가 생성 → §6 검증.

**reasoning trace (hard, long-CoT teacher)**:
```
요구사항: {requirement}
먼저 <think> 안에서 레이아웃 구조, 접근성(ARIA/키보드), 반응형 브레이크포인트,
상태관리, 엣지케이스를 단계적으로 설계하라. (목표 길이 ≈ {budget} 토큰)
그 다음 최종 {트리 JSON | 코드}를 출력하라.
```

**easy (no-think)**: `<think></think>` 강제 후 바로 코드 — trivial 컴포넌트(버튼/뱃지/카드)에 적용.

## 부록 B. 핵심 출처
- 시드: Magicoder/OSS-Instruct [2312.02120](https://arxiv.org/abs/2312.02120), WizardCoder/Evol [2306.08568](https://arxiv.org/pdf/2306.08568), Magpie [2406.08464](https://arxiv.org/abs/2406.08464), WebSight [2403.09029](https://arxiv.org/html/2403.09029v1) + [HF blog](https://huggingface.co/blog/websight), Flame [2503.01619](https://arxiv.org/html/2503.01619v1)
- 검증: UICoder [2406.07739](https://arxiv.org/html/2406.07739v1), AceCoder [2502.01718](https://arxiv.org/abs/2502.01718), Design2Code [2403.03163](https://arxiv.org/abs/2403.03163), Self-Refine [2303.17651](https://arxiv.org/pdf/2303.17651)
- hybrid reasoning: Qwen3 [2505.09388](https://arxiv.org/abs/2505.09388), DeepSeek-R1 [2501.12948](https://arxiv.org/html/2501.12948v1), HiPO [2509.23967](https://arxiv.org/pdf/2509.23967), AdaCtrl [2505.18822](https://arxiv.org/pdf/2505.18822), Budget Guidance [2506.13752](https://arxiv.org/pdf/2506.13752)
- 형식: Vercel AI SDK 6 [blog](https://vercel.com/blog/ai-sdk-6) + [generative UI](https://ai-sdk.dev/docs/ai-sdk-ui/generative-user-interfaces), Airbnb SDUI [deep-dive](https://medium.com/airbnb-engineering/a-deep-dive-into-airbnbs-server-driven-ui-system-842244c5f5), JSON Forms [docs](https://jsonforms.io/docs/uischema/controls/), Vega-Lite [spec](https://vega.github.io/vega-lite/docs/spec.html), Mermaid [syntax](https://mermaid.js.org/syntax/flowchart.html), shadcn [components.json](https://ui.shadcn.com/docs/components-json)
- decontam: survey [2502.14425](https://arxiv.org/html/2502.14425v2), rephrased [2311.04850](https://arxiv.org/pdf/2311.04850)
