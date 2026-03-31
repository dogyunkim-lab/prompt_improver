# Prompt Improver — 사용 설명서

## 1. 용어 정의

| 용어 | 설명 |
|------|------|
| **Task** | 개선하려는 프롬프트 실험의 최상위 단위. 하나의 생성 목적(예: "불편사항 요약")에 해당 |
| **Run** | Task 내 1회 실험 이터레이션. Run 1 → Run 2 → ... 로 반복 개선 |
| **Phase** | Run 내 6단계 파이프라인 (Phase 1~6). 각 Phase는 순차 실행 |
| **요약 유형 (generation_task)** | Dify 워크플로우에 전달되는 생성 목적 텍스트 (예: "불편사항 요약", "고객 민원 요약") |
| **Judge** | GPT가 생성 결과를 정답/과답/오답으로 판정하는 평가 모델 |
| **Judge JSON** | Judge가 사전에 평가한 케이스 목록 파일 (`[{id, stt, reference, keywords, generated, evaluation, reason}, ...]`) |
| **ObjectID** | Dify 워크플로우 고유 ID. Phase 3 전에 Dify에서 발급받아 입력 |
| **Delta** | 이전 Run 대비 케이스별 판정 변화 (improved / regressed / unchanged) |
| **LearningRate** | 이전 실험 성과 기반으로 자동 결정되는 변화 강도 (`explore` → `major` → `medium` → `minor`) |
| **score_total** | 정답+과답 비율 (%). 목표: 95% 이상 |

---

## 2. 전체 워크플로우

```
[Task 생성]
    │
    ▼
[Run 생성] ──── Judge JSON 업로드 (선택적)
    │
    ├─ Phase 1: 오류 분석 (GPT 버킷 분류)
    │       ↓ 오답/과답 케이스를 4가지 버킷으로 분류
    │
    ├─ Phase 2: 프롬프트 설계 (GPT 후보 생성)
    │       ↓ A/B/C 후보 프롬프트 설계 (explore or converge 모드)
    │
    ├─ Phase 3: Dify 실행 (Dify API 호출)
    │       ↓ ObjectID 등록 → 전체 케이스 생성 실행
    │
    ├─ Phase 4: Judge 재실행 (GPT 재판정)
    │       ↓ 생성 결과를 GPT Judge로 평가 → 점수 집계
    │
    ├─ Phase 5: 성과 분석 (자동 집계)
    │       ↓ Delta 분석, 추이 차트, 회귀 케이스 파악
    │
    └─ Phase 6: 전략 수립 (GPT 다음 방향 제시)
            ↓ backprop 분석, 유효/해로운 요소, 다음 방향 제시

    ▼
[목표 미달성] → 새 Run 시작 (Phase 6 결과를 참고)
[목표 달성] → 완료 (score_total ≥ 95%)
```

---

## 3. Step-by-step 사용법

### Step 1: Task 생성

1. 우측 상단 **[+ 새 Task]** 버튼 클릭
2. 입력:
   - **Task명**: 실험 이름 (예: "불편사항 요약 개선 v1")
   - **설명**: 선택적 메모
   - **요약 유형**: 자유 텍스트 입력 (예: "불편사항 요약") — Dify에 그대로 전달됨
3. **만들기** 클릭

### Step 2: Run 생성

1. Task 선택 후 **[+ 새 Run]** 버튼 클릭
2. 시작 방식 선택:
   - **Zero-start**: 처음부터 시작 (첫 Run 권장)
   - **Continue**: 이전 Run을 기반으로 이어가기
3. (선택) **Judge JSON** 파일 업로드:
   - 사전에 Judge가 평가한 케이스 목록 JSON
   - Phase 1에서 사용 (Phase 1 시작 전에도 업로드 가능)
4. **시작** 클릭

### Step 3: Phase 1 — 오류 분석

- Judge JSON이 없으면 먼저 업로드 필요
- **[Phase 1 분석 시작]** 클릭
- GPT가 오답/과답 케이스를 4가지 버킷으로 분류:
  - `stt_error`: STT 음성인식 오류
  - `prompt_missing`: 프롬프트에 지시가 없는 경우
  - `model_behavior`: 모델 고유 동작 특성
  - `judge_dispute`: Judge 판정 이견
- 완료 시 버킷 차트 + 케이스 테이블 표시

### Step 4: Phase 2 — 프롬프트 설계

- **[프롬프트 설계 시작]** 클릭
- GPT가 A/B/C 후보 프롬프트 생성
  - 첫 Run: `explore` 모드 (다양한 시도)
  - 이후 Run: `converge` 모드 (점수 기반 수렴)
- 각 후보의 노드별 프롬프트 확인 가능

### Step 5: Phase 3 — Dify 실행 (사전 준비 필요)

> **사전 준비**: Phase 2 결과를 바탕으로 Dify 워크플로우를 구성하고 ObjectID를 발급받아야 합니다.
> 자세한 내용은 [Dify 워크플로우 구성](#5-dify-워크플로우-구성)을 참조하세요.

1. Dify에서 발급받은 **ObjectID** 입력
2. **[연결 확인]** 클릭 → 연결 상태 확인
3. **[Dify 실행]** 클릭
4. 전체 케이스에 대해 생성 실행 (병렬 5건)

### Step 6: Phase 4 — Judge 재실행

- **[Judge 실행]** 클릭
- 생성된 결과를 GPT Judge가 재평가
- 완료 시 점수 집계 (정답+과답%, 정답%, 과답%, 오답%)

### Step 7: Phase 5 — 성과 분석

- Phase 5 탭 클릭 시 자동 로드
- 확인 항목:
  - 이번 Run 점수
  - Run별 성능 추이 차트
  - Delta (개선/회귀/변화없음 건수)
  - 회귀 케이스 목록
- **목표 달성 (95% 이상)** 시 "목표 달성!" 배너 표시

### Step 8: Phase 6 — 전략 수립

- **[전략 수립]** 클릭
- GPT가 다음 실험 방향을 제시:
  - Backprop 분석 (왜 틀렸는지)
  - 유효 요소 (다음 Run에 유지)
  - 해로운 요소 (다음 Run에서 제거)
  - 다음 방향 제안

### Step 9: 반복 또는 완료

- **목표 미달성**: Phase 6 결과를 참고하여 **새 Run 생성** (Continue 모드 권장)
- **목표 달성**: 최종 프롬프트 확인 후 서비스 적용

---

## 4. Judge JSON 파일 형식

```json
[
  {
    "id": "case_001",
    "stt": "고객 발화 텍스트...",
    "reference": "기준 요약문...",
    "keywords": "키워드1, 키워드2",
    "generated": "모델이 생성한 요약문...",
    "evaluation": "오답",
    "reason": "판정 이유..."
  },
  ...
]
```

**evaluation 값**: `정답` | `과답` | `오답`

---

## 5. Dify 워크플로우 구성

### Phase 3 실행 전 Dify에서 준비할 사항:

1. **Dify 접속** → 워크플로우 탭
2. **새 워크플로우 생성**:
   - 입력 변수: `stt` (대화 내용), `keywords` (키워드), `generation_task` (생성 목적)
   - 출력 변수: `generated` (생성된 요약)
3. **Phase 2 결과 반영**:
   - Phase 2에서 설계된 A/B/C 후보 중 실행할 후보의 노드 프롬프트를 Dify에 입력
   - 노드가 여러 개인 경우 (A→B→C) 체이닝 설정
4. **ObjectID 복사**: 워크플로우 상세 페이지 URL 또는 API 키에서 확인
5. **Phase 3 연결 확인** 후 실행

### Dify API 설정 (config.py / 환경변수):

```
DIFY_BASE_URL=https://your-dify-instance.com/v1
DIFY_API_KEY=your-api-key          # 기본 키 (ObjectID 인증 실패 시 fallback)
```

---

## 6. 서버 실행

```bash
cd prompt-improver
pip install -r requirements.txt
python main.py
```

브라우저에서 `http://localhost:8000` 접속

---

## 7. 자주 묻는 질문

**Q. Phase 1에서 "Judge JSON 파일이 없습니다" 오류가 뜹니다.**
A. Run 생성 시 Judge JSON을 업로드하거나, Phase 1 화면의 파일 업로드 영역에서 업로드 후 다시 시작하세요.

**Q. Phase 2에서 프롬프트 후보가 생성되지 않습니다.**
A. GPT 모델 응답에서 JSON 파싱 실패일 수 있습니다. 로그를 확인하고, 네트워크 연결 및 GPT_API_BASE 설정을 점검하세요.

**Q. Phase 3 Dify 연결 확인에 실패합니다.**
A. ObjectID가 올바른지, DIFY_BASE_URL이 정확한지, Dify 서버가 정상 동작 중인지 확인하세요.

**Q. Phase 4 점수가 이전과 동일합니다.**
A. Dify에서 Phase 2 설계 결과가 제대로 반영되었는지 확인하세요. 프롬프트가 바뀌지 않으면 점수도 비슷할 수 있습니다.

**Q. Learning Rate가 항상 "explore"로 나옵니다.**
A. 이전 Run들이 Phase 4까지 완료되어 score_total이 기록되어야 합니다. 완료된 Run이 없으면 항상 explore입니다.

**Q. 목표 95%는 어떤 기준인가요?**
A. `정답 + 과답` 케이스의 비율입니다. 오답이 5% 미만이면 목표 달성으로 간주합니다.

**Q. Run을 삭제하거나 되돌릴 수 있나요?**
A. 현재 UI에서는 삭제 기능이 없습니다. SQLite DB(`data/prompt_improver.db`)를 직접 수정하거나 새 Task를 생성하세요.
