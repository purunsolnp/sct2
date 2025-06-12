from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import uuid
import os
from sqlalchemy import create_engine, text
from openai import OpenAI  # 수정된 import
import json
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# FastAPI 앱 초기화
app = FastAPI(title="SCT 자동 해석 시스템", version="1.0.0")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 클라이언트 초기화
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("⚠️ Supabase 환경변수가 설정되지 않았습니다. 메모리 모드로 실행됩니다.")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI 클라이언트 초기화 (개선됨)
if not OPENAI_API_KEY:
    print("⚠️ OpenAI API 키가 설정되지 않았습니다. 해석 기능이 비활성화됩니다.")
    openai_client = None
else:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"⚠️ OpenAI 클라이언트 초기화 실패: {e}")
        openai_client = None

# 메모리 저장소 (Supabase 없을 때 사용)
MEMORY_SESSIONS = {}

# SCT 문항 데이터
SCT_ITEMS = [
    "나에게 이상한 일이 생겼을 때",
    "내 생각에 가끔 아버지는",
    "우리 윗사람들은",
    "나의 장래는",
    "어리석게도 내가 두려워하는 것은",
    "내 생각에 참다운 친구는",
    "내가 어렸을 때는",
    "남자에 대해서 무엇보다 좋지 않게 생각하는 것은",
    "내가 바라는 여인상(女人像)은",
    "남녀가 같이 있는 것을 볼 때",
    "내가 늘 원하기는",
    "다른 가정과 비교해서 우리 집안은",
    "나의 어머니는",
    "무슨 일을 해서라도 잊고 싶은 것은",
    "내가 믿고 있는 내 능력은",
    "내가 정말 행복할 수 있으려면",
    "어렸을 때 잘못했다고 느끼는 것은",
    "내가 보는 나의 앞날은",
    "대개 아버지들이란",
    "내 생각에 남자들이란",
    "다른 친구들이 모르는 나만의 두려움은",
    "내가 싫어하는 사람은",
    "결혼 생활에 대한 나의 생각은",
    "우리 가족이 나에 대해서",
    "내 생각에 여자들이란",
    "어머니와 나는",
    "내가 저지른 가장 큰 잘못은",
    "언젠가 나는",
    "내가 바라기에 아버지는",
    "나의 야망은",
    "윗사람이 오는 것을 보면 나는",
    "내가 제일 좋아하는 사람은",
    "내가 다시 젊어진다면",
    "나의 가장 큰 결점은",
    "내가 아는 대부분의 집안은",
    "완전한 남성상(男性像)은",
    "내가 성교를 했다면",
    "행운이 나를 외면했을 때",
    "대개 어머니들이란",
    "내가 잊고 싶은 두려움은",
    "내가 평생 가장 하고 싶은 일은",
    "내가 늙으면",
    "때때로 두려운 생각이 나를 휩쌀 때",
    "내가 없을 때 친구들은",
    "생생한 어린 시절의 기억은",
    "무엇보다도 좋지 않게 여기는 것은",
    "나의 성 생활은",
    "내가 어렸을 때 우리 가족은",
    "나는 어머니를 좋아했지만",
    "아버지와 나는"
]

# 문항별 해석 가이드 상수
SCT_ITEM_CATEGORIES = {
    "가족관계": [2, 13, 19, 26, 29, 39, 48, 49, 50],
    "대인관계": [6, 22, 32, 44],
    "자아개념": [15, 34, 30],
    "정서조절": [5, 21, 40, 43],
    "성_결혼관": [8, 9, 10, 23, 25, 36, 37, 47],
    "미래전망": [4, 16, 18, 28, 41, 42],
    "과거경험": [7, 17, 33, 45],
    "현실적응": [1, 3, 11, 31, 38, 46],
    "성격특성": [12, 14, 20, 24, 27, 35],
}

# Pydantic 모델들
class SCTResponse(BaseModel):
    item_no: int
    stem: str
    answer: str

class SCTSession(BaseModel):
    session_id: str
    assigned_by: str
    patient_name: str
    status: str
    created_at: datetime
    submitted_at: Optional[datetime] = None
    expires_at: datetime
    responses: Optional[List[SCTResponse]] = None
    interpretation: Optional[str] = None

class CreateSCTSessionRequest(BaseModel):
    assigned_by: str
    patient_name: str

class SubmitSCTResponseRequest(BaseModel):
    responses: List[SCTResponse]

class SCTSessionResponse(BaseModel):
    session_id: str
    assigned_by: str
    patient_name: str
    status: str
    created_at: datetime
    submitted_at: Optional[datetime]
    expires_at: datetime
    responses: Optional[List[Dict[str, Any]]]
    interpretation: Optional[str]

# 유틸리티 함수들
def get_current_time():
    return datetime.utcnow()

def get_expiry_time():
    return get_current_time() + timedelta(hours=72)

def is_session_expired(expires_at: datetime) -> bool:
    return get_current_time() > expires_at

def convert_responses_to_dict(responses):
    """SCTResponse 객체들을 딕셔너리로 변환하는 헬퍼 함수"""
    if not responses:
        return None
    
    result = []
    for resp in responses:
        if isinstance(resp, SCTResponse):
            result.append({
                "item_no": resp.item_no,
                "stem": resp.stem,
                "answer": resp.answer
            })
        else:
            # 이미 딕셔너리인 경우
            result.append(resp)
    return result

# 메모리 저장소 함수들
async def create_sct_session_memory(assigned_by: str, patient_name: str) -> SCTSession:
    session_id = str(uuid.uuid4())
    current_time = get_current_time()
    expiry_time = get_expiry_time()
    
    session_data = {
        "session_id": session_id,
        "assigned_by": assigned_by,
        "patient_name": patient_name,
        "status": "incomplete",
        "created_at": current_time.isoformat(),
        "submitted_at": None,
        "expires_at": expiry_time.isoformat(),
        "responses": [],
        "interpretation": None
    }
    
    MEMORY_SESSIONS[session_id] = session_data
    
    return SCTSession(
        session_id=session_id,
        assigned_by=assigned_by,
        patient_name=patient_name,
        status="incomplete",
        created_at=current_time,
        submitted_at=None,
        expires_at=expiry_time,
        responses=[],
        interpretation=None
    )

async def get_sct_session_memory(session_id: str) -> SCTSession:
    if session_id not in MEMORY_SESSIONS:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    session_data = MEMORY_SESSIONS[session_id]
    
    return SCTSession(
        session_id=session_data["session_id"],
        assigned_by=session_data["assigned_by"],
        patient_name=session_data["patient_name"],
        status=session_data["status"],
        created_at=datetime.fromisoformat(session_data["created_at"]),
        submitted_at=datetime.fromisoformat(session_data["submitted_at"]) if session_data["submitted_at"] else None,
        expires_at=datetime.fromisoformat(session_data["expires_at"]),
        responses=[SCTResponse(**resp) for resp in session_data["responses"]] if session_data["responses"] else [],
        interpretation=session_data["interpretation"]
    )

async def list_sct_sessions_by_user_memory(assigned_by: str) -> List[SCTSession]:
    sessions = []
    for session_data in MEMORY_SESSIONS.values():
        if session_data["assigned_by"] == assigned_by:
            sessions.append(SCTSession(
                session_id=session_data["session_id"],
                assigned_by=session_data["assigned_by"],
                patient_name=session_data["patient_name"],
                status=session_data["status"],
                created_at=datetime.fromisoformat(session_data["created_at"]),
                submitted_at=datetime.fromisoformat(session_data["submitted_at"]) if session_data["submitted_at"] else None,
                expires_at=datetime.fromisoformat(session_data["expires_at"]),
                responses=[SCTResponse(**resp) for resp in session_data["responses"]] if session_data["responses"] else [],
                interpretation=session_data["interpretation"]
            ))
    return sessions

# 주요 함수들
async def create_sct_session(assigned_by: str, patient_name: str) -> SCTSession:
    if supabase is None:
        return await create_sct_session_memory(assigned_by, patient_name)
    
    try:
        session_id = str(uuid.uuid4())
        current_time = get_current_time()
        expiry_time = get_expiry_time()
        
        session_data = {
            "session_id": session_id,
            "assigned_by": assigned_by,
            "patient_name": patient_name,
            "status": "incomplete",
            "created_at": current_time.isoformat(),
            "expires_at": expiry_time.isoformat(),
            "responses": [],
            "interpretation": None
        }
        
        result = supabase.table("sct_sessions").insert(session_data).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="세션 생성에 실패했습니다")
        
        return SCTSession(
            session_id=session_id,
            assigned_by=assigned_by,
            patient_name=patient_name,
            status="incomplete",
            created_at=current_time,
            submitted_at=None,
            expires_at=expiry_time,
            responses=[],
            interpretation=None
        )
    except Exception as e:
        print(f"Supabase 오류, 메모리 모드로 전환: {e}")
        return await create_sct_session_memory(assigned_by, patient_name)

async def get_sct_session(session_id: str) -> SCTSession:
    if supabase is None:
        return await get_sct_session_memory(session_id)
    
    try:
        result = supabase.table("sct_sessions").select("*").eq("session_id", session_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        session_data = result.data[0]
        
        return SCTSession(
            session_id=session_data["session_id"],
            assigned_by=session_data["assigned_by"],
            patient_name=session_data["patient_name"],
            status=session_data["status"],
            created_at=datetime.fromisoformat(session_data["created_at"].replace('Z', '+00:00')),
            submitted_at=datetime.fromisoformat(session_data["submitted_at"].replace('Z', '+00:00')) if session_data["submitted_at"] else None,
            expires_at=datetime.fromisoformat(session_data["expires_at"].replace('Z', '+00:00')),
            responses=[SCTResponse(**resp) for resp in session_data["responses"]] if session_data["responses"] else [],
            interpretation=session_data["interpretation"]
        )
    except Exception as e:
        print(f"Supabase 오류, 메모리 모드로 시도: {e}")
        return await get_sct_session_memory(session_id)

async def list_sct_sessions_by_user(assigned_by: str) -> List[SCTSession]:
    if supabase is None:
        return await list_sct_sessions_by_user_memory(assigned_by)
    
    try:
        result = supabase.table("sct_sessions").select("*").eq("assigned_by", assigned_by).order("created_at", desc=True).execute()
        
        sessions = []
        for session_data in result.data:
            sessions.append(SCTSession(
                session_id=session_data["session_id"],
                assigned_by=session_data["assigned_by"],
                patient_name=session_data["patient_name"],
                status=session_data["status"],
                created_at=datetime.fromisoformat(session_data["created_at"].replace('Z', '+00:00')),
                submitted_at=datetime.fromisoformat(session_data["submitted_at"].replace('Z', '+00:00')) if session_data["submitted_at"] else None,
                expires_at=datetime.fromisoformat(session_data["expires_at"].replace('Z', '+00:00')),
                responses=[SCTResponse(**resp) for resp in session_data["responses"]] if session_data["responses"] else [],
                interpretation=session_data["interpretation"]
            ))
        
        return sessions
    except Exception as e:
        print(f"Supabase 오류, 메모리 모드로 시도: {e}")
        return await list_sct_sessions_by_user_memory(assigned_by)

async def submit_sct_response(session_id: str, responses: List[SCTResponse]) -> SCTSession:
    """SCT 검사 응답을 제출하고 세션 상태를 complete로 변경합니다."""
    
    # 응답 개수 검증
    if len(responses) != 50:
        raise HTTPException(status_code=400, detail="50개 문항을 모두 완성해주세요")
    
    # 1. 메모리 모드 확인
    if session_id in MEMORY_SESSIONS:
        session_data = MEMORY_SESSIONS[session_id]
        
        if session_data["status"] == "complete":
            raise HTTPException(status_code=400, detail="이미 완료된 세션입니다")
        
        response_data = []
        for response in responses:
            response_data.append({
                "item_no": response.item_no,
                "stem": response.stem,
                "answer": response.answer
            })
        
        session_data["responses"] = response_data
        session_data["status"] = "complete"
        session_data["submitted_at"] = get_current_time().isoformat()
        
        return await get_sct_session_memory(session_id)
    
    # 2. Supabase 모드
    if supabase is not None:
        try:
            # 먼저 세션이 존재하는지 확인
            session_check = supabase.table("sct_sessions").select("status").eq("session_id", session_id).execute()
            
            if not session_check.data:
                raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
            
            if session_check.data[0]["status"] == "complete":
                raise HTTPException(status_code=400, detail="이미 완료된 세션입니다")
            
            # 응답 데이터 준비
            current_time = get_current_time()
            response_data = [{"item_no": r.item_no, "stem": r.stem, "answer": r.answer} for r in responses]
            
            # Supabase 업데이트
            result = supabase.table("sct_sessions").update({
                "responses": response_data,
                "status": "complete",
                "submitted_at": current_time.isoformat()
            }).eq("session_id", session_id).execute()
            
            if not result.data:
                raise HTTPException(status_code=500, detail="응답 저장에 실패했습니다")
            
            return await get_sct_session(session_id)
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"Supabase 업데이트 실패: {e}")
            # Supabase 실패 시 메모리 모드로 폴백 (세션이 있다면)
            if session_id in MEMORY_SESSIONS:
                return await submit_sct_response(session_id, responses)
    
    # 3. 둘 다 실패한 경우
    raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

async def generate_sct_interpretation(session_id: str) -> str:
    """숙련된 정신과 의사 수준의 전문적인 SCT 해석을 생성합니다."""
    if openai_client is None:
        return "OpenAI API 키가 설정되지 않아 해석을 생성할 수 없습니다."
    
    try:
        session = await get_sct_session(session_id)
        
        if session.status != "complete":
            raise HTTPException(status_code=400, detail="완료된 검사만 해석 가능합니다")
        
        # 응답 데이터 구조화
        responses_text = ""
        for i, response in enumerate(session.responses, 1):
            responses_text += f"{i}. {response.stem} → {response.answer}\n"
        
        # 전문적인 SCT 해석을 위한 상세한 프롬프트
        system_prompt = """당신은 20년 이상의 임상 경험을 가진 숙련된 정신과 의사이자 임상심리학자입니다. 
문장완성검사(SCT)를 통해 환자의 심리적 상태를 종합적으로 분석하고 임상적으로 유의미한 해석을 제공해야 합니다.

다음 14개 영역을 중심으로 분석하세요:
1. 가족 관계 (부모, 형제자매, 가족 전반)
2. 대인관계 (친구, 동료, 상하관계)
3. 성 관련 태도 (성역할, 성적 관심, 결혼관)
4. 자아개념 (자아상, 자존감, 정체성)
5. 감정 조절 (불안, 우울, 분노, 두려움)
6. 미래 전망 (목표, 야망, 희망)
7. 과거 경험 (어린 시절, 트라우마, 회상)
8. 현실 적응 (스트레스 대처, 문제해결)
9. 방어기제 (부인, 투사, 합리화 등)
10. 성격 특성 (외향성/내향성, 충동성, 강박성)
11. 정신병리적 징후 (우울증, 불안장애, 성격장애 등)
12. 인지적 특성 (사고 패턴, 인지 왜곡)
13. 사회적 기능 (역할 수행, 사회적 기대)
14. 치료적 시사점 (강점, 취약성, 개입 방향)

전문적이고 임상적으로 유용한 보고서를 작성하세요."""

        user_prompt = f"""
다음은 {session.patient_name} 환자의 문장완성검사(SCT) 결과입니다.

**검사 결과:**
{responses_text}

위 결과를 바탕으로 다음 형식으로 전문적인 임상 해석 보고서를 작성해주세요:

## 1. 검사 개요
- 환자명, 검사일, 협조도 등

## 2. 주요 심리적 특성

### 2.1 가족 관계 및 초기 대상관계
- 부모에 대한 인식과 관계
- 가족 역동 및 애착 양상
- 초기 경험이 현재에 미치는 영향

### 2.2 대인관계 패턴
- 타인에 대한 기본 신뢰도
- 친밀감 형성 능력
- 갈등 해결 방식

### 2.3 자아개념 및 정체성
- 자기 인식과 자존감
- 개인적 강점과 취약성
- 정체성 발달 수준

### 2.4 정서적 특성
- 주요 정서적 이슈
- 감정 조절 능력
- 스트레스 반응 패턴

### 2.5 성격 구조 및 방어기제
- 주요 성격 특성
- 사용하는 방어기제
- 적응 수준

## 3. 정신병리학적 소견
- 관찰되는 증상이나 징후
- 진단적 고려사항
- 위험 요소 평가

## 4. 치료적 고려사항
- 치료 동기 및 준비도
- 예상되는 치료 과정
- 권고되는 개입 방향

## 5. 요약 및 권고사항
- 핵심 소견 요약
- 구체적인 치료 권고
- 추가 평가 필요성

각 영역별로 구체적인 응답을 인용하며 임상적 근거를 제시하고, 
정신건강의학과 전문의 수준의 전문적인 언어와 관점으로 작성해주세요.
"""

        # OpenAI API 호출
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # 최신 모델 사용
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000,
            temperature=0.3,  # 일관성을 위해 낮은 temperature 사용
        )
        
        interpretation = response.choices[0].message.content
        
        # 해석 결과를 데이터베이스에 저장
        if supabase is not None:
            try:
                supabase.table("sct_sessions").update({
                    "interpretation": interpretation
                }).eq("session_id", session_id).execute()
            except Exception as e:
                print(f"해석 저장 실패 (Supabase): {e}")
        elif session_id in MEMORY_SESSIONS:
            MEMORY_SESSIONS[session_id]["interpretation"] = interpretation
        
        return interpretation
        
    except Exception as e:
        error_msg = f"해석 생성 중 오류가 발생했습니다: {str(e)}"
        print(f"해석 생성 오류: {e}")
        
        # 오류 발생 시 기본 임상 보고서 형태로 반환
        fallback_interpretation = f"""
# SCT (문장완성검사) 해석 보고서

## 1. 검사 개요
- **환자명**: {session.patient_name}
- **검사 완료일**: {session.submitted_at.strftime('%Y년 %m월 %d일 %H시 %M분') if session.submitted_at else '알 수 없음'}
- **세션 ID**: {session_id}
- **검사 협조도**: 총 50문항 완료

## 2. 임상적 소견
OpenAI API 연결 오류로 인해 자동 해석을 생성할 수 없습니다.
수동 해석을 위해 다음 사항을 참고하시기 바랍니다:

### 주요 평가 영역
1. **가족 관계**: 문항 2, 13, 19, 26, 29, 39, 48, 49, 50
2. **대인관계**: 문항 6, 22, 32, 44
3. **자아개념**: 문항 15, 34, 30
4. **정서 조절**: 문항 5, 21, 40, 43
5. **성 및 결혼관**: 문항 8, 9, 10, 23, 25, 36, 37, 47
6. **미래 전망**: 문항 4, 16, 18, 28, 41, 42
7. **과거 경험**: 문항 7, 17, 33, 45
8. **현실 적응**: 문항 1, 3, 11, 31, 38, 46

## 3. 권고사항
- 전문 임상심리학자 또는 정신건강의학과 전문의의 직접 해석이 필요합니다.
- 각 문항별 응답을 14개 주요 영역으로 분류하여 종합적으로 분석하시기 바랍니다.
- 필요시 추가적인 심리검사나 임상면담을 고려하십시오.

*본 보고서는 시스템 오류로 인한 임시 보고서입니다.*
        """
        
        return fallback_interpretation

# API 엔드포인트들
@app.get("/sct/items")
async def get_sct_items():
    items = []
    for i, stem in enumerate(SCT_ITEMS, 1):
        items.append({
            "item_no": i,
            "stem": stem
        })
    return {"items": items, "total_count": len(SCT_ITEMS)}

@app.post("/sct/sessions", response_model=SCTSessionResponse)
async def create_session_endpoint(request: CreateSCTSessionRequest):
    session = await create_sct_session(request.assigned_by, request.patient_name)
    
    return SCTSessionResponse(
        session_id=session.session_id,
        assigned_by=session.assigned_by,
        patient_name=session.patient_name,
        status=session.status,
        created_at=session.created_at,
        submitted_at=session.submitted_at,
        expires_at=session.expires_at,
        responses=convert_responses_to_dict(session.responses),
        interpretation=session.interpretation
    )

@app.get("/sct/sessions/{session_id}", response_model=SCTSessionResponse)
async def get_session_endpoint(session_id: str):
    session = await get_sct_session(session_id)
    
    return SCTSessionResponse(
        session_id=session.session_id,
        assigned_by=session.assigned_by,
        patient_name=session.patient_name,
        status=session.status,
        created_at=session.created_at,
        submitted_at=session.submitted_at,
        expires_at=session.expires_at,
        responses=convert_responses_to_dict(session.responses),
        interpretation=session.interpretation
    )

@app.post("/sct/sessions/{session_id}/submit", response_model=SCTSessionResponse)
async def submit_response_endpoint(session_id: str, request: SubmitSCTResponseRequest):
    """SCT 검사 응답을 제출합니다."""
    try:
        print(f"📝 제출 요청 받음: session_id={session_id}, responses_count={len(request.responses)}")
        
        # 세션 제출 처리
        session = await submit_sct_response(session_id, request.responses)
        
        print(f"✅ 제출 성공: session_id={session_id}, status={session.status}")
        
        # SCTResponse 객체들을 딕셔너리로 변환
        responses_dict = convert_responses_to_dict(session.responses)
        
        return SCTSessionResponse(
            session_id=session.session_id,
            assigned_by=session.assigned_by,
            patient_name=session.patient_name,
            status=session.status,
            created_at=session.created_at,
            submitted_at=session.submitted_at,
            expires_at=session.expires_at,
            responses=responses_dict,
            interpretation=session.interpretation
        )
        
    except HTTPException as e:
        print(f"❌ HTTP 오류: {e.detail}")
        raise
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"제출 처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/sct/sessions/{session_id}/interpret")
async def generate_interpretation_endpoint(session_id: str):
    interpretation = await generate_sct_interpretation(session_id)
    return {
        "session_id": session_id,
        "interpretation": interpretation,
        "generated_at": get_current_time()
    }

@app.get("/sct/sessions/{session_id}/interpretation")
async def get_interpretation_endpoint(session_id: str):
    session = await get_sct_session(session_id)
    
    if not session.interpretation:
        interpretation = await generate_sct_interpretation(session_id)
    else:
        interpretation = session.interpretation
    
    return {
        "session_id": session_id,
        "patient_name": session.patient_name,
        "interpretation": interpretation,
        "submitted_at": session.submitted_at
    }

@app.get("/sct/sessions/{session_id}/analysis")
async def get_categorical_analysis(session_id: str):
    """카테고리별 응답 분석을 제공합니다."""
    try:
        session = await get_sct_session(session_id)
        
        if session.status != "complete":
            raise HTTPException(status_code=400, detail="완료된 검사만 분석 가능합니다")
        
        # 카테고리별 응답 분류
        categorized_responses = {}
        for category, item_numbers in SCT_ITEM_CATEGORIES.items():
            categorized_responses[category] = []
            for item_no in item_numbers:
                for response in session.responses:
                    if response.item_no == item_no:
                        categorized_responses[category].append({
                            "item_no": response.item_no,
                            "stem": response.stem,
                            "answer": response.answer
                        })
        
        return {
            "session_id": session_id,
            "patient_name": session.patient_name,
            "categorized_responses": categorized_responses,
            "analysis_date": get_current_time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {str(e)}")

@app.get("/sct/sessions/by-user/{user_id}")
async def list_sessions_by_user_endpoint(user_id: str):
    sessions = await list_sct_sessions_by_user(user_id)
    
    session_responses = []
    for session in sessions:
        session_responses.append(SCTSessionResponse(
            session_id=session.session_id,
            assigned_by=session.assigned_by,
            patient_name=session.patient_name,
            status=session.status,
            created_at=session.created_at,
            submitted_at=session.submitted_at,
            expires_at=session.expires_at,
            responses=convert_responses_to_dict(session.responses),
            interpretation=session.interpretation
        ))
    
    return {"sessions": session_responses, "total_count": len(session_responses)}

@app.get("/")
async def root():
    return {"message": "SCT 자동 해석 시스템이 정상적으로 작동 중입니다"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": get_current_time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)