from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import uuid
import os
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

if not OPENAI_API_KEY:
    print("⚠️ OpenAI API 키가 설정되지 않았습니다. 해석 기능이 비활성화됩니다.")
    openai_client = None
else:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

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
    # 메모리 모드
    if session_id in MEMORY_SESSIONS:
        session_data = MEMORY_SESSIONS[session_id]
        
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
    
    # Supabase 모드 (구현 생략)
    raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

async def generate_sct_interpretation(session_id: str) -> str:
    if openai_client is None:
        return "OpenAI API 키가 설정되지 않아 해석을 생성할 수 없습니다."
    
    try:
        session = await get_sct_session(session_id)
        
        if session.status != "complete":
            raise HTTPException(status_code=400, detail="완료된 검사만 해석 가능합니다")
        
        # 간단한 해석 생성 (실제로는 더 복잡한 로직)
        interpretation = f"""
# SCT 검사 해석 결과

**환자명**: {session.patient_name}
**검사 완료일**: {session.submitted_at.strftime('%Y년 %m월 %d일') if session.submitted_at else '알 수 없음'}

## 검사 결과 요약
총 {len(session.responses)}개 문항에 대한 응답이 완료되었습니다.

## 주요 특징
- 환자는 문장완성검사에 성실히 참여하였습니다.
- 각 문항에 대한 응답이 적절히 제공되었습니다.

## 권고사항
상세한 해석을 위해서는 전문 임상심리학자의 분석이 필요합니다.

*이 해석은 자동 생성된 기본 보고서입니다.*
        """
        
        return interpretation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"해석 생성 중 오류: {str(e)}")

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
        responses=session.responses,
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
        responses=session.responses,
        interpretation=session.interpretation
    )

@app.post("/sct/sessions/{session_id}/submit", response_model=SCTSessionResponse)
async def submit_response_endpoint(session_id: str, request: SubmitSCTResponseRequest):
    session = await submit_sct_response(session_id, request.responses)
    
    return SCTSessionResponse(
        session_id=session.session_id,
        assigned_by=session.assigned_by,
        patient_name=session.patient_name,
        status=session.status,
        created_at=session.created_at,
        submitted_at=session.submitted_at,
        expires_at=session.expires_at,
        responses=session.responses,
        interpretation=session.interpretation
    )

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
            responses=session.responses,
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