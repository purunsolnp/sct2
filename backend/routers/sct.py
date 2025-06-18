from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
from schemas import SessionCreate, SCTResponseCreate, SessionRead
import crud
from database_config import get_db
from auth_utils import get_current_user

router = APIRouter()

# SCT 검사 항목들
SCT_ITEMS = [
    "나는 사람들과 함께 있을 때",
    "나는 혼자 있을 때",
    "나는 미래에 대해",
    "나는 과거에 대해",
    "나는 성공했을 때",
    "나는 실패했을 때",
    "나는 화가 날 때",
    "나는 기쁠 때",
    "나는 슬플 때",
    "나는 화가 날 때",
    "나는 기쁠 때",
    "나는 슬플 때",
    "나는 화가 날 때",
    "나는 기쁠 때",
    "나는 슬플 때"
]

@router.get("/sct/items")
def get_sct_items():
    """SCT 검사 항목들을 반환합니다."""
    items = []
    for i, stem in enumerate(SCT_ITEMS, 1):
        items.append({
            "item_no": i,
            "stem": stem
        })
    return {"items": items, "total_count": len(SCT_ITEMS)}

@router.post("/sct/sessions")
def create_session(
    session: SessionCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """새로운 SCT 세션을 생성합니다."""
    doctor_id = current_user.doctor_id
    db_session = crud.create_session(db, session, doctor_id)
    
    # 디버깅을 위한 로깅 추가
    print("생성된 세션 정보:", {
        "session_id": db_session.session_id,
        "patient_name": db_session.patient_name,
        "doctor_id": db_session.doctor_id,
        "status": db_session.status,
        "created_at": db_session.created_at,
        "expires_at": db_session.expires_at
    })
    
    # 응답 데이터
    response_data = {
        "session_id": db_session.session_id,
        "patient_name": db_session.patient_name,
        "doctor_id": db_session.doctor_id,
        "status": db_session.status,
        "created_at": db_session.created_at,
        "expires_at": db_session.expires_at
    }
    
    return response_data

@router.get("/sct/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)):
    """특정 세션 정보를 조회합니다."""
    session = crud.get_session_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    return session

@router.post("/sct/sessions/{session_id}/submit")
def submit_response(session_id: str, responses: List[SCTResponseCreate], db: Session = Depends(get_db)):
    """SCT 검사 응답을 제출합니다."""
    try:
        # 세션 제출 처리
        updated_session = crud.submit_session_responses(db, session_id, responses)
        return {"message": "응답이 성공적으로 제출되었습니다", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"제출 처리 중 오류가 발생했습니다: {str(e)}")

@router.post("/sct/sessions/{session_id}/interpret")
def generate_interpretation(session_id: str, db: Session = Depends(get_db)):
    """SCT 해석을 생성합니다."""
    try:
        interpretation = crud.generate_interpretation(db, session_id)
        return {
            "session_id": session_id,
            "interpretation": interpretation,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"해석 생성 중 오류가 발생했습니다: {str(e)}")

@router.get("/sct/sessions/{session_id}/interpretation")
def get_interpretation(session_id: str, db: Session = Depends(get_db)):
    """SCT 해석을 조회합니다."""
    session = crud.get_session_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    if not session.interpretation:
        interpretation = crud.generate_interpretation(db, session_id)
    else:
        interpretation = session.interpretation
    
    return {
        "session_id": session_id,
        "patient_name": session.patient_name,
        "interpretation": interpretation,
        "submitted_at": session.submitted_at
    }

@router.get("/sct/sessions/by-user/{user_id}")
def list_sessions_by_user(user_id: str, db: Session = Depends(get_db)):
    """사용자별 세션 목록을 조회합니다."""
    sessions = crud.get_sessions_by_user(db, user_id)
    return {"sessions": sessions, "total_count": len(sessions)} 