from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
import json
from schemas import SessionCreate, SCTResponseCreate, SessionRead
import crud
from database_config import get_db
from auth_utils import get_current_user

router = APIRouter()

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
    try:
        print(f"=== 세션 생성 시작 ===")
        print(f"현재 사용자: {current_user.doctor_id}")
        print(f"사용자 상태 - is_verified: {current_user.is_verified}, is_active: {current_user.is_active}")
        print(f"환자 이름: {session.patient_name}")
        
        # 사용자 승인 상태 확인
        if not current_user.is_verified:
            print(f"승인되지 않은 사용자: {current_user.doctor_id}")
            raise HTTPException(
                status_code=403, 
                detail="승인되지 않은 계정입니다. 관리자에게 문의하세요."
            )
        
        # 사용자 활성 상태 확인
        if not current_user.is_active:
            print(f"비활성화된 사용자: {current_user.doctor_id}")
            raise HTTPException(
                status_code=403, 
                detail="비활성화된 계정입니다. 관리자에게 문의하세요."
            )
        
        doctor_id = current_user.doctor_id
        print(f"세션 생성 시작 - doctor_id: {doctor_id}, patient_name: {session.patient_name}")
        
        db_session = crud.create_session(db, session, doctor_id)
        print(f"CRUD에서 반환된 세션: {db_session.__dict__}")
        
        # 응답 데이터 준비 - 명시적으로 구성
        response_data = {
            "session_id": str(db_session.session_id),  # 문자열로 확실히 변환
            "patient_name": str(db_session.patient_name),
            "doctor_id": str(db_session.doctor_id),
            "status": str(db_session.status),
            "created_at": db_session.created_at.isoformat(),
            "submitted_at": db_session.submitted_at.isoformat() if db_session.submitted_at else None,
            "expires_at": db_session.expires_at.isoformat()
        }
        
        print("=== 최종 응답 데이터 ===")
        print(json.dumps(response_data, ensure_ascii=False, indent=2))
        print("=== session_id 값 ===")
        print(f"session_id: {response_data['session_id']}")
        print(f"session_id 타입: {type(response_data['session_id'])}")
        print("=====================")
        
        return response_data
        
    except HTTPException:
        # HTTPException은 그대로 다시 발생시킴
        raise
    except Exception as e:
        print(f"세션 생성 중 예상치 못한 오류 발생: {str(e)}")
        print(f"오류 타입: {type(e)}")
        import traceback
        print(f"스택 트레이스: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"세션 생성 중 오류가 발생했습니다: {str(e)}")

@router.get("/sct/sessions/{session_id}")
def get_session(
    session_id: str, 
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """특정 세션 정보를 조회합니다."""
    print(f"=== 세션 조회 시작 ===")
    print(f"요청된 session_id: {session_id}")
    print(f"현재 사용자: {current_user.doctor_id}")
    print(f"현재 사용자 타입: {type(current_user.doctor_id)}")
    
    session = crud.get_session_by_id(db, session_id)
    if not session:
        print(f"세션을 찾을 수 없음: {session_id}")
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    print(f"세션 소유자: {session.doctor_id}")
    print(f"세션 소유자 타입: {type(session.doctor_id)}")
    print(f"현재 사용자: {current_user.doctor_id}")
    print(f"현재 사용자 타입: {type(current_user.doctor_id)}")
    print(f"소유자 일치 여부: {session.doctor_id == current_user.doctor_id}")
    print(f"소유자 일치 여부 (문자열 비교): {str(session.doctor_id) == str(current_user.doctor_id)}")
    
    # 세션 소유자 확인만 적용 (승인 상태 확인 제거)
    if session.doctor_id != current_user.doctor_id:
        print(f"세션 소유자 불일치 - 세션 소유자: {session.doctor_id}, 현재 사용자: {current_user.doctor_id}")
        print(f"세션 소유자 repr: {repr(session.doctor_id)}")
        print(f"현재 사용자 repr: {repr(current_user.doctor_id)}")
        raise HTTPException(status_code=403, detail="해당 세션에 대한 접근 권한이 없습니다.")
    
    # SCT 문항 추가
    items = []
    for i, stem in enumerate(SCT_ITEMS, 1):
        items.append({
            "item_no": i,
            "stem": stem,
            "answer": ""  # 초기값은 빈 문자열
        })
    
    # 응답 데이터 구성
    response_data = {
        "session_id": str(session.session_id),
        "patient_name": str(session.patient_name),
        "doctor_id": str(session.doctor_id),
        "status": str(session.status),
        "created_at": session.created_at.isoformat(),
        "submitted_at": session.submitted_at.isoformat() if session.submitted_at else None,
        "expires_at": session.expires_at.isoformat(),
        "items": items,  # SCT 문항 추가
        "responses": session.responses if session.responses else [],  # 기존 응답이 있으면 사용
        "interpretation": session.interpretation
    }
    
    print("=== 세션 조회 응답 데이터 (간소화) ===")
    simplified_response = {
        "session_id": str(session.session_id),
        "patient_name": str(session.patient_name),
        "doctor_id": str(session.doctor_id),
        "status": str(session.status),
        "created_at": session.created_at.isoformat(),
        "submitted_at": session.submitted_at.isoformat() if session.submitted_at else None,
        "expires_at": session.expires_at.isoformat(),
        "items_count": len(items),
        "responses_count": len(session.responses) if session.responses else 0,
        "has_interpretation": bool(session.interpretation)
    }
    print(json.dumps(simplified_response, ensure_ascii=False, indent=2))
    print("=====================")
    
    return response_data

@router.post("/sct/sessions/{session_id}/responses")
def save_responses(
    session_id: str, 
    responses: Dict[str, List[SCTResponseCreate]], 
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """SCT 검사 응답을 임시 저장합니다."""
    try:
        # 사용자 승인 상태 확인
        if not current_user.is_verified:
            raise HTTPException(
                status_code=403, 
                detail="승인되지 않은 계정입니다. 관리자에게 문의하세요."
            )
        
        # 사용자 활성 상태 확인
        if not current_user.is_active:
            raise HTTPException(
                status_code=403, 
                detail="비활성화된 계정입니다. 관리자에게 문의하세요."
            )
        
        # 세션 조회
        session = crud.get_session_by_id(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # 세션 소유자 확인
        if session.doctor_id != current_user.doctor_id:
            raise HTTPException(status_code=403, detail="해당 세션에 대한 접근 권한이 없습니다.")
        
        # 응답 저장 (상태는 변경하지 않음)
        session.responses = [response.dict() for response in responses["responses"]]
        db.commit()
        
        return {"message": "응답이 임시 저장되었습니다", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"응답 저장 중 오류가 발생했습니다: {str(e)}")

@router.post("/sct/sessions/{session_id}/responses/patient")
def save_responses_patient(
    session_id: str, 
    responses: Dict[str, List[SCTResponseCreate]], 
    db: Session = Depends(get_db)
):
    """환자용 SCT 검사 응답 임시 저장 (인증 불필요)"""
    try:
        # 세션 조회
        session = crud.get_session_by_id(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # 완료된 세션은 수정할 수 없음
        if session.status == "complete":
            raise HTTPException(status_code=403, detail="이미 완료된 검사입니다.")
        
        # 응답 저장 (상태는 변경하지 않음)
        session.responses = [response.dict() for response in responses["responses"]]
        db.commit()
        
        return {"message": "응답이 임시 저장되었습니다", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"응답 저장 중 오류가 발생했습니다: {str(e)}")

@router.post("/sct/sessions/{session_id}/complete")
def complete_session(
    session_id: str, 
    responses: Dict[str, List[SCTResponseCreate]], 
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """SCT 검사를 완료 처리합니다."""
    try:
        # 사용자 승인 상태 확인
        if not current_user.is_verified:
            raise HTTPException(
                status_code=403, 
                detail="승인되지 않은 계정입니다. 관리자에게 문의하세요."
            )
        
        # 사용자 활성 상태 확인
        if not current_user.is_active:
            raise HTTPException(
                status_code=403, 
                detail="비활성화된 계정입니다. 관리자에게 문의하세요."
            )
        
        # 세션 조회
        session = crud.get_session_by_id(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # 세션 소유자 확인
        if session.doctor_id != current_user.doctor_id:
            raise HTTPException(status_code=403, detail="해당 세션에 대한 접근 권한이 없습니다.")
        
        # 응답 저장 및 상태 업데이트
        session.responses = [response.dict() for response in responses["responses"]]
        session.status = "complete"
        session.submitted_at = datetime.utcnow()
        db.commit()
        
        return {"message": "검사가 완료되었습니다", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검사 완료 처리 중 오류가 발생했습니다: {str(e)}")

@router.post("/sct/sessions/{session_id}/complete/patient")
def complete_session_patient(
    session_id: str, 
    responses: Dict[str, List[SCTResponseCreate]], 
    db: Session = Depends(get_db)
):
    """환자용 SCT 검사 완료 처리 (인증 불필요)"""
    try:
        # 세션 조회
        session = crud.get_session_by_id(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # 이미 완료된 세션은 다시 완료할 수 없음
        if session.status == "complete":
            raise HTTPException(status_code=403, detail="이미 완료된 검사입니다.")
        
        # 응답 저장 및 상태 업데이트
        session.responses = [response.dict() for response in responses["responses"]]
        session.status = "complete"
        session.submitted_at = datetime.utcnow()
        db.commit()
        
        return {"message": "검사가 완료되었습니다", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검사 완료 처리 중 오류가 발생했습니다: {str(e)}")

@router.get("/sct/sessions/{session_id}/responses")
def get_session_responses(
    session_id: str, 
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """세션의 응답 목록을 조회합니다."""
    try:
        # 세션 조회
        session = crud.get_session_by_id(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # 세션 소유자 확인
        if session.doctor_id != current_user.doctor_id:
            raise HTTPException(status_code=403, detail="해당 세션에 대한 접근 권한이 없습니다.")
        
        if not session.responses:
            return {"responses": []}
            
        # 응답과 문항 정보를 합쳐서 반환
        responses_with_stems = []
        for response in session.responses:
            item_no = response.get("item_no")
            if 1 <= item_no <= len(SCT_ITEMS):
                response["stem"] = SCT_ITEMS[item_no - 1]
            responses_with_stems.append(response)
            
        return {"responses": responses_with_stems}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"응답 조회 중 오류가 발생했습니다: {str(e)}")

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
def list_sessions_by_user(
    user_id: str, 
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """특정 사용자의 세션 목록을 조회합니다."""
    # 본인의 세션만 조회 가능
    if user_id != current_user.doctor_id:
        raise HTTPException(status_code=403, detail="다른 사용자의 세션을 조회할 권한이 없습니다.")
    
    sessions = crud.get_sessions_by_user(db, user_id)
    
    # 세션 목록을 JSON 직렬화 가능한 형태로 변환
    session_list = []
    for session in sessions:
        session_data = {
            "session_id": str(session.session_id),
            "patient_name": str(session.patient_name),
            "doctor_id": str(session.doctor_id),
            "status": str(session.status),
            "created_at": session.created_at.isoformat(),
            "submitted_at": session.submitted_at.isoformat() if session.submitted_at else None,
            "expires_at": session.expires_at.isoformat(),
            "responses": session.responses if session.responses else [],
            "interpretation": session.interpretation
        }
        session_list.append(session_data)
    
    # 간소화된 로그 출력 (응답 데이터와 해석 제외)
    simplified_sessions = []
    for session in session_list:
        simplified_session = {
            "session_id": session["session_id"],
            "patient_name": session["patient_name"],
            "doctor_id": session["doctor_id"],
            "status": session["status"],
            "created_at": session["created_at"],
            "submitted_at": session["submitted_at"],
            "expires_at": session["expires_at"],
            "responses_count": len(session["responses"]) if session["responses"] else 0,
            "has_interpretation": bool(session["interpretation"])
        }
        simplified_sessions.append(simplified_session)
    
    print("=== 세션 목록 조회 응답 데이터 (간소화) ===")
    print(json.dumps({"sessions": simplified_sessions, "total_count": len(session_list)}, ensure_ascii=False, indent=2))
    print("=====================")
    
    return {"sessions": session_list, "total_count": len(session_list)}

@router.delete("/sct/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    """세션을 삭제합니다."""
    try:
        # 세션 조회
        session = crud.get_session_by_id(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
            
        # 권한 확인 (본인의 세션만 삭제 가능)
        if str(session.doctor_id) != str(current_user.doctor_id):
            raise HTTPException(status_code=403, detail="이 세션을 삭제할 권한이 없습니다")
            
        # 세션 삭제
        db.delete(session)
        db.commit()
        
        return {"message": "세션이 성공적으로 삭제되었습니다", "session_id": session_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"세션 삭제 중 오류가 발생했습니다: {str(e)}")

@router.get("/sct/sessions/{session_id}/patient")
def get_session_for_patient(
    session_id: str, 
    db: Session = Depends(get_db)
):
    """환자용 세션 정보 조회 (인증 불필요)"""
    print(f"=== 환자용 세션 조회 시작 ===")
    print(f"요청된 session_id: {session_id}")
    
    session = crud.get_session_by_id(db, session_id)
    if not session:
        print(f"세션을 찾을 수 없음: {session_id}")
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    print(f"세션 상태: {session.status}")
    
    # 완료된 세션은 환자가 접근할 수 없음
    if session.status == "complete":
        print(f"완료된 세션 접근 시도: {session_id}")
        raise HTTPException(status_code=403, detail="이미 완료된 검사입니다.")
    
    # SCT 문항 추가
    items = []
    for i, stem in enumerate(SCT_ITEMS, 1):
        items.append({
            "item_no": i,
            "stem": stem,
            "answer": ""  # 초기값은 빈 문자열
        })
    
    # 응답 데이터 구성 (환자용 - 민감한 정보 제외)
    response_data = {
        "session_id": str(session.session_id),
        "patient_name": str(session.patient_name),
        "status": str(session.status),
        "created_at": session.created_at.isoformat(),
        "expires_at": session.expires_at.isoformat(),
        "items": items,  # SCT 문항 추가
        "responses": session.responses if session.responses else []  # 기존 응답이 있으면 사용
    }
    
    print("=== 환자용 세션 조회 응답 데이터 (간소화) ===")
    simplified_response = {
        "session_id": str(session.session_id),
        "patient_name": str(session.patient_name),
        "status": str(session.status),
        "created_at": session.created_at.isoformat(),
        "expires_at": session.expires_at.isoformat(),
        "items_count": len(items),
        "responses_count": len(session.responses) if session.responses else 0
    }
    print(json.dumps(simplified_response, ensure_ascii=False, indent=2))
    print("=====================")
    
    return response_data 