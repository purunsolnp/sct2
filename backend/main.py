from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional, List
import hashlib
import jwt
import os
from openai import OpenAI
import json
import uuid
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 개선된 데이터베이스 설정 import
from database_config import engine, SessionLocal, Base, get_db, create_tables, check_database_health

# 환경 변수
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 (안전한 초기화)
openai_client = None
if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("✅ OpenAI 클라이언트 초기화 성공")
    except Exception as e:
        logger.error(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
        openai_client = None
else:
    logger.warning("⚠️ OpenAI API 키가 설정되지 않았습니다. 해석 기능이 제한됩니다.")

# FastAPI 앱 초기화
app = FastAPI(
    title="SCT 검사 시스템 API", 
    version="2.0.0",
    description="문장완성검사(SCT) 자동화 시스템"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영환경에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT 보안
security = HTTPBearer()

# 데이터베이스 모델
class User(Base):
    __tablename__ = "users"
    
    doctor_id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    first_name = Column(String)
    last_name = Column(String)
    specialty = Column(String)
    hospital = Column(String)
    phone = Column(String, nullable=True)
    medical_license = Column(String, nullable=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class SCTSession(Base):
    __tablename__ = "sct_sessions"
    
    session_id = Column(String, primary_key=True, index=True)
    doctor_id = Column(String, index=True)
    patient_name = Column(String)
    status = Column(String, default="incomplete")
    created_at = Column(DateTime, default=datetime.utcnow)
    submitted_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime)

class SCTResponse(Base):
    __tablename__ = "sct_responses"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    item_no = Column(Integer)
    stem = Column(Text)
    answer = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class SCTInterpretation(Base):
    __tablename__ = "sct_interpretations"
    
    session_id = Column(String, primary_key=True, index=True)
    interpretation = Column(Text)
    patient_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic 모델
class UserCreate(BaseModel):
    doctor_id: str
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    specialty: str
    hospital: str
    phone: Optional[str] = None
    medical_license: Optional[str] = None

class UserLogin(BaseModel):
    doctor_id: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_info: dict

class SessionCreate(BaseModel):
    patient_name: str
    assigned_by: str

class SCTResponseCreate(BaseModel):
    item_no: int
    stem: str
    answer: str

# 유틸리티 함수
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed_password: str) -> bool:
    return hash_password(password) == hashed_password

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        doctor_id: str = payload.get("sub")
        if doctor_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return doctor_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# SCT 검사 문항 (50개)
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

# 애플리케이션 시작 시 테이블 생성
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 애플리케이션 시작")
    create_tables()
    
    # 데이터베이스 연결 확인
    health = check_database_health()
    if health["status"] == "healthy":
        logger.info("✅ 데이터베이스 연결 확인됨")
    else:
        logger.warning(f"⚠️ 데이터베이스 연결 문제: {health}")

# API 엔드포인트
@app.get("/")
async def root():
    return {
        "message": "SCT 검사 시스템 API v2.0", 
        "status": "running",
        "database": check_database_health()["status"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "database": check_database_health(),
        "openai": "available" if openai_client else "unavailable"
    }

@app.post("/auth/register")
async def register(user: UserCreate, db = Depends(get_db)):
    try:
        logger.info(f"🏥 회원가입 시도: {user.doctor_id}")
        
        # 기존 사용자 확인
        existing_user = db.query(User).filter(
            (User.doctor_id == user.doctor_id) | (User.email == user.email)
        ).first()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="이미 존재하는 ID 또는 이메일입니다")
        
        # 새 사용자 생성
        db_user = User(
            doctor_id=user.doctor_id,
            email=user.email,
            hashed_password=hash_password(user.password),
            first_name=user.first_name,
            last_name=user.last_name,
            specialty=user.specialty,
            hospital=user.hospital,
            phone=user.phone,
            medical_license=user.medical_license
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"✅ 새 사용자 등록: {user.doctor_id}")
        return {"message": "회원가입이 완료되었습니다"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 회원가입 오류: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="회원가입 중 오류가 발생했습니다")

@app.post("/auth/login", response_model=TokenResponse)
async def login(user_login: UserLogin, db = Depends(get_db)):
    try:
        logger.info(f"🔐 로그인 시도: {user_login.doctor_id}")
        
        user = db.query(User).filter(User.doctor_id == user_login.doctor_id).first()
        
        if not user or not verify_password(user_login.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="잘못된 ID 또는 비밀번호입니다")
        
        access_token = create_access_token(data={"sub": user.doctor_id})
        
        logger.info(f"✅ 로그인 성공: {user.doctor_id}")
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_info": {
                "doctor_id": user.doctor_id,
                "name": f"{user.last_name}{user.first_name}",
                "email": user.email,
                "specialty": user.specialty,
                "hospital": user.hospital
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 로그인 오류: {e}")
        raise HTTPException(status_code=500, detail="로그인 중 오류가 발생했습니다")

@app.get("/auth/check-id/{doctor_id}")
async def check_doctor_id(doctor_id: str, db = Depends(get_db)):
    try:
        user = db.query(User).filter(User.doctor_id == doctor_id).first()
        return {"available": user is None}
    except Exception as e:
        logger.error(f"❌ ID 확인 오류: {e}")
        raise HTTPException(status_code=500, detail="ID 확인 중 오류가 발생했습니다")

@app.post("/sct/sessions")
async def create_session(
    session_data: SessionCreate, 
    db = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    try:
        logger.info(f"🏗️ 새 세션 생성 요청: patient={session_data.patient_name}, doctor={current_user}")
        
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(days=7)
        current_time = datetime.utcnow()
        
        db_session = SCTSession(
            session_id=session_id,
            doctor_id=current_user,
            patient_name=session_data.patient_name,
            status="incomplete",
            created_at=current_time,
            expires_at=expires_at
        )
        
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        
        logger.info(f"✅ 새 세션 생성 완료: {session_id}")
        
        return {
            "session_id": session_id, 
            "patient_name": session_data.patient_name,
            "doctor_id": current_user,
            "status": "incomplete",
            "created_at": current_time.isoformat(),
            "expires_at": expires_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ 세션 생성 오류: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"세션 생성 중 오류가 발생했습니다: {str(e)}")

@app.get("/sct/sessions/by-user/{doctor_id}")
async def get_sessions_by_user(
    doctor_id: str, 
    db = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    try:
        logger.info(f"🔍 세션 목록 조회 요청: doctor_id={doctor_id}, current_user={current_user}")
        
        if current_user != doctor_id:
            raise HTTPException(status_code=403, detail="접근 권한이 없습니다")
        
        # SCTSession 테이블에서 세션 목록 조회
        sessions = db.query(SCTSession).filter(
            SCTSession.doctor_id == doctor_id
        ).order_by(SCTSession.created_at.desc()).all()
        
        logger.info(f"📊 조회된 세션 수: {len(sessions)}")
        
        # 만료된 세션 상태 업데이트
        current_time = datetime.utcnow()
        for session in sessions:
            if session.expires_at < current_time and session.status != "complete":
                session.status = "expired"
                logger.info(f"⏰ 세션 만료 처리: {session.session_id}")
        
        db.commit()
        
        # 응답 데이터 구성
        session_list = []
        for session in sessions:
            # 각 세션의 응답 개수 확인
            response_count = db.query(SCTResponse).filter(
                SCTResponse.session_id == session.session_id
            ).count()
            
            session_data = {
                "session_id": session.session_id,
                "doctor_id": session.doctor_id,
                "patient_name": session.patient_name,
                "status": session.status,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "submitted_at": session.submitted_at.isoformat() if session.submitted_at else None,
                "expires_at": session.expires_at.isoformat() if session.expires_at else None,
                "response_count": response_count
            }
            session_list.append(session_data)
            
        logger.info(f"✅ 세션 목록 반환: {len(session_list)}개 세션")
        
        return {"sessions": session_list, "total_count": len(session_list)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 세션 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"세션 목록 조회 중 오류가 발생했습니다: {str(e)}")

@app.get("/sct/session/{session_id}")
async def get_session(session_id: str, db = Depends(get_db)):
    try:
        logger.info(f"🔍 세션 조회 요청: {session_id}")
        
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # 만료 확인
        if session.expires_at < datetime.utcnow():
            session.status = "expired"
            db.commit()
            raise HTTPException(status_code=410, detail="만료된 세션입니다")
        
        # 응답 목록 조회
        responses = db.query(SCTResponse).filter(
            SCTResponse.session_id == session_id
        ).order_by(SCTResponse.item_no).all()
        
        response_data = []
        for resp in responses:
            response_data.append({
                "item_no": resp.item_no,
                "stem": resp.stem,
                "answer": resp.answer
            })
        
        return {
            "session_id": session.session_id,
            "doctor_id": session.doctor_id,
            "patient_name": session.patient_name,
            "status": session.status,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "submitted_at": session.submitted_at.isoformat() if session.submitted_at else None,
            "expires_at": session.expires_at.isoformat() if session.expires_at else None,
            "responses": response_data,
            "total_items": len(SCT_ITEMS),
            "completed_items": len(response_data)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 세션 조회 오류: {e}")
        raise HTTPException(status_code=500, detail="세션 조회 중 오류가 발생했습니다")

@app.get("/sct/session/{session_id}/items")
async def get_session_items(session_id: str, db = Depends(get_db)):
    try:
        logger.info(f"📋 세션 문항 조회: {session_id}")
        
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        if session.expires_at < datetime.utcnow():
            session.status = "expired"
            db.commit()
            raise HTTPException(status_code=410, detail="만료된 세션입니다")
        
        # 기존 응답 가져오기
        existing_responses = db.query(SCTResponse).filter(
            SCTResponse.session_id == session_id
        ).all()
        existing_dict = {resp.item_no: resp.answer for resp in existing_responses}
        
        # 문항 목록 생성
        items = []
        for i, stem in enumerate(SCT_ITEMS, 1):
            items.append({
                "item_no": i,
                "stem": stem,
                "answer": existing_dict.get(i, "")
            })
        
        return {
            "session_id": session_id,
            "patient_name": session.patient_name,
            "items": items,
            "status": session.status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 세션 문항 조회 오류: {e}")
        raise HTTPException(status_code=500, detail="세션 문항 조회 중 오류가 발생했습니다")

@app.post("/sct/session/{session_id}/responses")
async def save_responses(
    session_id: str, 
    responses: List[SCTResponseCreate], 
    db = Depends(get_db)
):
    try:
        logger.info(f"💾 응답 저장 요청: session={session_id}, responses={len(responses)}")
        
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        if session.expires_at < datetime.utcnow():
            raise HTTPException(status_code=410, detail="만료된 세션입니다")
        
        # 기존 응답 삭제
        db.query(SCTResponse).filter(SCTResponse.session_id == session_id).delete()
        
        # 새 응답 저장
        saved_count = 0
        for response in responses:
            if response.answer and response.answer.strip():
                db_response = SCTResponse(
                    session_id=session_id,
                    item_no=response.item_no,
                    stem=response.stem,
                    answer=response.answer.strip()
                )
                db.add(db_response)
                saved_count += 1
        
        # 모든 문항에 답변이 있으면 완료 상태로 변경
        if saved_count >= 45:  # 최소 45개 이상 답변 시 완료로 간주
            session.status = "complete"
            session.submitted_at = datetime.utcnow()
            logger.info(f"✅ 세션 완료 처리: {session_id}")
        
        db.commit()
        
        logger.info(f"✅ 응답 저장 완료: {saved_count}개 응답")
        return {"message": "응답이 저장되었습니다", "saved_count": saved_count}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 응답 저장 오류: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="응답 저장 중 오류가 발생했습니다")

@app.get("/sct/sessions/{session_id}/analysis")
async def get_categorical_analysis(session_id: str, db = Depends(get_db)):
    """카테고리별 응답 분석을 제공합니다."""
    try:
        logger.info(f"📊 카테고리 분석 요청: {session_id}")
        
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        if session.status != "complete":
            raise HTTPException(status_code=400, detail="완료된 검사만 분석 가능합니다")
        
        # 응답 목록 조회
        responses = db.query(SCTResponse).filter(
            SCTResponse.session_id == session_id
        ).order_by(SCTResponse.item_no).all()
        
        # 카테고리별 응답 분류
        categorized_responses = {}
        for category, item_numbers in SCT_ITEM_CATEGORIES.items():
            categorized_responses[category] = []
            for item_no in item_numbers:
                for response in responses:
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
            "analysis_date": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 카테고리 분석 오류: {e}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {str(e)}")

@app.post("/sct/sessions/{session_id}/interpret")
async def generate_interpretation_endpoint(session_id: str, db = Depends(get_db)):
    """SCT 해석을 생성합니다."""
    try:
        logger.info(f"🧠 해석 생성 요청: {session_id}")
        
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        if session.status != "complete":
            raise HTTPException(status_code=400, detail="완료된 검사만 해석 가능합니다")
        
        # 응답 목록 조회
        responses = db.query(SCTResponse).filter(
            SCTResponse.session_id == session_id
        ).order_by(SCTResponse.item_no).all()
        
        if not responses:
            raise HTTPException(status_code=400, detail="응답이 없습니다")
        
        # AI 해석 생성
        interpretation = await generate_ai_interpretation(responses, session.patient_name)
        
        # 해석 결과 저장
        existing_interpretation = db.query(SCTInterpretation).filter(
            SCTInterpretation.session_id == session_id
        ).first()
        
        if existing_interpretation:
            existing_interpretation.interpretation = interpretation
            existing_interpretation.created_at = datetime.utcnow()
        else:
            new_interpretation = SCTInterpretation(
                session_id=session_id,
                interpretation=interpretation,
                patient_name=session.patient_name,
                created_at=datetime.utcnow()
            )
            db.add(new_interpretation)
        
        db.commit()
        
        return {
            "session_id": session_id,
            "interpretation": interpretation,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 해석 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"해석 생성 중 오류: {str(e)}")

@app.get("/sct/sessions/{session_id}/interpretation")
async def get_interpretation_endpoint(session_id: str, db = Depends(get_db)):
    """저장된 해석을 조회합니다."""
    try:
        logger.info(f"📖 해석 조회 요청: {session_id}")
        
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # 저장된 해석 조회
        interpretation_record = db.query(SCTInterpretation).filter(
            SCTInterpretation.session_id == session_id
        ).first()
        
        if not interpretation_record:
            raise HTTPException(status_code=404, detail="해석이 아직 생성되지 않았습니다")
        
        return {
            "session_id": session_id,
            "patient_name": session.patient_name,
            "interpretation": interpretation_record.interpretation,
            "submitted_at": session.submitted_at.isoformat() if session.submitted_at else None,
            "created_at": interpretation_record.created_at.isoformat() if interpretation_record.created_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 해석 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"해석 조회 중 오류: {str(e)}")

# AI 해석 생성 함수
async def generate_ai_interpretation(responses: List[SCTResponse], patient_name: str) -> str:
    """AI를 사용하여 SCT 해석을 생성합니다."""
    if not openai_client:
        return generate_default_interpretation(responses, patient_name)
    
    # 응답 텍스트 구성
    responses_text = "\n".join([
        f"{resp.item_no}. {resp.stem} → {resp.answer}"
        for resp in responses
    ])
    
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
다음은 {patient_name} 환자의 문장완성검사(SCT) 결과입니다.

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
    
    try:
        # OpenAI API 호출
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000,
            temperature=0.3,  # 일관성을 위해 낮은 temperature 사용
        )
        
        interpretation = response.choices[0].message.content
        logger.info(f"✅ AI 해석 생성 완료: {len(interpretation)} 문자")
        return interpretation
        
    except Exception as e:
        logger.error(f"❌ OpenAI API 오류: {e}")
        return generate_default_interpretation(responses, patient_name)

def generate_default_interpretation(responses: List[SCTResponse], patient_name: str) -> str:
    """OpenAI API를 사용할 수 없을 때의 기본 해석"""
    return f"""
# SCT (문장완성검사) 해석 보고서

## 1. 검사 개요
- **환자명**: {patient_name}
- **검사 완료일**: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')}
- **검사 협조도**: 총 {len(responses)}개 문항 완료

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

### 응답 특성 요약
- 총 응답 문항: {len(responses)}개
- 평균 응답 길이: {sum(len(r.answer) for r in responses) // len(responses) if responses else 0}자

## 3. 권고사항
- 전문 임상심리학자 또는 정신건강의학과 전문의의 직접 해석이 필요합니다.
- 각 문항별 응답을 14개 주요 영역으로 분류하여 종합적으로 분석하시기 바랍니다.
- 필요시 추가적인 심리검사나 임상면담을 고려하십시오.

*본 보고서는 시스템 오류로 인한 임시 보고서입니다.*
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)