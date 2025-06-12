from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, func, extract
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
    version="2.1.0",
    description="문장완성검사(SCT) 자동화 시스템 - 확장 기능 포함"
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
        "message": "SCT 검사 시스템 API v2.1", 
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
        logger.info(f"🔍 입력받은 patient_name: '{session_data.patient_name}' (타입: {type(session_data.patient_name)})")
        
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(days=7)
        current_time = datetime.utcnow()
        
        # patient_name 검증 및 정제
        patient_name = session_data.patient_name.strip() if session_data.patient_name else None
        if not patient_name:
            logger.error(f"❌ patient_name이 비어있음: '{session_data.patient_name}'")
            raise HTTPException(status_code=400, detail="환자 이름이 비어있습니다")
        
        logger.info(f"🔍 정제된 patient_name: '{patient_name}'")
        
        db_session = SCTSession(
            session_id=session_id,
            doctor_id=current_user,
            patient_name=patient_name,  # 정제된 이름 사용
            status="incomplete",
            created_at=current_time,
            expires_at=expires_at
        )
        
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        
        # 저장 후 확인
        logger.info(f"🔍 DB 저장 후 patient_name: '{db_session.patient_name}'")
        logger.info(f"✅ 새 세션 생성 완료: {session_id}")
        
        return {
            "session_id": session_id, 
            "patient_name": db_session.patient_name,  # DB에서 읽은 값 사용
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

# ===== 새로 추가된 기능들 =====

@app.delete("/sct/sessions/{session_id}")
async def delete_session(
    session_id: str, 
    db = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    """세션과 관련된 모든 데이터를 삭제합니다."""
    try:
        logger.info(f"🗑️ 세션 삭제 요청: session_id={session_id}, user={current_user}")
        
        # 세션 조회 및 권한 확인
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            logger.warning(f"❌ 존재하지 않는 세션: {session_id}")
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # 세션 소유자 확인
        if session.doctor_id != current_user:
            logger.warning(f"❌ 권한 없는 삭제 시도: session_id={session_id}, owner={session.doctor_id}, requester={current_user}")
            raise HTTPException(status_code=403, detail="이 세션을 삭제할 권한이 없습니다")
        
        patient_name = session.patient_name
        
        # 관련 데이터 삭제 (순서 중요)
        logger.info(f"🧹 관련 데이터 삭제 시작: {session_id}")
        
        # 1. 해석 결과 삭제
        interpretation_count = db.query(SCTInterpretation).filter(
            SCTInterpretation.session_id == session_id
        ).count()
        
        if interpretation_count > 0:
            db.query(SCTInterpretation).filter(
                SCTInterpretation.session_id == session_id
            ).delete()
            logger.info(f"✅ 해석 결과 삭제: {interpretation_count}개")
        
        # 2. 응답 데이터 삭제
        response_count = db.query(SCTResponse).filter(
            SCTResponse.session_id == session_id
        ).count()
        
        if response_count > 0:
            db.query(SCTResponse).filter(
                SCTResponse.session_id == session_id
            ).delete()
            logger.info(f"✅ 응답 데이터 삭제: {response_count}개")
        
        # 3. 세션 삭제
        db.delete(session)
        db.commit()
        
        logger.info(f"✅ 세션 완전 삭제 완료: {session_id} (환자: {patient_name})")
        
        return {
            "message": "세션이 성공적으로 삭제되었습니다",
            "session_id": session_id,
            "patient_name": patient_name,
            "deleted_responses": response_count,
            "deleted_interpretations": interpretation_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 세션 삭제 오류: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"세션 삭제 중 오류가 발생했습니다: {str(e)}")

@app.get("/sct/sessions/by-user/{doctor_id}/paginated")
async def get_sessions_by_user_paginated(
    doctor_id: str,
    page: int = 1,
    limit: int = 20,
    search: str = None,
    status: str = None,
    date_from: str = None,
    date_to: str = None,
    db = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    """페이징과 필터링을 지원하는 세션 목록 조회"""
    try:
        logger.info(f"📄 페이징된 세션 목록 조회: doctor_id={doctor_id}, page={page}, limit={limit}")
        
        if current_user != doctor_id:
            raise HTTPException(status_code=403, detail="접근 권한이 없습니다")
        
        # 기본 쿼리
        query = db.query(SCTSession).filter(SCTSession.doctor_id == doctor_id)
        
        # 검색 필터 (환자명)
        if search:
            search_term = f"%{search}%"
            query = query.filter(SCTSession.patient_name.ilike(search_term))
        
        # 상태 필터
        if status:
            query = query.filter(SCTSession.status == status)
        
        # 날짜 범위 필터
        if date_from:
            try:
                from_date = datetime.strptime(date_from, "%Y-%m-%d")
                query = query.filter(SCTSession.created_at >= from_date)
            except ValueError:
                logger.warning(f"잘못된 시작 날짜 형식: {date_from}")
        
        if date_to:
            try:
                to_date = datetime.strptime(date_to, "%Y-%m-%d")
                # 하루 끝까지 포함
                to_date = to_date.replace(hour=23, minute=59, second=59)
                query = query.filter(SCTSession.created_at <= to_date)
            except ValueError:
                logger.warning(f"잘못된 종료 날짜 형식: {date_to}")
        
        # 전체 개수 계산
        total_count = query.count()
        
        # 페이징 적용
        offset = (page - 1) * limit
        sessions = query.order_by(SCTSession.created_at.desc()).offset(offset).limit(limit).all()
        
        # 만료된 세션 상태 업데이트
        current_time = datetime.utcnow()
        for session in sessions:
            if session.expires_at < current_time and session.status != "complete":
                session.status = "expired"
        
        db.commit()
        
        # 응답 데이터 구성
        session_list = []
        for session in sessions:
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
        
        total_pages = (total_count + limit - 1) // limit
        
        logger.info(f"✅ 페이징된 세션 목록 반환: {len(session_list)}개 세션 (전체 {total_count}개)")
        
        return {
            "sessions": session_list,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_count": total_count,
                "per_page": limit,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 페이징된 세션 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"세션 목록 조회 중 오류가 발생했습니다: {str(e)}")

@app.get("/sct/sessions/statistics/{doctor_id}")
async def get_session_statistics(
    doctor_id: str,
    db = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    """세션 통계 정보를 제공합니다."""
    try:
        logger.info(f"📊 세션 통계 조회: doctor_id={doctor_id}")
        
        if current_user != doctor_id:
            raise HTTPException(status_code=403, detail="접근 권한이 없습니다")
        
        # 기본 통계
        total_sessions = db.query(SCTSession).filter(SCTSession.doctor_id == doctor_id).count()
        completed_sessions = db.query(SCTSession).filter(
            SCTSession.doctor_id == doctor_id,
            SCTSession.status == "complete"
        ).count()
        pending_sessions = db.query(SCTSession).filter(
            SCTSession.doctor_id == doctor_id,
            SCTSession.status == "incomplete"
        ).count()
        expired_sessions = db.query(SCTSession).filter(
            SCTSession.doctor_id == doctor_id,
            SCTSession.status == "expired"
        ).count()
        
        # 월별 통계 (최근 6개월)
        six_months_ago = datetime.utcnow() - timedelta(days=180)
        
        monthly_stats = db.query(
            extract('year', SCTSession.created_at).label('year'),
            extract('month', SCTSession.created_at).label('month'),
            func.count(SCTSession.session_id).label('count')
        ).filter(
            SCTSession.doctor_id == doctor_id,
            SCTSession.created_at >= six_months_ago
        ).group_by(
            extract('year', SCTSession.created_at),
            extract('month', SCTSession.created_at)
        ).order_by(
            extract('year', SCTSession.created_at),
            extract('month', SCTSession.created_at)
        ).all()
        
        # 완료율 계산
        completion_rate = (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        # 평균 완료 시간 (세션 생성부터 제출까지)
        completed_sessions_with_time = db.query(SCTSession).filter(
            SCTSession.doctor_id == doctor_id,
            SCTSession.status == "complete",
            SCTSession.submitted_at.isnot(None)
        ).all()
        
        avg_completion_hours = 0
        if completed_sessions_with_time:
            total_hours = sum([
                (session.submitted_at - session.created_at).total_seconds() / 3600
                for session in completed_sessions_with_time
            ])
            avg_completion_hours = total_hours / len(completed_sessions_with_time)
        
        statistics = {
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "pending_sessions": pending_sessions,
            "expired_sessions": expired_sessions,
            "completion_rate": round(completion_rate, 1),
            "avg_completion_hours": round(avg_completion_hours, 1),
            "monthly_stats": [
                {
                    "year": int(stat.year),
                    "month": int(stat.month),
                    "count": stat.count
                }
                for stat in monthly_stats
            ]
        }
        
        logger.info(f"✅ 세션 통계 반환: {statistics}")
        return statistics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 세션 통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류가 발생했습니다: {str(e)}")

@app.post("/admin/cleanup")
async def cleanup_expired_sessions(
    days_old: int = 30,
    db = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    """만료된 오래된 세션들을 정리합니다. (관리자용)"""
    try:
        # 관리자 권한 확인 (실제 구현 시 관리자 권한 체크 로직 추가)
        if current_user not in ["admin", "doctor1"]:  # 임시 관리자 계정
            raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다")
        
        logger.info(f"🧹 만료된 세션 정리 시작: {days_old}일 이전")
        
        cleanup_date = datetime.utcnow() - timedelta(days=days_old)
        
        # 오래된 만료 세션 조회
        old_expired_sessions = db.query(SCTSession).filter(
            SCTSession.status == "expired",
            SCTSession.created_at < cleanup_date
        ).all()
        
        cleanup_count = 0
        for session in old_expired_sessions:
            session_id = session.session_id
            
            # 관련 데이터 삭제
            db.query(SCTInterpretation).filter(
                SCTInterpretation.session_id == session_id
            ).delete()
            
            db.query(SCTResponse).filter(
                SCTResponse.session_id == session_id
            ).delete()
            
            db.delete(session)
            cleanup_count += 1
        
        db.commit()
        
        logger.info(f"✅ 정리 완료: {cleanup_count}개 세션 삭제")
        
        return {
            "message": f"{cleanup_count}개의 오래된 만료 세션이 정리되었습니다",
            "cleaned_count": cleanup_count,
            "cleanup_date": cleanup_date.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 세션 정리 오류: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"세션 정리 중 오류가 발생했습니다: {str(e)}")

# ===== 기존 기능들 계속 =====

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

# main.py의 generate_ai_interpretation 함수를 이것으로 교체하세요

async def generate_ai_interpretation(responses: List[SCTResponse], patient_name: str) -> str:
    """적정 길이의 상세한 SCT 해석을 생성합니다."""
    if not openai_client:
        return generate_default_interpretation(responses, patient_name)
    
    # 응답 텍스트 구성
    responses_text = "\n".join([
        f"{resp.item_no}. {resp.stem} → {resp.answer}"
        for resp in responses
    ])
    
    # 카테고리별 응답 분류
    category_mapping = {
        "가족관계": [2, 13, 19, 26, 29, 39, 48, 49, 50],
        "대인관계": [6, 22, 32, 44], 
        "자아개념": [15, 34, 30],
        "정서조절": [5, 21, 40, 43],
        "성_결혼관": [8, 9, 10, 23, 25, 36, 37, 47],
        "미래전망": [4, 16, 18, 28, 41, 42],
        "과거경험": [7, 17, 33, 45],
        "현실적응": [1, 3, 11, 31, 38, 46],
        "성격특성": [12, 14, 20, 24, 27, 35]
    }
    
    # 카테고리별 응답 정리
    category_text = ""
    for category, item_numbers in category_mapping.items():
        items = [f"{resp.item_no}. {resp.stem} → {resp.answer}" 
                for resp in responses if resp.item_no in item_numbers]
        if items:
            category_text += f"\n【{category}】\n" + "\n".join(items) + "\n"

    # 균형잡힌 상세 프롬프트 (5000자 목표)
    system_prompt = """당신은 25년 경력의 정신건강의학과 전문의이자 임상심리학자입니다. 
SCT 문장완성검사의 전문가로서, 실용적이면서도 충분히 상세한 임상 해석을 제공해야 합니다.

## 보고서 작성 원칙
1. **실용성**: 임상에서 즉시 활용 가능한 정보 제공
2. **구체성**: 각 영역별로 대표적 응답 2-3개씩 인용하며 분석
3. **균형성**: 강점과 약점을 균형있게 제시
4. **실행가능성**: 구체적이고 실현 가능한 치료 권고안 제시
5. **적절한 분량**: 5000자 내외의 읽기 편한 길이

## 해석 시 주의사항  
- 진단보다는 기능적 평가에 집중
- 각 영역별로 핵심 응답 인용하며 근거 제시
- 치료적 개입의 우선순위 명확히 제시
- 환자의 협력 가능성과 동기 평가 포함"""

    user_prompt = f"""
환자: {patient_name}
검사일: {datetime.now().strftime('%Y년 %m월 %d일')}

## 전체 응답 (50문항)
{responses_text}

## 카테고리별 분류
{category_text}

위 SCT 결과를 바탕으로 다음 구조로 **5000자 내외**의 실용적인 임상 해석 보고서를 작성해주세요:

# SCT (문장완성검사) 임상 해석 보고서

## 1. 검사 개요 (300자)
- 환자 기본정보 및 검사 협조도
- 응답 특성 및 전반적 인상

## 2. 주요 심리적 특성 분석 (3200자)

### 2.1 가족관계 및 애착 패턴 (550자)
- 부모에 대한 인식과 가족 역동
- **핵심 응답 2-3개 인용하며 분석**

### 2.2 대인관계 및 사회적 기능 (550자)  
- 친밀감 형성 능력과 대인 신뢰도
- **핵심 응답 2-3개 인용하며 분석**

### 2.3 자아개념 및 정체성 (450자)
- 자기 인식과 자존감 수준  
- **핵심 응답 2개 인용하며 분석**

### 2.4 정서조절 및 스트레스 대처 (550자)
- 주요 정서 이슈와 대처 방식
- **핵심 응답 2-3개 인용하며 분석**

### 2.5 성역할 및 이성관계 (350자)
- 성정체성과 이성에 대한 태도
- **핵심 응답 1-2개 인용**

### 2.6 미래전망 및 목표지향성 (350자)
- 미래 계획과 동기 수준
- **핵심 응답 1-2개 인용**

### 2.7 과거경험 및 현실적응 (400자)
- 과거 경험의 영향과 현실 대처능력
- **핵심 응답 1-2개 인용**

## 3. 임상적 평가 (800자)

### 3.1 주요 방어기제 및 성격특성 (400자)
- 사용하는 방어기제와 성격 구조

### 3.2 정신병리학적 고려사항 (400자)  
- 관찰되는 증상 및 위험요소 평가

## 4. 치료적 권고사항 (500자)

### 4.1 우선 개입 영역 (300자)
- 즉시 다뤄야 할 핵심 이슈 3가지

### 4.2 생활관리 및 지원방안 (200자)
- 일상 개선방안과 사회적 지지체계

## 5. 요약 및 예후 (500자)
- 핵심 특성 3-4가지 요약
- 치료 예후와 협력 가능성
- 재평가 권고시기
- 환자 강점 및 성장 잠재력

**전체 분량: 약 5000자 내외로 작성하되, 각 영역별로 구체적 응답을 인용하며 실용적인 정보 제공에 집중해주세요.**
"""

    try:
        # GPT-4o-mini 사용으로 비용 절약
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=6000,  # 5000자 목표 + 여유분
            temperature=0.2,
        )
        
        interpretation = response.choices[0].message.content
        logger.info(f"✅ 적정 길이 해석 생성 완료: {len(interpretation)} 문자")
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