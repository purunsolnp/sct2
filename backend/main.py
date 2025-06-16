from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, func, extract, and_, or_, create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import hashlib
import jwt
import os
from openai import OpenAI
import json
import uuid
import logging
import pytz

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 한국 시간대 설정
KST = pytz.timezone('Asia/Seoul')

def get_kst_now():
    """현재 한국 시간 반환"""
    return datetime.now(KST)

def to_kst(dt):
    """UTC 시간을 KST로 변환"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KST)

# 데이터베이스 설정
DATABASE_URL = os.getenv("DATABASE_URL")

# Render에서 제공하는 기본 PostgreSQL URL 형식 처리
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# 데이터베이스 엔진 생성
def create_database_engine():
    if not DATABASE_URL:
        logger.info("⚠️ DATABASE_URL이 설정되지 않았습니다. SQLite로 폴백합니다.")
        # SQLite 폴백 (개발용)
        return create_engine(
            "sqlite:///./sct_app.db",
            connect_args={"check_same_thread": False},
            echo=False
        )
    
    # PostgreSQL 연결 설정
    connect_args = {
        "sslmode": "require",
        "connect_timeout": 30,
    }
    
    try:
        engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args=connect_args,
            echo=False
        )
        
        # 연결 테스트
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        logger.info("✅ PostgreSQL 데이터베이스 연결 성공")
        return engine
        
    except Exception as e:
        logger.error(f"❌ PostgreSQL 연결 실패: {e}")
        logger.info("⚠️ SQLite로 폴백합니다.")
        
        # SQLite 폴백
        return create_engine(
            "sqlite:///./sct_app.db",
            connect_args={"check_same_thread": False},
            echo=False
        )

# 데이터베이스 엔진 생성
engine = create_database_engine()

# 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스
Base = declarative_base()

# 의존성 주입용 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 테이블 생성 함수
def create_tables():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ 데이터베이스 테이블 생성 완료")
        return True
    except Exception as e:
        logger.error(f"❌ 테이블 생성 실패: {e}")
        return False

# 헬스체크 함수
def check_database_health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": f"error: {str(e)}"}

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
    created_at = Column(DateTime, default=lambda: get_kst_now().replace(tzinfo=None))

class SCTSession(Base):
    __tablename__ = "sct_sessions"
    
    session_id = Column(String, primary_key=True, index=True)
    doctor_id = Column(String, index=True)
    patient_name = Column(String)
    status = Column(String, default="incomplete")
    created_at = Column(DateTime, default=lambda: get_kst_now().replace(tzinfo=None))
    submitted_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime)

class SCTResponse(Base):
    __tablename__ = "sct_responses"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    item_no = Column(Integer)
    stem = Column(Text)
    answer = Column(Text)
    created_at = Column(DateTime, default=lambda: get_kst_now().replace(tzinfo=None))

class SCTInterpretation(Base):
    __tablename__ = "sct_interpretations"
    
    session_id = Column(String, primary_key=True, index=True)
    interpretation = Column(Text)
    patient_name = Column(String)
    created_at = Column(DateTime, default=lambda: get_kst_now().replace(tzinfo=None))

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
    to_encode.update({"exp": expire.timestamp()})
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

# 관리자 권한 확인 함수
def check_admin_permission(current_user: str):
    """관리자 권한 확인"""
    admin_users = ["admin", "doctor1"]  # 임시 관리자 계정들
    if current_user not in admin_users:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다")

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
    "내가 늘 원하는 것은",
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
        "timestamp": get_kst_now(),
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
        expires_at = get_kst_now() + timedelta(days=7)
        current_time = get_kst_now()
        
        # patient_name 검증 및 정제
        patient_name = session_data.patient_name.strip() if session_data.patient_name else None
        if not patient_name:
            logger.error(f"❌ patient_name이 비어있음: '{session_data.patient_name}'")
            raise HTTPException(status_code=400, detail="환자 이름이 비어있습니다")
        
        db_session = SCTSession(
            session_id=session_id,
            doctor_id=current_user,
            patient_name=patient_name,
            status="incomplete",
            created_at=current_time.replace(tzinfo=None),
            expires_at=expires_at.replace(tzinfo=None)
        )
        
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        
        logger.info(f"✅ 새 세션 생성 완료: {session_id}")
        
        return {
            "session_id": session_id, 
            "patient_name": db_session.patient_name,
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
        current_time = get_kst_now()
        for session in sessions:
            if session.expires_at and session.expires_at < current_time.replace(tzinfo=None) and session.status != "complete":
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
        
        # 월별 통계 (최근 6개월) - KST 기준
        six_months_ago = get_kst_now() - timedelta(days=180)
        
        monthly_stats = db.query(
            extract('year', SCTSession.created_at).label('year'),
            extract('month', SCTSession.created_at).label('month'),
            func.count(SCTSession.session_id).label('count')
        ).filter(
            SCTSession.doctor_id == doctor_id,
            SCTSession.created_at >= six_months_ago.replace(tzinfo=None)
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

# ===== 관리자 기능 =====

# 관리자 대시보드 통계
@app.get("/admin/dashboard/stats")
async def get_admin_dashboard_stats(
    db = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    """관리자 대시보드 통계 정보"""
    try:
        check_admin_permission(current_user)
        
        # 전체 사용자 수
        total_users = db.query(User).count()
        
        # 전체 세션 수 (각 상태별)
        total_sessions = db.query(SCTSession).count()
        completed_sessions = db.query(SCTSession).filter(SCTSession.status == 'complete').count()
        pending_sessions = db.query(SCTSession).filter(SCTSession.status == 'incomplete').count()
        expired_sessions = db.query(SCTSession).filter(SCTSession.status == 'expired').count()
        
        # 이번 달 생성된 세션 수
        now = get_kst_now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        this_month_sessions = db.query(SCTSession).filter(
            SCTSession.created_at >= month_start.replace(tzinfo=None)
        ).count()
        
        # 이번 달 완료된 검사 수
        this_month_completed = db.query(SCTSession).filter(
            and_(
                SCTSession.status == 'complete',
                SCTSession.submitted_at >= month_start.replace(tzinfo=None)
            )
        ).count()
        
        # 활성 사용자 수 (최근 30일 내 세션 생성한 사용자)
        thirty_days_ago = now - timedelta(days=30)
        active_users = db.query(User.doctor_id).join(
            SCTSession, User.doctor_id == SCTSession.doctor_id
        ).filter(
            SCTSession.created_at >= thirty_days_ago.replace(tzinfo=None)
        ).distinct().count()
        
        return {
            "total_users": total_users,
            "total_sessions": total_sessions,
            "completed_sessions": completed_sessions,
            "pending_sessions": pending_sessions,
            "expired_sessions": expired_sessions,
            "this_month_sessions": this_month_sessions,
            "this_month_completed": this_month_completed,
            "active_users": active_users,
            "completion_rate": round((completed_sessions / total_sessions * 100) if total_sessions > 0 else 0, 1)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 관리자 통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류: {str(e)}")

# 전체 사용자 목록 조회
@app.get("/admin/users")
async def get_all_users(
    page: int = 1,
    limit: int = 20,
    search: str = None,
    db = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    """전체 사용자 목록 조회 (관리자용)"""
    try:
        check_admin_permission(current_user)
        
        # 기본 쿼리
        query = db.query(User)
        
        # 검색 필터
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    User.doctor_id.ilike(search_term),
                    User.email.ilike(search_term),
                    User.first_name.ilike(search_term),
                    User.last_name.ilike(search_term),
                    User.hospital.ilike(search_term)
                )
            )
        
        # 전체 개수
        total_count = query.count()
        
        # 페이징
        offset = (page - 1) * limit
        users = query.order_by(User.created_at.desc()).offset(offset).limit(limit).all()
        
        # 각 사용자별 통계 계산
        user_list = []
        for user in users:
            # 최근 30일간 세션 수
            thirty_days_ago = get_kst_now() - timedelta(days=30)
            recent_sessions = db.query(SCTSession).filter(
                and_(
                    SCTSession.doctor_id == user.doctor_id,
                    SCTSession.created_at >= thirty_days_ago.replace(tzinfo=None)
                )
            ).count()
            
            # 전체 세션 수
            total_user_sessions = db.query(SCTSession).filter(
                SCTSession.doctor_id == user.doctor_id
            ).count()
            
            # 완료된 세션 수
            completed_user_sessions = db.query(SCTSession).filter(
                and_(
                    SCTSession.doctor_id == user.doctor_id,
                    SCTSession.status == 'complete'
                )
            ).count()
            
            # 마지막 활동일
            last_session = db.query(SCTSession).filter(
                SCTSession.doctor_id == user.doctor_id
            ).order_by(SCTSession.created_at.desc()).first()
            
            last_activity = last_session.created_at if last_session else user.created_at
            
            user_data = {
                "doctor_id": user.doctor_id,
                "name": f"{user.last_name}{user.first_name}",
                "email": user.email,
                "specialty": user.specialty,
                "hospital": user.hospital,
                "phone": user.phone,
                "medical_license": user.medical_license,
                "is_verified": user.is_verified,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_activity": last_activity.isoformat() if last_activity else None,
                "total_sessions": total_user_sessions,
                "completed_sessions": completed_user_sessions,
                "recent_30days_sessions": recent_sessions,
                "is_active": recent_sessions > 0
            }
            user_list.append(user_data)
        
        total_pages = (total_count + limit - 1) // limit
        
        return {
            "users": user_list,
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
        logger.error(f"❌ 사용자 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"사용자 목록 조회 중 오류: {str(e)}")

# 사용자 계정 활성화/비활성화
@app.patch("/admin/users/{doctor_id}/status")
async def toggle_user_status(
    doctor_id: str,
    is_verified: bool,
    db = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    """사용자 계정 활성화/비활성화"""
    try:
        check_admin_permission(current_user)
        
        user = db.query(User).filter(User.doctor_id == doctor_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
        
        user.is_verified = is_verified
        db.commit()
        
        status_text = "활성화" if is_verified else "비활성화"
        logger.info(f"✅ 사용자 계정 {status_text}: {doctor_id}")
        
        return {
            "message": f"사용자 계정이 {status_text}되었습니다",
            "doctor_id": doctor_id,
            "is_verified": is_verified
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 사용자 상태 변경 오류: {e}")
        raise HTTPException(status_code=500, detail=f"상태 변경 중 오류: {str(e)}")

# 월별 사용 통계
@app.get("/admin/usage-stats")
async def get_usage_statistics(
    months: int = 12,
    db = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    """월별 사용 통계"""
    try:
        check_admin_permission(current_user)
        
        now = get_kst_now()
        stats = []
        
        for i in range(months):
            # 각 달의 시작과 끝
            target_date = now - timedelta(days=30 * i)
            month_start = target_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            if target_date.month == 12:
                next_month_start = target_date.replace(year=target_date.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                next_month_start = target_date.replace(month=target_date.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # 해당 월의 통계
            month_sessions = db.query(SCTSession).filter(
                and_(
                    SCTSession.created_at >= month_start.replace(tzinfo=None),
                    SCTSession.created_at < next_month_start.replace(tzinfo=None)
                )
            ).count()
            
            month_completed = db.query(SCTSession).filter(
                and_(
                    SCTSession.status == 'complete',
                    SCTSession.submitted_at >= month_start.replace(tzinfo=None),
                    SCTSession.submitted_at < next_month_start.replace(tzinfo=None)
                )
            ).count()
            
            # 해당 월에 활동한 사용자 수
            active_users = db.query(User.doctor_id).join(
                SCTSession, User.doctor_id == SCTSession.doctor_id
            ).filter(
                and_(
                    SCTSession.created_at >= month_start.replace(tzinfo=None),
                    SCTSession.created_at < next_month_start.replace(tzinfo=None)
                )
            ).distinct().count()
            
            stats.append({
                "year": target_date.year,
                "month": target_date.month,
                "month_name": target_date.strftime('%Y년 %m월'),
                "total_sessions": month_sessions,
                "completed_sessions": month_completed,
                "active_users": active_users,
                "completion_rate": round((month_completed / month_sessions * 100) if month_sessions > 0 else 0, 1)
            })
        
        # 최신 월부터 정렬
        stats.reverse()
        
        return {
            "monthly_stats": stats,
            "period": f"{months}개월"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 사용 통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류: {str(e)}")

# 시스템 로그 조회 (간단한 버전)
@app.get("/admin/system-logs")
async def get_system_logs(
    page: int = 1,
    limit: int = 50,
    level: str = None,  # info, warning, error
    db = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    """시스템 로그 조회 (기본적인 세션 로그)"""
    try:
        check_admin_permission(current_user)
        
        # 최근 세션 활동을 로그로 표시
        query = db.query(SCTSession).order_by(SCTSession.created_at.desc())
        
        total_count = query.count()
        offset = (page - 1) * limit
        sessions = query.offset(offset).limit(limit).all()
        
        logs = []
        for session in sessions:
            # 세션 생성 로그
            logs.append({
                "timestamp": session.created_at.isoformat() if session.created_at else None,
                "level": "info",
                "action": "session_created",
                "message": f"새 검사 세션 생성: {session.patient_name} (의사: {session.doctor_id})",
                "details": {
                    "session_id": session.session_id,
                    "doctor_id": session.doctor_id,
                    "patient_name": session.patient_name,
                    "status": session.status
                }
            })
            
            # 세션 완료 로그 (완료된 경우)
            if session.status == 'complete' and session.submitted_at:
                logs.append({
                    "timestamp": session.submitted_at.isoformat() if session.submitted_at else None,
                    "level": "info",
                    "action": "session_completed",
                    "message": f"검사 완료: {session.patient_name} (의사: {session.doctor_id})",
                    "details": {
                        "session_id": session.session_id,
                        "doctor_id": session.doctor_id,
                        "patient_name": session.patient_name,
                        "duration": str(session.submitted_at - session.created_at) if session.submitted_at and session.created_at else None
                    }
                })
        
        # 시간순 정렬
        logs.sort(key=lambda x: x['timestamp'] or '', reverse=True)
        
        # 레벨 필터 적용
        if level:
            logs = [log for log in logs if log['level'] == level]
        
        total_pages = (total_count + limit - 1) // limit
        
        return {
            "logs": logs[:limit],  # 페이징 적용
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_count": len(logs),
                "per_page": limit
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 시스템 로그 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"로그 조회 중 오류: {str(e)}")

# 데이터베이스 정리 (관리자용)
@app.post("/admin/cleanup")
async def admin_cleanup_database(
    days_old: int = 30,
    dry_run: bool = True,
    db = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    """데이터베이스 정리 (관리자용)"""
    try:
        check_admin_permission(current_user)
        
        logger.info(f"🧹 데이터베이스 정리 {'시뮬레이션' if dry_run else '실행'}: {days_old}일 이전 데이터")
        
        cleanup_date = get_kst_now() - timedelta(days=days_old)
        
        # 정리 대상 조회
        old_expired_sessions = db.query(SCTSession).filter(
            and_(
                SCTSession.status == "expired",
                SCTSession.created_at < cleanup_date.replace(tzinfo=None)
            )
        ).all()
        
        cleanup_summary = {
            "cleanup_date": cleanup_date.isoformat(),
            "days_old": days_old,
            "dry_run": dry_run,
            "sessions_to_cleanup": len(old_expired_sessions),
            "cleanup_details": []
        }
        
        if not dry_run:
            cleanup_count = 0
            for session in old_expired_sessions:
                session_id = session.session_id
                patient_name = session.patient_name
                
                # 관련 데이터 삭제
                interpretations_deleted = db.query(SCTInterpretation).filter(
                    SCTInterpretation.session_id == session_id
                ).count()
                
                responses_deleted = db.query(SCTResponse).filter(
                    SCTResponse.session_id == session_id
                ).count()
                
                # 실제 삭제
                db.query(SCTInterpretation).filter(
                    SCTInterpretation.session_id == session_id
                ).delete()
                
                db.query(SCTResponse).filter(
                    SCTResponse.session_id == session_id
                ).delete()
                
                db.delete(session)
                cleanup_count += 1
                
                cleanup_summary["cleanup_details"].append({
                    "session_id": session_id,
                    "patient_name": patient_name,
                    "responses_deleted": responses_deleted,
                    "interpretations_deleted": interpretations_deleted
                })
            
            db.commit()
            cleanup_summary["actual_cleaned"] = cleanup_count
            logger.info(f"✅ 정리 완료: {cleanup_count}개 세션 삭제")
        
        else:
            # 시뮬레이션 모드
            for session in old_expired_sessions:
                interpretations_count = db.query(SCTInterpretation).filter(
                    SCTInterpretation.session_id == session.session_id
                ).count()
                
                responses_count = db.query(SCTResponse).filter(
                    SCTResponse.session_id == session.session_id
                ).count()
                
                cleanup_summary["cleanup_details"].append({
                    "session_id": session.session_id,
                    "patient_name": session.patient_name,
                    "responses_to_delete": responses_count,
                    "interpretations_to_delete": interpretations_count
                })
        
        return cleanup_summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 데이터베이스 정리 오류: {e}")
        if not dry_run:
            db.rollback()
        raise HTTPException(status_code=500, detail=f"정리 중 오류: {str(e)}")

# ===== 기존 기능들 계속 =====

@app.get("/sct/session/{session_id}")
async def get_session(session_id: str, db = Depends(get_db)):
    try:
        logger.info(f"🔍 세션 조회 요청: {session_id}")
        
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # 만료 확인
        if session.expires_at and session.expires_at < get_kst_now().replace(tzinfo=None):
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
        
        if session.expires_at and session.expires_at < get_kst_now().replace(tzinfo=None):
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
        
        if session.expires_at and session.expires_at < get_kst_now().replace(tzinfo=None):
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
            session.submitted_at = get_kst_now().replace(tzinfo=None)
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
            "analysis_date": get_kst_now().isoformat()
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
            existing_interpretation.created_at = get_kst_now().replace(tzinfo=None)
        else:
            new_interpretation = SCTInterpretation(
                session_id=session_id,
                interpretation=interpretation,
                patient_name=session.patient_name,
                created_at=get_kst_now().replace(tzinfo=None)
            )
            db.add(new_interpretation)
        
        db.commit()
        
        return {
            "session_id": session_id,
            "interpretation": interpretation,
            "generated_at": get_kst_now().isoformat()
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

async def generate_ai_interpretation(responses: List[SCTResponse], patient_name: str) -> str:
    """자연스럽고 전문적인 SCT 해석을 생성합니다."""
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

    # 자연스럽고 전문적인 프롬프트
    system_prompt = """당신은 25년 경력의 정신건강의학과 전문의이자 임상심리학자입니다. 
SCT 문장완성검사의 전문가로서, 임상에서 실제로 활용 가능한 종합적인 해석을 제공해야 합니다.

## 보고서 작성 원칙
1. **임상적 유용성**: 치료 계획 수립에 실질적으로 도움이 되는 정보 제공
2. **구체성과 근거**: 각 영역별로 대표적 응답을 인용하며 분석
3. **균형적 관점**: 강점과 취약성을 균형있게 제시
4. **실행 가능성**: 구체적이고 현실적인 치료 권고안 제시
5. **전문성**: 임상 용어를 적절히 사용하되 이해하기 쉽게 설명

## 해석 시 주의사항  
- 진단보다는 기능적 평가와 성격 구조 분석에 집중
- 각 영역별로 핵심 응답을 인용하며 근거 제시
- 치료적 개입의 우선순위를 명확히 제시
- 환자의 협력 가능성과 동기 수준 평가 포함
- 예후와 성장 잠재력에 대한 전문적 견해 제시"""

    user_prompt = f"""
환자: {patient_name}
검사일: {get_kst_now().strftime('%Y년 %m월 %d일')}

## 전체 응답 (50문항)
{responses_text}

## 카테고리별 분류
{category_text}

위 SCT 결과를 바탕으로 다음 구조로 **종합적이고 실용적인** 임상 해석 보고서를 작성해주세요:

# SCT (문장완성검사) 임상 해석 보고서

## 1. 검사 개요
- 환자 기본정보 및 검사 협조도
- 응답 특성 및 전반적 인상

## 2. 주요 심리적 특성 분석

### 2.1 가족관계 및 애착 패턴
- 부모에 대한 인식과 가족 역동
- **핵심 응답 2-3개 인용하며 분석**

### 2.2 대인관계 및 사회적 기능
- 친밀감 형성 능력과 대인 신뢰도
- **핵심 응답 2-3개 인용하며 분석**

### 2.3 자아개념 및 정체성
- 자기 인식과 자존감 수준  
- **핵심 응답 2개 인용하며 분석**

### 2.4 정서조절 및 스트레스 대처
- 주요 정서 이슈와 대처 방식
- **핵심 응답 2-3개 인용하며 분석**

### 2.5 성역할 및 이성관계
- 성정체성과 이성에 대한 태도
- **핵심 응답 1-2개 인용**

### 2.6 미래전망 및 목표지향성
- 미래 계획과 동기 수준
- **핵심 응답 1-2개 인용**

### 2.7 과거경험 및 현실적응
- 과거 경험의 영향과 현실 대처능력
- **핵심 응답 1-2개 인용**

## 3. 임상적 평가

### 3.1 주요 방어기제 및 성격특성
- 사용하는 방어기제와 성격 구조

### 3.2 정신병리학적 고려사항
- 관찰되는 증상 및 위험요소 평가

## 4. 치료적 권고사항

### 4.1 우선 개입 영역
- 즉시 다뤄야 할 핵심 이슈

### 4.2 생활관리 및 지원방안
- 일상 개선방안과 사회적 지지체계

## 5. 요약 및 예후
- 핵심 특성 요약
- 치료 예후와 협력 가능성
- 재평가 권고시기
- 환자 강점 및 성장 잠재력

**각 영역별로 구체적 응답을 인용하며, 임상에서 실제로 활용 가능한 전문적이고 종합적인 해석을 제공해주세요.**
"""

    try:
        # GPT-4o-mini 사용
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000,
            temperature=0.2,
        )
        
        interpretation = response.choices[0].message.content
        logger.info(f"✅ 전문적 해석 생성 완료: {len(interpretation)} 문자")
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
- **검사 완료일**: {get_kst_now().strftime('%Y년 %m월 %d일 %H시 %M분')}
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
- 각 문항별 응답을 9개 주요 영역으로 분류하여 종합적으로 분석하시기 바랍니다.
- 필요시 추가적인 심리검사나 임상면담을 고려하십시오.

*본 보고서는 시스템 오류로 인한 임시 보고서입니다.*
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)