from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, func, extract, and_, or_, create_engine, text, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
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

# Import database configuration
from database_config import engine, SessionLocal, Base, get_db, create_tables, check_database_health

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
security = HTTPBearer()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        doctor_id: str = payload.get("sub")
        if doctor_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return doctor_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

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
    title="SCT API",
    description="SCT 검사 시스템을 위한 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500", "https://purunsolsct.netlify.app"],  # 프론트엔드 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT 보안
security = HTTPBearer()

# 데이터베이스 모델
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    first_name = Column(String)
    last_name = Column(String)
    specialty = Column(String)
    hospital = Column(String)
    phone = Column(String, nullable=True)
    medical_license = Column(String, nullable=True)
    is_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)  # 관리자 권한 필드 추가
    created_at = Column(DateTime, default=datetime.utcnow)
    last_password_change = Column(DateTime, default=datetime.utcnow)
    password_history = Column(JSON, default=list)
    login_attempts = Column(Integer, default=0)
    last_login_attempt = Column(DateTime)
    last_login = Column(DateTime)
    is_locked = Column(Boolean, default=False)
    lock_until = Column(DateTime)

# Password policy constants
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128
PASSWORD_HISTORY_SIZE = 5
MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_DURATION = 30  # minutes

def validate_password(password: str) -> bool:
    """Validate password against security policy"""
    if len(password) < PASSWORD_MIN_LENGTH:
        return False
    if len(password) > PASSWORD_MAX_LENGTH:
        return False
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(not c.isalnum() for c in password)
    
    return has_upper and has_lower and has_digit and has_special

def check_password_history(user: User, new_password: str) -> bool:
    """Check if password was used in recent history"""
    if not user.password_history:
        return True
    
    for old_password in user.password_history:
        if verify_password(new_password, old_password):
            return False
    return True

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

class GPTTokenUsage(Base):
    __tablename__ = "gpt_token_usage"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    doctor_id = Column(String, index=True)
    session_id = Column(String, index=True)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    model = Column(String)  # gpt-4, gpt-3.5-turbo 등
    cost = Column(Float)  # USD 기준
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

class UserStatusUpdate(BaseModel):
    is_verified: bool
    is_active: bool = None  # 선택적 필드로 변경

# 유틸리티 함수
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed_password: str) -> bool:
    # 임시: sha256 해시 또는 평문 모두 허용 (마이그레이션 후 반드시 원복!)
    if hash_password(password) == hashed_password:
        return True
    if password == hashed_password:
        return True
    return False

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
def check_admin_permission(current_user: str, db: Session):
    """관리자 권한 확인 - 데이터베이스 기반"""
    user = db.query(User).filter(User.doctor_id == current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
    
    # 특정 사용자들을 관리자로 지정 (임시)
    admin_users = ["admin", "doctor1", "purunsolnp"]  # purunsolnp 추가
    if current_user not in admin_users and not getattr(user, 'is_admin', False):
        raise HTTPException(status_code=403, detail="관리자 권한이 없습니다. 관리자에게 문의하세요.")
    
    return user

def check_user_permission(current_user: str, db: Session, action: str = "access"):
    """사용자 권한을 확인하는 통합 함수"""
    user = db.query(User).filter(User.doctor_id == current_user).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
    
    if not user.is_active:
        raise HTTPException(status_code=403, detail="비활성화된 계정입니다. 관리자에게 문의하세요.")
    
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="승인되지 않은 계정입니다. 관리자의 승인을 기다려주세요.")
    
    return user

@app.post("/sct/sessions")
async def create_session(
    session_data: SessionCreate, 
    db = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    try:
        logger.info(f"🏗️ 새 세션 생성 요청: patient={session_data.patient_name}, doctor={current_user}")
        
        # 통합된 권한 확인
        check_user_permission(current_user, db)
        
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
    current_user: str = Depends(get_current_user)
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
    current_user: str = Depends(get_current_user)
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
    current_user: str = Depends(get_current_user)
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
    current_user: str = Depends(get_current_user)
):
    """관리자 대시보드 통계 정보"""
    try:
        check_admin_permission(current_user, db)
        
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
    current_user: str = Depends(get_current_user)
):
    """전체 사용자 목록 조회 (관리자용)"""
    try:
        check_admin_permission(current_user, db)
        
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
    status_update: UserStatusUpdate,
    db = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """사용자 계정 활성화/비활성화"""
    try:
        check_admin_permission(current_user, db)
        user = db.query(User).filter(User.doctor_id == doctor_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
        
        # is_verified와 is_active 모두 업데이트
        user.is_verified = status_update.is_verified
        user.is_active = status_update.is_verified  # 인증 상태와 동일하게 설정
        
        db.commit()
        status_text = "활성화" if status_update.is_verified else "비활성화"
        logger.info(f"✅ 사용자 계정 {status_text}: {doctor_id}")
        return {
            "message": f"사용자 계정이 {status_text}되었습니다",
            "doctor_id": doctor_id,
            "is_verified": status_update.is_verified,
            "is_active": user.is_active
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
    doctor_id: str = None,  # 추가된 파라미터
    db = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """월별 사용 통계"""
    try:
        check_admin_permission(current_user, db)
        
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
            
            # 기본 필터 조건
            base_filter = [
                SCTSession.created_at >= month_start.replace(tzinfo=None),
                SCTSession.created_at < next_month_start.replace(tzinfo=None)
            ]
            
            # doctor_id가 있으면 해당 의사만 필터링
            if doctor_id:
                base_filter.append(SCTSession.doctor_id == doctor_id)
            
            # 해당 월의 통계
            month_sessions = db.query(SCTSession).filter(
                and_(*base_filter)
            ).count()
            
            # 완료된 세션 수
            completed_filter = base_filter + [SCTSession.status == 'complete']
            month_completed = db.query(SCTSession).filter(
                and_(*completed_filter)
            ).count()
            
            # 해당 월에 활동한 사용자 수 (doctor_id가 있을 때는 1)
            if doctor_id:
                active_users = 1 if month_sessions > 0 else 0
            else:
                active_users = db.query(User.doctor_id).join(
                    SCTSession, User.doctor_id == SCTSession.doctor_id
                ).filter(
                    and_(*base_filter)
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
            "period": f"{months}개월",
            "doctor_id": doctor_id  # 응답에 doctor_id 포함
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
    current_user: str = Depends(get_current_user)
):
    """시스템 로그 조회 (기본적인 세션 로그)"""
    try:
        check_admin_permission(current_user, db)
        
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
    current_user: str = Depends(get_current_user)
):
    """데이터베이스 정리 (관리자용)"""
    try:
        check_admin_permission(current_user, db)
        
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
async def generate_interpretation_endpoint(session_id: str, db = Depends(get_db), current_user: str = Depends(get_current_user)):
    """SCT 해석을 생성합니다."""
    try:
        logger.info(f"🧠 해석 생성 요청: {session_id}")
        
        # 통합된 권한 확인
        check_user_permission(current_user, db)
        
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # 세션 소유권 확인
        if session.doctor_id != current_user:
            raise HTTPException(status_code=403, detail="해당 세션에 대한 접근 권한이 없습니다")
        
        if session.status != "complete":
            raise HTTPException(status_code=400, detail="완료된 검사만 해석 가능합니다")
        
        # 응답 목록 조회
        responses = db.query(SCTResponse).filter(
            SCTResponse.session_id == session_id
        ).order_by(SCTResponse.item_no).all()
        
        if not responses:
            raise HTTPException(status_code=400, detail="응답이 없습니다")
        
        # AI 해석 생성
        interpretation = await generate_ai_interpretation(responses, session.patient_name, session.doctor_id, session.session_id, db)
        
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

async def generate_ai_interpretation(responses: List[SCTResponse], patient_name: str, doctor_id: str, session_id: str, db) -> str:
    """OpenAI API를 사용하여 SCT 응답을 해석합니다."""
    if not openai_client:
        logger.warning("⚠️ OpenAI 클라이언트가 초기화되지 않았습니다. 기본 해석을 반환합니다.")
        return generate_default_interpretation(responses, patient_name)
    try:
        # 검사일(제출일) 정보 추출
        exam_date = None
        if responses and hasattr(responses[0], 'created_at') and responses[0].created_at:
            exam_date = responses[0].created_at.strftime('%Y-%m-%d')
        else:
            exam_date = get_kst_now().strftime('%Y-%m-%d')

        # 임상적으로 풍부한 프롬프트 구성 (각 항목 200자 이상, 마크다운 헤더 금지, 환자 지칭 방식 명확화)
        prompt = f"""
당신은 숙련된 임상심리사입니다. 아래의 SCT(문장완성검사) 응답을 바탕으로 임상 해석 보고서를 작성하세요.

[중요 지침]
- 마크다운 헤더(`#`, `##`, `###`, `####` 등)는 절대 사용하지 마세요.
- 각 섹션 제목은 반드시 숫자와 점으로 시작하고, 볼드(굵은 글씨)는 `**`로 감싸서 작성하세요. (예: **1. 검사 개요**)
- 환자를 지칭할 때는 반드시 '{patient_name}님'처럼 이름 뒤에 '님'을 붙여서만 지칭하세요. '그', '그녀', '환자', '내담자' 등으로 지칭하지 마세요.
- 줄바꿈(`\\n\\n`)과 볼드(`**`)만 사용하세요. 리스트, 표, 헤더, 이탤릭 등 다른 마크다운 문법은 사용하지 마세요.
- 각 번호(1, 2, 3...)는 반드시 한 줄 띄우고, 볼드(굵은 글씨)로 작성하세요.
- 각 소제목(2.1, 2.2 등)도 볼드로 감싸세요.
- 각 항목은 반드시 400자 이상으로 작성하세요. 분석은 진단 가능성과 방어기제 수준, 대인 기능, 자아 강도, 성격특성 등을 포함하여 실제 임상에 쓸 수 있을 정도의 깊이로 기술하세요.
- 단순한 문장 요약이 아니라, 해당 응답이 어떤 방어기제(defense), 성격 특성(personality trait), 임상증상(psychopathology)과 연결되는지를 명확히 해석해 주세요.
- 가능한 경우 DSM-5 진단 기준, Vaillant 방어기제 분류, 성격 스펙트럼 개념과 연계해 주세요.
- 응답의 진실성(진단 신뢰도)이나 과장/저반응 가능성에 대한 평가도 포함해 주세요.
- 보고서 말미에는 치료 예후, 치료적 제휴 가능성, 강점과 취약성을 구분해서 서술하고, 재평가 시점과 그 이유도 포함하세요.

다음의 보고서 구조를 반드시 따르세요:


**1. 검사 개요**  
{patient_name}님, 검사일, 검사 협조도, 응답의 전반적 특성, 응답 스타일, 검사 신뢰도 등을 요약해 주세요. 정서의 깊이, 문장 구조의 성실성, 회피나 과장 여부 등도 평가해 주세요.

**2. 주요 심리적 특성 분석**  
**2.1 가족관계 및 애착 패턴**  
**2.2 대인관계 및 사회적 기능**  
**2.3 자아개념 및 정체성**  
**2.4 정서조절 및 스트레스 대처**  
**2.5 성역할 및 이성관계**  
**2.6 미래전망 및 목표지향성**  
**2.7 과거경험 및 현실적응**  
각 항목별로 {patient_name}님의 실제 응답을 구체적으로 인용하며, 임상적으로 해석해 주세요.

**3. 임상적 평가**  
**3.1 주요 방어기제 및 성격특성**  
Vaillant의 방어기제 분류 체계를 기반으로 주요 방어기제를 평가하고, 성격 구조 및 적응 수준을 분석해 주세요.

**3.2 정신병리학적 고려사항**  
우울, 불안, 자기애, 충동성, 관계 회피, 현실 검증 등 심리적 증상 및 기능 저하와 관련된 병리적 요소를 서술해 주세요. DSM-5 기준과 연결 가능하다면 명시적으로 진단 가설을 제시해 주세요.

**4. 치료적 권고사항**  
**4.1 우선 개입 영역**  
정서조절, 관계 문제, 자아통합 등 임상적 개입 우선순위를 제시해 주세요. 치료 목표는 단기-중기-장기로 구분하여 구체화해 주세요.

**4.2 생활관리 및 지원방안**  
일상생활에서 실천 가능한 정서 안정 전략, 사회적 지원, 자기 구조화 기술 등 생활 차원의 개입을 제안해 주세요.

**5. 종합 해석 및 예후**  
{patient_name}님의 응답 전반에서 드러난 심리 구조, 핵심 정서, 반복되는 심리 주제, 방어기제 수준, 성격 특성 간의 연결성을 종합적으로 통합하여 서술해 주세요.  
이 항목은 단순 요약이 아닌 전체 구조적 해석의 핵심입니다.

**5.1 심리적 강점**  
{patient_name}님의 치료에 긍정적으로 작용할 수 있는 자원, 자기 통찰, 회복 탄력성 등을 구체적으로 기술해 주세요.

**5.2 심리적 취약성**  
{patient_name}님의 정서적 취약 영역, 반복되는 갈등 패턴, 방어 실패 지점, 사회적 적응의 제약 요인을 명확히 제시해 주세요.

**5.3 치료적 제휴 형성 가능성**  
치료자와의 관계 형성 가능성을 예측하고, 관계 유지를 위한 고려 요소를 평가해 주세요.

**5.4 재평가 및 추적 관찰 권고**  
재검 권장 시점(예: 초기 개입 3-6개월 후)과 그 이유(예: 정서 조절 변화, 대인 기능 향상 여부 등)를 구체적으로 기술해 주세요.

아래는 {patient_name}님의 실제 응답입니다.
---
"""
        for response in responses:
            prompt += f"\n{response.item_no}. {response.stem} → {response.answer}"
        prompt += """
---
보고서는 반드시 위의 지침을 엄격히 지키고, 임상적으로 깊이 있게, 실제 임상 보고서처럼 작성하세요. 말투는 존댓말로 작성해주세요.  
각 항목별로 소제목을 붙이고, {patient_name}님의 실제 응답을 인용해 해석해 주세요.  
불필요한 반복이나 단순 요약은 피하고, 임상적 통찰과 치료적 제안을 충분히 포함하세요.
"""

        # API 호출 (gpt-4o로 고정)
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 임상심리학 전문가입니다. SCT 응답을 분석하여 전문적이고 객관적인 해석을 제공해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        # 토큰 사용량 기록
        usage = response.usage
        model = "gpt-4o"
        cost = calculate_gpt_cost(model, usage.prompt_tokens, usage.completion_tokens)
        
        token_usage = GPTTokenUsage(
            doctor_id=doctor_id,
            session_id=session_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model=model,
            cost=cost
        )
        db.add(token_usage)
        db.commit()
        
        logger.info(f"✅ GPT 해석 생성 완료: {usage.total_tokens} 토큰 사용 (${cost})")
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"❌ GPT 해석 생성 실패: {e}")
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

@app.get("/admin/gpt-usage")
async def get_gpt_usage(
    doctor_id: str = None,
    start_date: str = None,
    end_date: str = None,
    db = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """GPT 토큰 사용량과 비용을 조회합니다."""
    try:
        check_admin_permission(current_user, db)
        
        # 기본 쿼리
        query = db.query(GPTTokenUsage)
        
        # 의사 ID 필터
        if doctor_id:
            query = query.filter(GPTTokenUsage.doctor_id == doctor_id)
        
        # 날짜 필터
        if start_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=KST)
            query = query.filter(GPTTokenUsage.created_at >= start)
        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=KST)
            end = end.replace(hour=23, minute=59, second=59)
            query = query.filter(GPTTokenUsage.created_at <= end)
        
        # 사용량 집계
        usage_stats = query.with_entities(
            func.sum(GPTTokenUsage.prompt_tokens).label("total_prompt_tokens"),
            func.sum(GPTTokenUsage.completion_tokens).label("total_completion_tokens"),
            func.sum(GPTTokenUsage.total_tokens).label("total_tokens"),
            func.sum(GPTTokenUsage.cost).label("total_cost")
        ).first()
        
        # 모델별 사용량
        model_stats = db.query(
            GPTTokenUsage.model,
            func.count(GPTTokenUsage.id).label("usage_count"),
            func.sum(GPTTokenUsage.total_tokens).label("total_tokens"),
            func.sum(GPTTokenUsage.cost).label("total_cost")
        ).group_by(GPTTokenUsage.model).all()
        
        # 의사별 사용량
        doctor_stats = db.query(
            GPTTokenUsage.doctor_id,
            func.count(GPTTokenUsage.id).label("usage_count"),
            func.sum(GPTTokenUsage.total_tokens).label("total_tokens"),
            func.sum(GPTTokenUsage.cost).label("total_cost")
        ).group_by(GPTTokenUsage.doctor_id).all()
        
        return {
            "total_usage": {
                "prompt_tokens": usage_stats.total_prompt_tokens or 0,
                "completion_tokens": usage_stats.total_completion_tokens or 0,
                "total_tokens": usage_stats.total_tokens or 0,
                "total_cost": round(usage_stats.total_cost or 0, 6)
            },
            "model_stats": [
                {
                    "model": stat.model,
                    "usage_count": stat.usage_count,
                    "total_tokens": stat.total_tokens,
                    "total_cost": round(stat.total_cost, 6)
                }
                for stat in model_stats
            ],
            "doctor_stats": [
                {
                    "doctor_id": stat.doctor_id,
                    "usage_count": stat.usage_count,
                    "total_tokens": stat.total_tokens,
                    "total_cost": round(stat.total_cost, 6)
                }
                for stat in doctor_stats
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ GPT 사용량 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"사용량 조회 중 오류: {str(e)}")

# IP security constants
MAX_IP_ATTEMPTS = 10
IP_BLOCK_DURATION = 60  # minutes

class IPBlock(Base):
    __tablename__ = "ip_blocks"
    
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String, index=True)
    attempts = Column(Integer, default=0)
    last_attempt = Column(DateTime, default=datetime.utcnow)
    blocked_until = Column(DateTime)
    is_blocked = Column(Boolean, default=False)

class LoginAttempt(Base):
    __tablename__ = "login_attempts"
    
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String, index=True)
    doctor_id = Column(String, index=True)
    attempt_time = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=False)
    user_agent = Column(String)

def check_ip_block(ip_address: str, db: Session) -> bool:
    """Check if IP is blocked"""
    ip_block = db.query(IPBlock).filter(IPBlock.ip_address == ip_address).first()
    
    if not ip_block:
        return False
    
    if ip_block.is_blocked:
        if ip_block.blocked_until and ip_block.blocked_until > datetime.utcnow():
            return True
        else:
            # Reset block if block period has expired
            ip_block.is_blocked = False
            ip_block.attempts = 0
            ip_block.blocked_until = None
            db.commit()
            return False
    
    return False

def record_login_attempt(
    ip_address: str,
    doctor_id: str,
    success: bool,
    user_agent: str,
    db: Session
):
    """Record login attempt and update IP block status"""
    # Record attempt
    attempt = LoginAttempt(
        ip_address=ip_address,
        doctor_id=doctor_id,
        success=success,
        user_agent=user_agent
    )
    db.add(attempt)
    
    # Update IP block
    ip_block = db.query(IPBlock).filter(IPBlock.ip_address == ip_address).first()
    if not ip_block:
        ip_block = IPBlock(ip_address=ip_address)
        db.add(ip_block)
    
    if not success:
        ip_block.attempts += 1
        ip_block.last_attempt = datetime.utcnow()
        
        if ip_block.attempts >= MAX_IP_ATTEMPTS:
            ip_block.is_blocked = True
            ip_block.blocked_until = datetime.utcnow() + timedelta(minutes=IP_BLOCK_DURATION)
    else:
        ip_block.attempts = 0
        ip_block.is_blocked = False
        ip_block.blocked_until = None
    
    db.commit()

@app.post("/login")
async def login(
    user_data: UserLogin,
    request: Request,
    db: Session = Depends(get_db)
):
    try:
        ip_address = request.client.host
        user_agent = request.headers.get("user-agent")
        
        # Check IP block
        if check_ip_block(ip_address, db):
            raise HTTPException(
                status_code=403,
                detail=f"너무 많은 로그인 시도로 인해 IP가 차단되었습니다. {IP_BLOCK_DURATION}분 후에 다시 시도해주세요."
            )
        
        user = db.query(User).filter(User.doctor_id == user_data.doctor_id).first()
        
        # Check if account is locked
        if user and user.is_locked:
            if user.lock_until and user.lock_until > datetime.utcnow():
                remaining_time = (user.lock_until - datetime.utcnow()).total_seconds() / 60
                record_login_attempt(ip_address, user_data.doctor_id, False, user_agent, db)
                raise HTTPException(
                    status_code=403,
                    detail=f"계정이 잠겨 있습니다. {int(remaining_time)}분 후에 다시 시도해주세요."
                )
            else:
                # Reset lock if lock period has expired
                user.is_locked = False
                user.login_attempts = 0
                user.lock_until = None
                db.commit()
        
        if not user or not verify_password(user_data.password, user.hashed_password):
            if user:
                user.login_attempts += 1
                user.last_login_attempt = datetime.utcnow()
                
                if user.login_attempts >= MAX_LOGIN_ATTEMPTS:
                    user.is_locked = True
                    user.lock_until = datetime.utcnow() + timedelta(minutes=LOGIN_LOCKOUT_DURATION)
                
                db.commit()
            
            record_login_attempt(ip_address, user_data.doctor_id, False, user_agent, db)
            raise HTTPException(status_code=401, detail="잘못된 ID 또는 비밀번호입니다.")
        
        # Reset login attempts on successful login
        user.login_attempts = 0
        user.last_login = datetime.utcnow()
        db.commit()
        
        if not user.is_active:
            record_login_attempt(ip_address, user_data.doctor_id, False, user_agent, db)
            raise HTTPException(status_code=403, detail="비활성화된 계정입니다. 관리자에게 문의하세요.")
        
        record_login_attempt(ip_address, user_data.doctor_id, True, user_agent, db)
        access_token = create_access_token(data={"sub": user.doctor_id})
        return {"access_token": access_token, "token_type": "bearer"}
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="로그인 중 오류가 발생했습니다.")

@app.get("/admin/login-attempts")
async def get_login_attempts(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Check if user is admin
        user = db.query(User).filter(User.doctor_id == current_user).first()
        if not user or not user.is_admin:
            raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
        
        # Get recent login attempts
        attempts = db.query(LoginAttempt).order_by(
            LoginAttempt.attempt_time.desc()
        ).limit(100).all()
        
        return attempts
        
    except Exception as e:
        logger.error(f"Login attempts retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="로그인 시도 기록 조회 중 오류가 발생했습니다.")

@app.get("/admin/ip-blocks")
async def get_ip_blocks(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Check if user is admin
        user = db.query(User).filter(User.doctor_id == current_user).first()
        if not user or not user.is_admin:
            raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
        
        # Get active IP blocks
        blocks = db.query(IPBlock).filter(
            IPBlock.is_blocked == True,
            IPBlock.blocked_until > datetime.utcnow()
        ).all()
        
        return blocks
        
    except Exception as e:
        logger.error(f"IP blocks retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="IP 차단 목록 조회 중 오류가 발생했습니다.")

class SystemSettings(Base):
    __tablename__ = "system_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    value = Column(String)
    description = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String)

# Default system settings
DEFAULT_SETTINGS = [
    {
        "key": "max_concurrent_sessions",
        "value": "2",
        "description": "사용자당 최대 동시 세션 수"
    },
    {
        "key": "session_timeout_minutes",
        "value": "30",
        "description": "세션 타임아웃 시간(분)"
    }
]

def get_system_setting(key: str, db: Session) -> str:
    """Get system setting value"""
    setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()
    if not setting:
        # Create default setting if not exists
        default = next((s for s in DEFAULT_SETTINGS if s["key"] == key), None)
        if default:
            setting = SystemSettings(**default)
            db.add(setting)
            db.commit()
            db.refresh(setting)
    return setting.value if setting else None

@app.post("/admin/settings")
async def update_system_settings(
    settings: dict,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Check if user is admin
        user = db.query(User).filter(User.doctor_id == current_user).first()
        if not user or not user.is_admin:
            raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
        
        for key, value in settings.items():
            setting = db.query(SystemSettings).filter(SystemSettings.key == key).first()
            if setting:
                setting.value = str(value)
                setting.updated_by = current_user
            else:
                setting = SystemSettings(
                    key=key,
                    value=str(value),
                    updated_by=current_user
                )
                db.add(setting)
        
        db.commit()
        return {"message": "시스템 설정이 업데이트되었습니다."}
        
    except Exception as e:
        logger.error(f"System settings update error: {str(e)}")
        raise HTTPException(status_code=500, detail="시스템 설정 업데이트 중 오류가 발생했습니다.")

@app.get("/admin/settings")
async def get_system_settings(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Check if user is admin
        user = db.query(User).filter(User.doctor_id == current_user).first()
        if not user or not user.is_admin:
            raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")
        
        settings = db.query(SystemSettings).all()
        return {setting.key: setting.value for setting in settings}
        
    except Exception as e:
        logger.error(f"System settings retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="시스템 설정 조회 중 오류가 발생했습니다.")

@app.post("/sessions")
async def create_session(
    session_data: SessionCreate,
    current_user: str = Depends(get_current_user),
    request: Request = None,
    db: Session = Depends(get_db)
):
    try:
        # Check user's active status
        user = db.query(User).filter(User.doctor_id == current_user).first()
        if not user or not user.is_active:
            raise HTTPException(status_code=403, detail="비활성화된 계정입니다. 관리자에게 문의하세요.")
        
        # Get max concurrent sessions from settings
        max_sessions = int(get_system_setting("max_concurrent_sessions", db))
        
        # Check concurrent sessions
        active_sessions = db.query(Session).filter(
            Session.doctor_id == current_user,
            Session.is_active == True
        ).all()
        
        # Deactivate timed out sessions
        timeout_minutes = int(get_system_setting("session_timeout_minutes", db))
        for session in active_sessions:
            if (datetime.utcnow() - session.last_activity).total_seconds() > (timeout_minutes * 60):
                session.is_active = False
                db.commit()
        
        # Count remaining active sessions
        active_sessions = [s for s in active_sessions if s.is_active]
        if len(active_sessions) >= max_sessions:
            raise HTTPException(
                status_code=403,
                detail=f"최대 {max_sessions}개의 동시 세션만 허용됩니다. 다른 세션을 종료해주세요."
            )
        
        # Create new session
        session_id = str(uuid.uuid4())
        new_session = Session(
            session_id=session_id,
            doctor_id=current_user,
            patient_name=session_data.patient_name,
            ip_address=request.client.host if request else None,
            user_agent=request.headers.get("user-agent") if request else None
        )
        
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        
        return {"session_id": session_id}
        
    except Exception as e:
        logger.error(f"Session creation error: {str(e)}")
        raise HTTPException(status_code=500, detail="세션 생성 중 오류가 발생했습니다.")

def get_current_user():
    # TODO: Replace with real authentication logic (e.g., JWT token validation)
    return "admin"

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

@app.post("/auth/change-password")
async def change_password(
    data: PasswordChangeRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.doctor_id == current_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    # 현재 비밀번호 확인 (임시 평문/해시 모두 허용)
    if not verify_password(data.current_password, user.hashed_password):
        raise HTTPException(status_code=401, detail="현재 비밀번호가 일치하지 않습니다.")

    # 새 비밀번호 정책 검사
    if not validate_password(data.new_password):
        raise HTTPException(
            status_code=400,
            detail="비밀번호는 8자 이상이며, 대문자, 소문자, 숫자, 특수문자를 포함해야 합니다."
        )

    # 새 비밀번호가 최근 사용한 비밀번호와 중복되는지 검사
    if not check_password_history(user, data.new_password):
        raise HTTPException(
            status_code=400,
            detail="최근 사용한 비밀번호는 사용할 수 없습니다."
        )

    # 비밀번호 변경
    hashed_new = hash_password(data.new_password)
    password_history = user.password_history or []
    password_history.append(hashed_new)
    if len(password_history) > 5:
        password_history = password_history[-5:]
    user.hashed_password = hashed_new
    user.password_history = password_history
    user.last_password_change = datetime.utcnow()
    db.commit()

    return {"message": "비밀번호가 성공적으로 변경되었습니다."}

@app.post("/sct/sessions/{session_id}/regenerate")
async def regenerate_interpretation(
    session_id: str, 
    db = Depends(get_db), 
    current_user: str = Depends(get_current_user)
):
    """해석을 재생성합니다."""
    try:
        logger.info(f"🔄 해석 재생성 요청: session_id={session_id}, user={current_user}")
        
        # 사용자 권한 확인 - 더 안전한 방식
        try:
            user = db.query(User).filter(User.doctor_id == current_user).first()
            if not user:
                logger.error(f"❌ 사용자를 찾을 수 없음: {current_user}")
                raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")
            
            # admin 계정 권한 임시 설정
            if user.doctor_id == "admin":
                user.is_admin = True
                logger.info(f"✅ admin 계정 권한 임시 설정: {current_user}")
            
            if not user.is_active:
                logger.error(f"❌ 비활성화된 계정: {current_user}")
                raise HTTPException(status_code=403, detail="비활성화된 계정입니다")
            
            if not user.is_verified:
                logger.error(f"❌ 미승인 계정: {current_user}")
                raise HTTPException(status_code=403, detail="승인되지 않은 계정입니다")
                
        except Exception as e:
            logger.error(f"❌ 사용자 권한 확인 실패: {e}")
            raise HTTPException(status_code=403, detail="사용자 권한 확인에 실패했습니다")
        
        # 세션 정보 확인
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            logger.error(f"❌ 세션을 찾을 수 없음: {session_id}")
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # 세션 소유권 확인 (관리자는 모든 세션에 접근 가능)
        if session.doctor_id != current_user:
            # 관리자 권한 확인
            if not user.is_admin:
                logger.error(f"❌ 권한 없는 접근: session_owner={session.doctor_id}, requester={current_user}")
                raise HTTPException(status_code=403, detail="해당 세션에 대한 접근 권한이 없습니다")
            else:
                logger.info(f"✅ 관리자 권한으로 접근: {current_user}")
        
        if session.status != "complete":
            logger.error(f"❌ 완료되지 않은 세션: {session_id}, status={session.status}")
            raise HTTPException(status_code=400, detail="완료된 검사만 해석이 가능합니다")
        
        # 응답 데이터 확인
        responses = db.query(SCTResponse).filter(
            SCTResponse.session_id == session_id
        ).order_by(SCTResponse.item_no).all()
        
        if not responses:
            logger.error(f"❌ 응답이 없음: {session_id}")
            raise HTTPException(status_code=400, detail="응답 데이터가 없습니다")
        
        logger.info(f"📊 응답 데이터 확인됨: {len(responses)}개 응답")
        
        # 기존 해석 삭제 (더 안전한 방식)
        try:
            existing_interpretation = db.query(SCTInterpretation).filter(
                SCTInterpretation.session_id == session_id
            ).first()
            
            if existing_interpretation:
                logger.info(f"🗑️ 기존 해석 삭제: {session_id}")
                db.delete(existing_interpretation)
                db.flush()  # commit 대신 flush 사용
                
        except Exception as e:
            logger.error(f"❌ 기존 해석 삭제 실패: {e}")
            db.rollback()
            raise HTTPException(status_code=500, detail="기존 해석 삭제에 실패했습니다")
        
        # 새로운 해석 생성
        try:
            logger.info(f"🧠 AI 해석 생성 시작: {session_id}")
            interpretation = await generate_ai_interpretation(
                responses, 
                session.patient_name, 
                session.doctor_id, 
                session.session_id, 
                db
            )
            
            if not interpretation or len(interpretation.strip()) < 100:
                logger.error(f"❌ 해석 생성 실패 또는 너무 짧음: {len(interpretation) if interpretation else 0}자")
                raise Exception("해석 생성에 실패했거나 결과가 부적절합니다")
                
            logger.info(f"✅ AI 해석 생성 완료: {len(interpretation)}자")
            
        except Exception as e:
            logger.error(f"❌ AI 해석 생성 실패: {e}")
            db.rollback()
            # OpenAI 실패 시 기본 해석 사용
            interpretation = generate_default_interpretation(responses, session.patient_name)
            logger.info("⚠️ 기본 해석으로 대체됨")
        
        # 새로운 해석 저장
        try:
            new_interpretation = SCTInterpretation(
                session_id=session_id,
                interpretation=interpretation,
                patient_name=session.patient_name,
                created_at=get_kst_now().replace(tzinfo=None)
            )
            
            db.add(new_interpretation)
            db.commit()
            logger.info(f"✅ 새로운 해석 저장 완료: {session_id}")
            
        except Exception as e:
            logger.error(f"❌ 해석 저장 실패: {e}")
            db.rollback()
            raise HTTPException(status_code=500, detail="해석 저장에 실패했습니다")
        
        return {
            "session_id": session_id,
            "interpretation": interpretation,
            "generated_at": get_kst_now().isoformat(),
            "patient_name": session.patient_name,
            "message": "해석이 성공적으로 재생성되었습니다"
        }
        
    except HTTPException:
        # HTTPException은 그대로 전달
        raise
    except Exception as e:
        logger.error(f"❌ 해석 재생성 중 예상치 못한 오류: {e}")
        logger.error(f"❌ 오류 상세: {type(e).__name__}: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"해석 재생성 중 서버 오류가 발생했습니다: {str(e)}"
        )

@app.get("/sct/sessions/{session_id}/responses")
async def get_session_responses(session_id: str, db = Depends(get_db)):
    """세션의 원본 응답을 조회합니다."""
    try:
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        responses = db.query(SCTResponse).filter(
            SCTResponse.session_id == session_id
        ).order_by(SCTResponse.item_no).all()
        
        return {
            "session_id": session_id,
            "patient_name": session.patient_name,
            "responses": [
                {
                    "item_no": r.item_no,
                    "stem": r.stem,
                    "answer": r.answer
                } for r in responses
            ],
            "submitted_at": session.submitted_at
        }
        
    except Exception as e:
        logger.error(f"❌ 응답 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"응답 조회 중 오류: {str(e)}")

# 환자용 세션 조회 (인증 불필요)
@app.get("/sct/sessions/{session_id}/patient")
async def get_session_for_patient(session_id: str, db = Depends(get_db)):
    """환자용 세션 정보 조회 (인증 불필요)"""
    try:
        logger.info(f"👤 환자용 세션 조회: {session_id}")
        
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        # 만료 확인
        current_time = get_kst_now().replace(tzinfo=None)
        if session.expires_at and session.expires_at < current_time:
            session.status = "expired"
            db.commit()
            raise HTTPException(status_code=410, detail="만료된 세션입니다")
        
        # 기존 응답 조회
        responses = db.query(SCTResponse).filter(
            SCTResponse.session_id == session_id
        ).order_by(SCTResponse.item_no).all()
        
        # 문항 목록 생성 (기존 응답 포함)
        items = []
        existing_answers = {resp.item_no: resp.answer for resp in responses}
        
        for i, stem in enumerate(SCT_ITEMS, 1):
            items.append({
                "item_no": i,
                "stem": stem,
                "answer": existing_answers.get(i, "")
            })
        
        return {
            "session_id": session.session_id,
            "patient_name": session.patient_name,
            "status": session.status,
            "items": items,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "expires_at": session.expires_at.isoformat() if session.expires_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 환자용 세션 조회 오류: {e}")
        raise HTTPException(status_code=500, detail="세션 조회 중 오류가 발생했습니다")

# 환자용 응답 저장 (인증 불필요)
@app.post("/sct/sessions/{session_id}/responses")
async def save_patient_responses(
    session_id: str, 
    request_data: dict,
    db = Depends(get_db)
):
    """환자용 응답 저장 (인증 불필요)"""
    try:
        logger.info(f"💾 환자 응답 저장: {session_id}")
        
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        if session.status == "expired":
            raise HTTPException(status_code=410, detail="만료된 세션입니다")
        
        responses = request_data.get("responses", [])
        
        # 기존 응답 삭제
        db.query(SCTResponse).filter(SCTResponse.session_id == session_id).delete()
        
        # 새 응답 저장
        saved_count = 0
        for response in responses:
            if response.get("answer") and response.get("answer").strip():
                db_response = SCTResponse(
                    session_id=session_id,
                    item_no=response.get("item_no"),
                    stem=response.get("stem", ""),
                    answer=response.get("answer").strip(),
                    created_at=get_kst_now().replace(tzinfo=None)
                )
                db.add(db_response)
                saved_count += 1
        
        db.commit()
        
        return {"message": "응답이 저장되었습니다", "saved_count": saved_count}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 환자 응답 저장 오류: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="응답 저장 중 오류가 발생했습니다")

# 환자용 검사 완료 (인증 불필요)
@app.post("/sct/sessions/{session_id}/complete")
async def complete_patient_session(
    session_id: str,
    request_data: dict,
    db = Depends(get_db)
):
    """환자용 검사 완료 (인증 불필요)"""
    try:
        logger.info(f"✅ 환자 검사 완료: {session_id}")
        
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        responses = request_data.get("responses", [])
        
        # 기존 응답 삭제 및 새로 저장
        db.query(SCTResponse).filter(SCTResponse.session_id == session_id).delete()
        
        saved_count = 0
        for response in responses:
            if response.get("answer") and response.get("answer").strip():
                db_response = SCTResponse(
                    session_id=session_id,
                    item_no=response.get("item_no"),
                    stem=response.get("stem", ""),
                    answer=response.get("answer").strip(),
                    created_at=get_kst_now().replace(tzinfo=None)
                )
                db.add(db_response)
                saved_count += 1
        
        # 세션 완료 처리
        session.status = "complete"
        session.submitted_at = get_kst_now().replace(tzinfo=None)
        
        db.commit()
        
        logger.info(f"✅ 검사 완료: {saved_count}개 응답 저장")
        
        return {
            "message": "검사가 완료되었습니다",
            "session_id": session_id,
            "saved_count": saved_count,
            "completed_at": session.submitted_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 검사 완료 오류: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="검사 완료 중 오류가 발생했습니다")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)