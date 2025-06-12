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

# SCT 검사 문항 (50개로 수정)
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
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(days=7)
        
        db_session = SCTSession(
            session_id=session_id,
            doctor_id=current_user,
            patient_name=session_data.patient_name,
            expires_at=expires_at
        )
        
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        
        logger.info(f"✅ 새 세션 생성: {session_id} by {current_user}")
        return {"session_id": session_id, "expires_at": expires_at}
        
    except Exception as e:
        logger.error(f"❌ 세션 생성 오류: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="세션 생성 중 오류가 발생했습니다")

@app.get("/sct/sessions/by-user/{doctor_id}")
async def get_sessions_by_user(
    doctor_id: str, 
    db = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    try:
        if current_user != doctor_id:
            raise HTTPException(status_code=403, detail="접근 권한이 없습니다")
        
        sessions = db.query(SCTSession).filter(SCTSession.doctor_id == doctor_id).all()
        
        # 만료된 세션 상태 업데이트
        for session in sessions:
            if session.expires_at < datetime.utcnow() and session.status != "complete":
                session.status = "expired"
        
        db.commit()
        
        return {"sessions": sessions}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 세션 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail="세션 목록 조회 중 오류가 발생했습니다")

# 기타 엔드포인트들도 유사한 오류 처리 추가...
# (patient.html에서 사용하는 엔드포인트들)

@app.get("/sct/session/{session_id}")
async def get_session(session_id: str, db = Depends(get_db)):
    try:
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        if session.expires_at < datetime.utcnow():
            session.status = "expired"
            db.commit()
            raise HTTPException(status_code=410, detail="만료된 세션입니다")
        
        responses = db.query(SCTResponse).filter(SCTResponse.session_id == session_id).all()
        
        return {
            "session": session,
            "responses": responses,
            "total_items": len(SCT_ITEMS),
            "completed_items": len(responses)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 세션 조회 오류: {e}")
        raise HTTPException(status_code=500, detail="세션 조회 중 오류가 발생했습니다")

@app.get("/sct/session/{session_id}/items")
async def get_session_items(session_id: str, db = Depends(get_db)):
    try:
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        if session.expires_at < datetime.utcnow():
            raise HTTPException(status_code=410, detail="만료된 세션입니다")
        
        # 기존 응답 가져오기
        existing_responses = db.query(SCTResponse).filter(SCTResponse.session_id == session_id).all()
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
        session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
        
        if session.expires_at < datetime.utcnow():
            raise HTTPException(status_code=410, detail="만료된 세션입니다")
        
        # 기존 응답 삭제
        db.query(SCTResponse).filter(SCTResponse.session_id == session_id).delete()
        
        # 새 응답 저장
        for response in responses:
            if response.answer.strip():
                db_response = SCTResponse(
                    session_id=session_id,
                    item_no=response.item_no,
                    stem=response.stem,
                    answer=response.answer.strip()
                )
                db.add(db_response)
        
        # 세션 상태 업데이트
        session.status = "complete"
        session.submitted_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"✅ 응답 저장 완료: {session_id}")
        return {"message": "응답이 저장되었습니다"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 응답 저장 오류: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="응답 저장 중 오류가 발생했습니다")

# AI 해석 생성 함수
async def generate_ai_interpretation(responses: List[SCTResponse], patient_name: str) -> str:
    if not openai_client:
        return generate_default_interpretation(responses, patient_name)
    
    responses_text = "\n".join([
        f"{resp.item_no}. {resp.stem} → {resp.answer}"
        for resp in responses
    ])
    
    prompt = f"""
당신은 20년 경력의 임상심리 전문가입니다. 다음 SCT(문장완성검사) 응답을 분석하여 전문적인 해석을 제공해주세요.

환자명: {patient_name}
검사일: {datetime.now().strftime('%Y년 %m월 %d일')}

SCT 응답:
{responses_text}

다음 구조로 해석해주세요:

# SCT 검사 해석 보고서

## 1. 전반적 개관
- 검사 태도 및 전반적 인상
- 주요 특징 요약

## 2. 주요 심리적 특성

### 2.1 자아개념 및 정체성
- 자기 인식과 자존감
- 정체성 발달 수준

### 2.2 대인관계 패턴
- 사회적 관계의 질
- 애착 스타일

### 2.3 정서적 적응
- 감정 조절 능력
- 스트레스 대처 방식

### 2.4 가족 관계
- 부모와의 관계
- 가족 역동성

### 2.5 미래 전망 및 포부
- 목표 의식
- 미래에 대한 태도

## 3. 임상적 시사점
- 주요 강점
- 관심 영역
- 권고사항

전문적이고 객관적인 언어를 사용하되, 따뜻하고 이해하기 쉽게 작성해주세요.
"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 전문 임상심리사입니다. SCT 검사 결과를 정확하고 전문적으로 해석해주세요."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI API 오류: {e}")
        return generate_default_interpretation(responses, patient_name)

def generate_default_interpretation(responses: List[SCTResponse], patient_name: str) -> str:
    return f"""
# SCT 검사 해석 보고서

**환자명:** {patient_name}
**검사일:** {datetime.now().strftime('%Y년 %m월 %d일')}

## 1. 전반적 개관
환자는 SCT 검사에 성실하게 응답하였으며, 총 {len(responses)}개의 문항에 대해 의미 있는 응답을 제공했습니다.

## 2. 주요 특성 분석
응답 패턴을 통해 다음과 같은 특성들이 관찰됩니다:

### 자아개념
- 자기 인식 수준과 정체성 발달 상태
- 자존감 및 자기효능감

### 대인관계
- 사회적 관계에 대한 태도
- 타인과의 상호작용 패턴

### 정서적 측면
- 감정 표현 및 조절 능력
- 스트레스 대처 방식

## 3. 임상적 제언
- 지속적인 관찰 및 추가 평가 필요
- 강점 활용 및 발전 영역 확인

*주의: 기본 분석이므로 전문가의 직접 검토가 필요합니다.*
*OpenAI API 연동 후 더 상세한 해석이 제공됩니다.*
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)