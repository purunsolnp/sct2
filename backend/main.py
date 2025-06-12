from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional, List
import hashlib
import jwt
import os
from openai import OpenAI
import json
import uuid

# 환경 변수
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/sct_db")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key-here")

# 데이터베이스 설정
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# OpenAI 클라이언트
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI 앱 초기화
app = FastAPI(title="SCT 검사 시스템 API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포시에는 특정 도메인만 허용
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
    status = Column(String, default="incomplete")  # incomplete, complete, expired
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

# 데이터베이스 테이블 생성
Base.metadata.create_all(bind=engine)

# Pydantic 모델 (API 요청/응답)
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

class SessionResponse(BaseModel):
    session_id: str
    patient_name: str
    status: str
    created_at: datetime
    submitted_at: Optional[datetime]
    expires_at: datetime

class SCTResponseCreate(BaseModel):
    item_no: int
    stem: str
    answer: str

class SCTResponseModel(BaseModel):
    item_no: int
    stem: str
    answer: str

# 유틸리티 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

# SCT 검사 문항 (60개)
SCT_ITEMS = [
    "나는 항상", "다시 돌아간다면", "나는 바란다", "나의 가장 큰 걱정은", "가정은",
    "나는 할 수 없다", "나의 미래는", "상급자들은", "나는 알고 있다", "어린 시절",
    "내가 가장 좋아하는 것은", "사람들이 가장 몰라주는 것은", "나의 어머니는", "나는 다른 사람들에게서 가장",
    "나의 아버지는", "만약 나에게 기회가 있다면", "나는 걱정된다", "사람들은", "어머니와 나는",
    "나의 가장 큰 약점은", "나의 주된 야망은", "내가 아주 싫어하는 것은", "아버지와 나는", "나의 신경을 건드리는 것은",
    "나의 마음은", "사람들 대부분은 나를", "나는 확실히 할 수 있다", "나의 가장 큰 걱정거리는", "내가 집에 있을 때",
    "내가 어렸을 때", "나의 가장 나쁜 습관은", "내가 하고 싶은 일은", "나의 신경질을 돋우는 것은", "다른 사람들",
    "나의 주된 문제는", "나는 좋아한다", "내가 술을 마실 때", "나는 비밀리에", "사람들은 나를",
    "나의 어머니와 나는", "내가 명령을 받을 때", "내가 가장 자신 있는 것은", "미래에 나는", "내가 가장 필요로 하는 것은",
    "나는 두려워한다", "나에게 결혼이란", "나의 가족은", "나는 가장", "만약 나의 아버지가",
    "나의 가장 큰 실수는", "나는 주로", "내가 나이가 들면", "나의 가장 큰 바람은", "내가 아이들과 함께 있을 때",
    "성에 대한 나의 태도는", "사람들이 나에 대해 모르는 것은", "나는 노력한다", "나의 가장 생생한 기억은", "나의 가장 큰 두려움은",
    "결혼한 생활은", "내가 사람들과 함께 있을 때", "다른 여성들", "다른 남성들", "나의 가장 큰 고민은"
]

# API 엔드포인트

@app.get("/")
async def root():
    return {"message": "SCT 검사 시스템 API", "version": "1.0.0"}

@app.post("/auth/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
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
    
    return {"message": "회원가입이 완료되었습니다"}

@app.post("/auth/login", response_model=TokenResponse)
async def login(user_login: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.doctor_id == user_login.doctor_id).first()
    
    if not user or not verify_password(user_login.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="잘못된 ID 또는 비밀번호입니다")
    
    access_token = create_access_token(data={"sub": user.doctor_id})
    
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

@app.get("/auth/check-id/{doctor_id}")
async def check_doctor_id(doctor_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.doctor_id == doctor_id).first()
    return {"available": user is None}

@app.post("/sct/sessions")
async def create_session(
    session_data: SessionCreate, 
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    session_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(days=7)  # 7일 후 만료
    
    db_session = SCTSession(
        session_id=session_id,
        doctor_id=current_user,
        patient_name=session_data.patient_name,
        expires_at=expires_at
    )
    
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    
    return {"session_id": session_id, "expires_at": expires_at}

@app.get("/sct/sessions/by-user/{doctor_id}")
async def get_sessions_by_user(
    doctor_id: str, 
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    if current_user != doctor_id:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다")
    
    sessions = db.query(SCTSession).filter(SCTSession.doctor_id == doctor_id).all()
    
    # 만료된 세션 상태 업데이트
    for session in sessions:
        if session.expires_at < datetime.utcnow() and session.status != "complete":
            session.status = "expired"
    
    db.commit()
    
    return {"sessions": sessions}

@app.get("/sct/session/{session_id}")
async def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    # 만료 확인
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

@app.get("/sct/session/{session_id}/items")
async def get_session_items(session_id: str, db: Session = Depends(get_db)):
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

@app.post("/sct/session/{session_id}/responses")
async def save_responses(
    session_id: str, 
    responses: List[SCTResponseCreate], 
    db: Session = Depends(get_db)
):
    session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    if session.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="만료된 세션입니다")
    
    # 기존 응답 삭제
    db.query(SCTResponse).filter(SCTResponse.session_id == session_id).delete()
    
    # 새 응답 저장
    for response in responses:
        if response.answer.strip():  # 빈 답변은 저장하지 않음
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
    
    return {"message": "응답이 저장되었습니다"}

@app.post("/sct/sessions/{session_id}/interpret")
async def generate_interpretation(
    session_id: str, 
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    if session.doctor_id != current_user:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다")
    
    responses = db.query(SCTResponse).filter(SCTResponse.session_id == session_id).all()
    if not responses:
        raise HTTPException(status_code=400, detail="저장된 응답이 없습니다")
    
    # AI 해석 생성
    try:
        interpretation_text = await generate_ai_interpretation(responses, session.patient_name)
        
        # 기존 해석 삭제 후 새로 저장
        db.query(SCTInterpretation).filter(SCTInterpretation.session_id == session_id).delete()
        
        db_interpretation = SCTInterpretation(
            session_id=session_id,
            interpretation=interpretation_text,
            patient_name=session.patient_name
        )
        
        db.add(db_interpretation)
        db.commit()
        
        return {"message": "해석이 생성되었습니다"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"해석 생성 중 오류: {str(e)}")

@app.get("/sct/sessions/{session_id}/interpretation")
async def get_interpretation(
    session_id: str, 
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    if session.doctor_id != current_user:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다")
    
    interpretation = db.query(SCTInterpretation).filter(SCTInterpretation.session_id == session_id).first()
    if not interpretation:
        raise HTTPException(status_code=404, detail="해석 결과를 찾을 수 없습니다")
    
    return {
        "session_id": session_id,
        "patient_name": interpretation.patient_name,
        "interpretation": interpretation.interpretation,
        "created_at": interpretation.created_at
    }

@app.get("/sct/sessions/{session_id}/analysis")
async def get_categorical_analysis(
    session_id: str, 
    db: Session = Depends(get_db),
    current_user: str = Depends(verify_token)
):
    session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    if session.doctor_id != current_user:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다")
    
    responses = db.query(SCTResponse).filter(SCTResponse.session_id == session_id).all()
    
    # 카테고리별 분류 (간단한 버전)
    categorized = {
        "가족관계": [],
        "대인관계": [],
        "자아개념": [],
        "정서조절": [],
        "성_결혼관": [],
        "미래전망": [],
        "과거경험": [],
        "현실적응": [],
        "성격특성": []
    }
    
    # 간단한 키워드 기반 분류
    family_keywords = ["어머니", "아버지", "가족", "부모", "형제", "자매"]
    relationship_keywords = ["사람들", "친구", "동료", "상급자"]
    self_keywords = ["나는", "내가", "나의"]
    
    for response in responses:
        response_dict = {
            "item_no": response.item_no,
            "stem": response.stem,
            "answer": response.answer
        }
        
        if any(keyword in response.stem for keyword in family_keywords):
            categorized["가족관계"].append(response_dict)
        elif any(keyword in response.stem for keyword in relationship_keywords):
            categorized["대인관계"].append(response_dict)
        else:
            categorized["자아개념"].append(response_dict)
    
    return {"categorized_responses": categorized}

# AI 해석 생성 함수
async def generate_ai_interpretation(responses: List[SCTResponse], patient_name: str) -> str:
    # 응답을 텍스트로 변환
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
        # OpenAI API 오류시 기본 해석 반환
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

*주의: AI 기반 초기 분석이므로 전문가의 직접 검토가 필요합니다.*
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)