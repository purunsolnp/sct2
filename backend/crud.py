from sqlalchemy.orm import Session
from models import User, SCTSession, SCTResponse, SCTInterpretation, GPTTokenUsage, IPBlock, LoginAttempt, SystemSettings
from schemas import UserCreate, SessionCreate
from datetime import datetime
import uuid

# User CRUD

def create_user(db: Session, user: UserCreate):
    # 비밀번호는 이미 해싱되어 전달됨
    db_user = User(
        doctor_id=user.doctor_id,
        email=user.email,
        hashed_password=user.password,  # 이미 해싱된 비밀번호
        first_name=user.first_name,
        last_name=user.last_name,
        specialty=user.specialty,
        hospital=user.hospital,
        phone=user.phone,
        medical_license=user.medical_license,
        created_at=datetime.utcnow(),
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_id(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_doctor_id(db: Session, doctor_id: str):
    return db.query(User).filter(User.doctor_id == doctor_id).first()

# Session CRUD

def create_session(db: Session, session: SessionCreate, doctor_id: str):
    session_id = str(uuid.uuid4())
    print(f"생성할 session_id: {session_id}")
    
    db_session = SCTSession(
        session_id=session_id,
        doctor_id=doctor_id,
        patient_name=session.patient_name,
        status="incomplete",
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow(),  # 실제 만료일 계산 필요
    )
    
    try:
        print(f"세션 생성 시도: {db_session.__dict__}")
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        
        print(f"DB에 저장된 세션: {db_session.__dict__}")
        
        # 저장 확인
        created_session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
        if not created_session:
            raise Exception("세션이 데이터베이스에 저장되지 않았습니다")
        
        print(f"조회된 세션: {created_session.__dict__}")
        return created_session
        
    except Exception as e:
        db.rollback()
        print(f"세션 생성 중 오류: {str(e)}")
        raise Exception(f"세션 생성 중 오류 발생: {str(e)}")

def get_sessions_by_user(db: Session, doctor_id: str):
    return db.query(SCTSession).filter(SCTSession.doctor_id == doctor_id).all()

def authenticate_user(db: Session, user_login):
    user = db.query(User).filter(User.doctor_id == user_login.doctor_id).first()
    if user and user.hashed_password == user_login.password:  # 실제로는 해시 비교 필요
        return user
    return None

def get_session_by_id(db: Session, session_id: str):
    return db.query(SCTSession).filter(SCTSession.session_id == session_id).first()

def get_responses_by_session_id(db: Session, session_id: str):
    """세션 ID로 응답 목록을 조회합니다."""
    session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
    if not session or not session.responses:
        return []
    
    # responses는 JSON 형태로 저장되어 있으므로 그대로 반환
    return session.responses

def submit_session_responses(db: Session, session_id: str, responses):
    session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
    if not session:
        raise Exception("세션을 찾을 수 없습니다")
    
    # 응답 데이터를 JSON 형태로 저장
    response_data = [{"item_no": r.item_no, "stem": r.stem, "answer": r.answer} for r in responses]
    
    session.responses = response_data
    session.status = "complete"
    session.submitted_at = datetime.utcnow()
    
    db.commit()
    db.refresh(session)
    return session

def generate_interpretation(db: Session, session_id: str):
    session = db.query(SCTSession).filter(SCTSession.session_id == session_id).first()
    if not session:
        raise Exception("세션을 찾을 수 없습니다")
    
    # TODO: 실제 GPT 해석 로직 구현
    # 현재는 더미 데이터 반환
    interpretation = f"환자 {session.patient_name}의 SCT 검사 해석 결과입니다. (더미 데이터)"
    
    session.interpretation = interpretation
    db.commit()
    db.refresh(session)
    
    return interpretation 