from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc
from models import User, SCTSession, SCTResponse, SCTInterpretation, GPTTokenUsage, IPBlock, LoginAttempt, SystemSettings
from schemas import UserCreate, SessionCreate
from datetime import datetime, timedelta
from typing import List, Optional
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
    
    # 만료 시간을 7일 후로 설정
    now = datetime.utcnow()
    expires_at = now + timedelta(days=7)
    
    db_session = SCTSession(
        session_id=session_id,
        doctor_id=doctor_id,
        patient_name=session.patient_name,
        status="incomplete",
        created_at=now,
        expires_at=expires_at,
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
        # 상세한 오류 메시지 포함
        error_msg = f"세션 생성 중 오류 발생 - doctor_id: {doctor_id}, patient_name: {session.patient_name}, error: {str(e)}"
        raise Exception(error_msg)

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

def get_all_users(db: Session) -> List[User]:
    """모든 사용자 목록을 반환합니다."""
    return db.query(User).all()

def get_user_stats(db: Session) -> dict:
    """사용자 관련 통계를 반환합니다."""
    total_users = db.query(func.count(User.id)).scalar()
    active_users = db.query(func.count(User.id)).filter(User.last_login > datetime.now() - timedelta(days=30)).scalar()
    
    return {
        "total_users": total_users or 0,
        "active_users": active_users or 0
    }

def get_session_stats(db: Session) -> dict:
    """세션 관련 통계를 반환합니다."""
    now = datetime.now()
    start_of_month = datetime(now.year, now.month, 1)
    
    total_sessions = db.query(func.count(SCTSession.session_id)).scalar()
    this_month_sessions = db.query(func.count(SCTSession.session_id))\
        .filter(SCTSession.created_at >= start_of_month).scalar()
    pending_sessions = db.query(func.count(SCTSession.session_id))\
        .filter(SCTSession.status == 'incomplete').scalar()
    completed_sessions = db.query(func.count(SCTSession.session_id))\
        .filter(SCTSession.status == 'complete').scalar()
    this_month_completed = db.query(func.count(SCTSession.session_id))\
        .filter(and_(
            SCTSession.status == 'complete',
            SCTSession.created_at >= start_of_month
        )).scalar()
    expired_sessions = db.query(func.count(SCTSession.session_id))\
        .filter(SCTSession.expires_at < now).scalar()
    
    completion_rate = (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0
    
    return {
        "total_sessions": total_sessions or 0,
        "this_month_sessions": this_month_sessions or 0,
        "pending_sessions": pending_sessions or 0,
        "completed_sessions": completed_sessions or 0,
        "completion_rate": round(completion_rate, 1),
        "this_month_completed": this_month_completed or 0,
        "expired_sessions": expired_sessions or 0
    }

def get_monthly_stats(db: Session, months: int = 12) -> List[dict]:
    """월별 통계를 반환합니다."""
    now = datetime.now()
    monthly_stats = []
    
    for i in range(months):
        start_date = datetime(now.year, now.month, 1) - timedelta(days=30*i)
        end_date = start_date + timedelta(days=32)
        end_date = datetime(end_date.year, end_date.month, 1)
        
        total_sessions = db.query(func.count(SCTSession.session_id))\
            .filter(and_(
                SCTSession.created_at >= start_date,
                SCTSession.created_at < end_date
            )).scalar()
            
        completed_sessions = db.query(func.count(SCTSession.session_id))\
            .filter(and_(
                SCTSession.created_at >= start_date,
                SCTSession.created_at < end_date,
                SCTSession.status == 'complete'
            )).scalar()
            
        token_usage = db.query(
            func.sum(GPTTokenUsage.total_tokens).label('total_tokens'),
            func.sum(GPTTokenUsage.cost).label('total_cost')
        ).filter(and_(
            GPTTokenUsage.created_at >= start_date,
            GPTTokenUsage.created_at < end_date
        )).first()
        
        monthly_stats.append({
            "month_name": start_date.strftime("%Y-%m"),
            "total_sessions": total_sessions or 0,
            "completed_sessions": completed_sessions or 0,
            "total_tokens": token_usage.total_tokens or 0 if token_usage else 0,
            "total_cost": float(token_usage.total_cost or 0) if token_usage else 0
        })
    
    return monthly_stats

def get_gpt_usage_stats(db: Session, start_date: datetime, end_date: datetime) -> dict:
    """GPT 사용량 통계를 반환합니다."""
    # 전체 사용량
    total_usage = db.query(
        func.sum(GPTTokenUsage.total_tokens).label('total_tokens'),
        func.sum(GPTTokenUsage.cost).label('total_cost')
    ).filter(and_(
        GPTTokenUsage.created_at >= start_date,
        GPTTokenUsage.created_at <= end_date
    )).first()
    
    # 일별 사용량
    daily_usage = db.query(
        func.date(GPTTokenUsage.created_at).label('date'),
        func.sum(GPTTokenUsage.total_tokens).label('tokens'),
        func.sum(GPTTokenUsage.cost).label('cost')
    ).filter(and_(
        GPTTokenUsage.created_at >= start_date,
        GPTTokenUsage.created_at <= end_date
    )).group_by(func.date(GPTTokenUsage.created_at))\
    .order_by(func.date(GPTTokenUsage.created_at)).all()
    
    # 사용자별 사용량
    user_usage = db.query(
        GPTTokenUsage.doctor_id,
        func.sum(GPTTokenUsage.total_tokens).label('tokens'),
        func.sum(GPTTokenUsage.cost).label('cost')
    ).filter(and_(
        GPTTokenUsage.created_at >= start_date,
        GPTTokenUsage.created_at <= end_date
    )).group_by(GPTTokenUsage.doctor_id).all()
    
    return {
        "total_usage": {
            "total_tokens": total_usage.total_tokens or 0 if total_usage else 0,
            "total_cost": float(total_usage.total_cost or 0) if total_usage else 0
        },
        "daily_usage": [
            {
                "date": usage.date.strftime("%Y-%m-%d"),
                "tokens": usage.tokens or 0,
                "cost": float(usage.cost or 0)
            }
            for usage in daily_usage
        ],
        "user_usage": [
            {
                "doctor_id": usage.doctor_id,
                "tokens": usage.tokens or 0,
                "cost": float(usage.cost or 0)
            }
            for usage in user_usage
        ]
    } 