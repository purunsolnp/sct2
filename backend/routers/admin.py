from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database_config import get_db
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import crud
from auth_utils import get_current_admin_user
from models import User, LoginAttempt, SystemSettings

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_current_admin_user)]
)

@router.get("/stats")
def get_admin_stats(db: Session = Depends(get_db)):
    """관리자 대시보드의 통계 데이터를 반환합니다."""
    # 사용자 통계
    user_stats = crud.get_user_stats(db)
    
    # 세션 통계
    session_stats = crud.get_session_stats(db)
    
    # 로그인 시도 통계 (최근 24시간)
    yesterday = datetime.now() - timedelta(days=1)
    login_attempts = db.query(LoginAttempt)\
        .filter(LoginAttempt.attempt_time >= yesterday)\
        .order_by(LoginAttempt.attempt_time.desc())\
        .all()
    
    login_stats = [
        {
            "timestamp": attempt.attempt_time.isoformat(),
            "ip": attempt.ip_address,
            "success": attempt.success,
            "username": attempt.username
        }
        for attempt in login_attempts
    ]
    
    # GPT 사용량 통계 (최근 30일)
    thirty_days_ago = datetime.now() - timedelta(days=30)
    gpt_stats = crud.get_gpt_usage_stats(db, thirty_days_ago, datetime.now())
    
    # 월별 통계 (최근 12개월)
    monthly_stats = crud.get_monthly_stats(db)
    
    return {
        "user_stats": user_stats,
        "session_stats": session_stats,
        "login_attempts": login_stats,
        "gpt_usage": gpt_stats,
        "monthly_stats": monthly_stats
    }

@router.get("/users")
def get_users(db: Session = Depends(get_db)):
    """모든 사용자 목록을 반환합니다."""
    users = crud.get_all_users(db)
    return [
        {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
        for user in users
    ]

@router.get("/login-attempts")
def get_login_attempts(
    days: Optional[int] = 1,
    db: Session = Depends(get_db)
):
    """최근 로그인 시도 기록을 반환합니다."""
    start_date = datetime.now() - timedelta(days=days)
    
    attempts = db.query(LoginAttempt)\
        .filter(LoginAttempt.attempt_time >= start_date)\
        .order_by(LoginAttempt.attempt_time.desc())\
        .all()
    
    return [
        {
            "timestamp": attempt.attempt_time.isoformat(),
            "ip": attempt.ip_address,
            "success": attempt.success,
            "username": attempt.username
        }
        for attempt in attempts
    ]

@router.get("/settings")
def get_settings(db: Session = Depends(get_db)):
    """시스템 설정을 반환합니다."""
    settings = db.query(SystemSettings).first()
    if not settings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="System settings not found"
        )
    
    return {
        "max_tokens_per_request": settings.max_tokens_per_request,
        "max_tokens_per_day": settings.max_tokens_per_day,
        "max_tokens_per_month": settings.max_tokens_per_month,
        "cost_per_token": float(settings.cost_per_token),
        "session_timeout_minutes": settings.session_timeout_minutes,
        "max_login_attempts": settings.max_login_attempts,
        "ip_block_duration_minutes": settings.ip_block_duration_minutes,
        "updated_at": settings.updated_at.isoformat() if settings.updated_at else None
    }

@router.put("/settings")
def update_settings(
    settings: dict,
    db: Session = Depends(get_db)
):
    """시스템 설정을 업데이트합니다."""
    db_settings = db.query(SystemSettings).first()
    if not db_settings:
        db_settings = SystemSettings()
        db.add(db_settings)
    
    # 설정 업데이트
    for key, value in settings.items():
        if hasattr(db_settings, key):
            setattr(db_settings, key, value)
    
    db_settings.updated_at = datetime.now()
    db.commit()
    db.refresh(db_settings)
    
    return {
        "message": "Settings updated successfully",
        "settings": {
            "max_tokens_per_request": db_settings.max_tokens_per_request,
            "max_tokens_per_day": db_settings.max_tokens_per_day,
            "max_tokens_per_month": db_settings.max_tokens_per_month,
            "cost_per_token": float(db_settings.cost_per_token),
            "session_timeout_minutes": db_settings.session_timeout_minutes,
            "max_login_attempts": db_settings.max_login_attempts,
            "ip_block_duration_minutes": db_settings.ip_block_duration_minutes,
            "updated_at": db_settings.updated_at.isoformat() if db_settings.updated_at else None
        }
    }

@router.get("/usage-stats")
def get_usage_stats(months: int = 12, db: Session = Depends(get_db)):
    """월별 사용 통계를 반환합니다."""
    # 현재 월부터 이전 months개월의 통계 생성
    now = datetime.now()
    monthly_stats = []
    
    for i in range(months):
        date = now - timedelta(days=30*i)
        monthly_stats.append({
            "month_name": date.strftime("%Y-%m"),
            "total_sessions": 0,
            "completed_sessions": 0,
            "total_tokens": 0,
            "total_cost": 0
        })
    
    return {
        "monthly_stats": monthly_stats,
        "total": {
            "sessions": 0,
            "completed": 0,
            "tokens": 0,
            "cost": 0
        }
    }

@router.get("/ip-blocks")
def get_ip_blocks(db: Session = Depends(get_db)):
    """차단된 IP 목록을 반환합니다."""
    return {
        "ip_blocks": [],
        "total": 0
    }

@router.get("/dashboard/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    """대시보드 통계를 반환합니다."""
    return {
        "total_users": 0,
        "total_sessions": 0,
        "this_month_sessions": 0,
        "pending_sessions": 0,
        "active_users": 0,
        "completion_rate": 0,
        "this_month_completed": 0,
        "expired_sessions": 0
    }

@router.get("/gpt-usage")
def get_gpt_usage(start_date: str, end_date: str, db: Session = Depends(get_db)):
    """GPT 사용량 통계를 반환합니다."""
    return {
        "total_usage": {
            "total_tokens": 0,
            "total_cost": 0.0
        },
        "daily_usage": [],
        "user_usage": []
    } 