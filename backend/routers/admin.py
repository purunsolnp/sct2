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
    user_list = [
        {
            "id": user.id,
            "doctor_id": user.doctor_id,
            "email": user.email,
            "is_admin": user.is_admin,
            "is_active": user.is_active,
            "is_verified": user.is_verified,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "name": f"{user.first_name} {user.last_name}".strip(),
            "specialty": user.specialty,
            "hospital": user.hospital,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
        for user in users
    ]
    
    return {"users": user_list}

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
            "username": attempt.doctor_id
        }
        for attempt in attempts
    ]

@router.get("/settings")
def get_settings(db: Session = Depends(get_db)):
    """시스템 설정을 반환합니다."""
    settings = db.query(SystemSettings).all()
    
    # 기본 설정값
    default_settings = {
        "max_tokens_per_request": 4000,
        "max_tokens_per_day": 100000,
        "max_tokens_per_month": 3000000,
        "cost_per_token": 0.0001,
        "session_timeout_minutes": 1440,  # 24시간
        "max_login_attempts": 5,
        "ip_block_duration_minutes": 30
    }
    
    # 데이터베이스에서 설정값 가져오기
    for setting in settings:
        if setting.key in default_settings:
            try:
                if setting.key in ["max_tokens_per_request", "max_tokens_per_day", "max_tokens_per_month", "session_timeout_minutes", "max_login_attempts", "ip_block_duration_minutes"]:
                    default_settings[setting.key] = int(setting.value)
                elif setting.key == "cost_per_token":
                    default_settings[setting.key] = float(setting.value)
            except (ValueError, TypeError):
                pass  # 기본값 유지
    
    return default_settings

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
    monthly_stats = crud.get_monthly_stats(db, months)
    
    # 총계 계산
    total_sessions = sum(stat["total_sessions"] for stat in monthly_stats)
    total_completed = sum(stat["completed_sessions"] for stat in monthly_stats)
    total_tokens = sum(stat["total_tokens"] for stat in monthly_stats)
    total_cost = sum(stat["total_cost"] for stat in monthly_stats)
    
    return {
        "monthly_stats": monthly_stats,
        "total": {
            "sessions": total_sessions,
            "completed": total_completed,
            "tokens": total_tokens,
            "cost": total_cost
        }
    }

@router.get("/ip-blocks")
def get_ip_blocks(db: Session = Depends(get_db)):
    """차단된 IP 목록을 반환합니다."""
    # 실제 IP 블록 데이터가 없으므로 빈 배열 반환
    return []

@router.get("/dashboard/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    """대시보드 통계를 반환합니다."""
    user_stats = crud.get_user_stats(db)
    session_stats = crud.get_session_stats(db)
    
    return {
        "total_users": user_stats["total_users"],
        "active_users": user_stats["active_users"],
        "total_sessions": session_stats["total_sessions"],
        "this_month_sessions": session_stats["this_month_sessions"],
        "pending_sessions": session_stats["pending_sessions"],
        "completion_rate": session_stats["completion_rate"],
        "this_month_completed": session_stats["this_month_completed"],
        "expired_sessions": session_stats["expired_sessions"]
    }

@router.get("/gpt-usage")
def get_gpt_usage(
    start_date: str,
    end_date: str,
    db: Session = Depends(get_db)
):
    """특정 기간의 GPT 사용량 통계를 반환합니다."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        end = end + timedelta(days=1) - timedelta(seconds=1)  # 해당 일의 마지막 시간까지 포함
        
        gpt_stats = crud.get_gpt_usage_stats(db, start, end)
        
        # doctor_stats 형식 수정
        doctor_stats = []
        for user_usage in gpt_stats.get("user_usage", []):
            doctor_stats.append({
                "doctor_id": user_usage.get("doctor_id", ""),
                "usage_count": 1,  # 실제로는 사용 횟수를 계산해야 하지만 임시로 1
                "total_tokens": user_usage.get("tokens", 0),
                "total_cost": user_usage.get("cost", 0.0)
            })
        
        # 프론트엔드가 기대하는 형식으로 반환
        return {
            "total_usage": {
                "total_tokens": gpt_stats.get("total_usage", {}).get("total_tokens", 0),
                "total_cost": gpt_stats.get("total_usage", {}).get("total_cost", 0.0)
            },
            "doctor_stats": doctor_stats,
            "model_stats": [],  # 모델별 통계는 현재 구현되지 않음
            "usage_data": gpt_stats.get("daily_usage", [])
        }
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD"
        ) 