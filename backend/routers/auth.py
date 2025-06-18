from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas import UserLogin, TokenResponse, UserCreate
import crud
from database_config import get_db
from auth_utils import verify_password, create_access_token, get_password_hash, get_current_user
from datetime import timedelta
from pydantic import BaseModel

router = APIRouter(prefix="/auth")

class PasswordChange(BaseModel):
    current_password: str
    new_password: str

@router.get("/check-id/{doctor_id}")
def check_id_duplicate(doctor_id: str, db: Session = Depends(get_db)):
    user = crud.get_user_by_doctor_id(db, doctor_id)
    return {"exists": user is not None}

@router.post("/login", response_model=TokenResponse)
def login(user: UserLogin, db: Session = Depends(get_db)):
    """사용자 로그인"""
    db_user = crud.get_user_by_doctor_id(db, user.doctor_id)
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # JWT 토큰 생성
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": db_user.doctor_id}, expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_info={
            "doctor_id": db_user.doctor_id,
            "email": db_user.email,
            "first_name": db_user.first_name,
            "last_name": db_user.last_name
        }
    )

@router.post("/register", response_model=TokenResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """사용자 회원가입"""
    # 기존 사용자 확인
    existing_user = crud.get_user_by_doctor_id(db, user.doctor_id)
    if existing_user:
        raise HTTPException(status_code=400, detail="이미 존재하는 사용자입니다")
    
    # 비밀번호 해싱
    hashed_password = get_password_hash(user.password)
    
    # 사용자 생성 (비밀번호 해싱된 버전으로)
    user_data = user.dict()
    user_data["password"] = hashed_password
    
    db_user = crud.create_user(db, UserCreate(**user_data))
    
    # JWT 토큰 생성
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": db_user.doctor_id}, expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_info={
            "doctor_id": db_user.doctor_id,
            "email": db_user.email,
            "first_name": db_user.first_name,
            "last_name": db_user.last_name
        }
    )

@router.post("/change-password")
def change_password(
    password_data: PasswordChange,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """비밀번호 변경"""
    # 현재 비밀번호 확인
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(status_code=401, detail="현재 비밀번호가 일치하지 않습니다")
    
    # 새 비밀번호 해싱
    new_hashed_password = get_password_hash(password_data.new_password)
    
    # 비밀번호 업데이트
    current_user.hashed_password = new_hashed_password
    db.commit()
    
    return {"message": "비밀번호가 성공적으로 변경되었습니다"} 