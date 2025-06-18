from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas import UserCreate, TokenResponse
import crud
from database_config import get_db

router = APIRouter()

@router.post("/users", response_model=TokenResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = crud.create_user(db, user)
    # 토큰 생성 등 추가 로직
    return TokenResponse(access_token="dummy", token_type="bearer", user_info={"doctor_id": db_user.doctor_id}) 