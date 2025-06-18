from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from schemas import SessionCreate
import crud
from database_config import get_db

router = APIRouter()

@router.post("/sct/sessions", response_model=SessionCreate)
def create_session(session: SessionCreate, db: Session = Depends(get_db)):
    # TODO: 실제 사용자 인증 로직 추가 필요
    doctor_id = "temp_doctor"  # 임시 값, 실제로는 인증된 사용자에서 가져와야 함
    db_session = crud.create_session(db, session, doctor_id)
    return db_session 