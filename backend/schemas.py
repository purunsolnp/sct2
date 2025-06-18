from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class UserCreate(BaseModel):
    doctor_id: str
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    specialty: Optional[str] = None
    hospital: Optional[str] = None
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

class SessionRead(BaseModel):
    session_id: str
    patient_name: str
    doctor_id: str
    status: str
    created_at: datetime
    submitted_at: Optional[datetime] = None
    expires_at: datetime
    
    class Config:
        from_attributes = True  # SQLAlchemy 모델과 호환
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }

class SCTResponseCreate(BaseModel):
    item_no: int
    stem: str
    answer: str

class SCTResponse(BaseModel):
    item_no: int
    stem: str
    answer: str

class SCTSessionResponse(BaseModel):
    session_id: str
    doctor_id: str
    patient_name: str
    status: str
    created_at: datetime
    submitted_at: Optional[datetime]
    expires_at: datetime

class UserStatusUpdate(BaseModel):
    is_active: bool

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str 