from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    first_name = Column(String)
    last_name = Column(String)
    specialty = Column(String)
    hospital = Column(String)
    phone = Column(String, nullable=True)
    medical_license = Column(String, nullable=True)
    is_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_password_change = Column(DateTime, default=datetime.utcnow)
    password_history = Column(JSON, default=list)
    login_attempts = Column(Integer, default=0)
    last_login_attempt = Column(DateTime)
    last_login = Column(DateTime)
    is_locked = Column(Boolean, default=False)
    lock_until = Column(DateTime)

class SCTSession(Base):
    __tablename__ = "sct_sessions"
    session_id = Column(String, primary_key=True, index=True)
    doctor_id = Column(String, index=True)
    patient_name = Column(String)
    status = Column(String, default="incomplete")
    created_at = Column(DateTime, default=datetime.utcnow)
    submitted_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime)
    responses = Column(JSON, nullable=True)
    interpretation = Column(Text, nullable=True)

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

class GPTTokenUsage(Base):
    __tablename__ = "gpt_token_usage"
    id = Column(Integer, primary_key=True, autoincrement=True)
    doctor_id = Column(String, index=True)
    session_id = Column(String, index=True)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    model = Column(String)
    cost = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class IPBlock(Base):
    __tablename__ = "ip_blocks"
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String, index=True)
    attempts = Column(Integer, default=0)
    last_attempt = Column(DateTime, default=datetime.utcnow)
    blocked_until = Column(DateTime)
    is_blocked = Column(Boolean, default=False)

class LoginAttempt(Base):
    __tablename__ = "login_attempts"
    id = Column(Integer, primary_key=True, index=True)
    ip_address = Column(String, index=True)
    doctor_id = Column(String, index=True)
    attempt_time = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=False)
    user_agent = Column(String)

class SystemSettings(Base):
    __tablename__ = "system_settings"
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    value = Column(String)
    description = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String) 