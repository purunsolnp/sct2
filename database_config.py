# database_config.py - 데이터베이스 연결 설정 개선

import os
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import ssl

# 환경 변수에서 데이터베이스 URL 가져오기
DATABASE_URL = os.getenv("DATABASE_URL")

# Render에서 제공하는 기본 PostgreSQL URL 형식 처리
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# SSL 설정 및 연결 풀 설정
def create_database_engine():
    if not DATABASE_URL:
        print("⚠️ DATABASE_URL이 설정되지 않았습니다. SQLite로 폴백합니다.")
        # SQLite 폴백 (개발용)
        return create_engine(
            "sqlite:///./sct_app.db",
            connect_args={"check_same_thread": False},
            echo=False
        )
    
    # PostgreSQL 연결 설정
    connect_args = {
        "sslmode": "require",  # SSL 연결 강제
        "sslcert": None,
        "sslkey": None,
        "sslrootcert": None,
        "connect_timeout": 30,  # 연결 타임아웃 30초
    }
    
    try:
        engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # 연결 확인
            pool_recycle=300,    # 5분마다 연결 재생성
            connect_args=connect_args,
            echo=False
        )
        
        # 연결 테스트
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        
        print("✅ PostgreSQL 데이터베이스 연결 성공")
        return engine
        
    except Exception as e:
        print(f"❌ PostgreSQL 연결 실패: {e}")
        print("⚠️ SQLite로 폴백합니다.")
        
        # SQLite 폴백
        return create_engine(
            "sqlite:///./sct_app.db",
            connect_args={"check_same_thread": False},
            echo=False
        )

# 데이터베이스 엔진 생성
engine = create_database_engine()

# 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스
Base = declarative_base()

# 의존성 주입용 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 테이블 생성 함수
def create_tables():
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ 데이터베이스 테이블 생성 완료")
    except Exception as e:
        print(f"❌ 테이블 생성 실패: {e}")

# 헬스체크 함수
def check_database_health():
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": f"error: {str(e)}"}