from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/")
def root():
    return {"message": "SCT 자동 해석 시스템이 정상적으로 작동 중입니다"}

@router.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()} 