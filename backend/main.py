from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app_factory import create_app

app = create_app()

# CORS 미들웨어를 최상위 레벨에서 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)