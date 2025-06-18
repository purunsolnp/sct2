from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import user, health, auth, gpt, sct, admin
from middlewares import setup_middlewares
from events import setup_events

def create_app():
    app = FastAPI(
        title="SCT API",
        description="SCT 검사 시스템을 위한 API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS 미들웨어를 직접 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
    
    setup_middlewares(app)
    setup_events(app)
    app.include_router(user.router)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(gpt.router)
    app.include_router(sct.router)
    app.include_router(admin.router)
    return app 