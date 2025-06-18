from fastapi import FastAPI
from routers import user, session, health, auth, gpt, sct, admin
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
    setup_middlewares(app)
    setup_events(app)
    app.include_router(user.router)
    app.include_router(session.router)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(gpt.router)
    app.include_router(sct.router)
    app.include_router(admin.router)
    return app 