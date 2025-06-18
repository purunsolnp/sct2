def setup_events(app):
    @app.on_event("startup")
    async def startup_event():
        # 초기화 작업 (예: DB 연결 확인, 캐시 초기화 등)
        pass

    @app.on_event("shutdown")
    async def shutdown_event():
        # 종료 작업 (예: 리소스 정리 등)
        pass 