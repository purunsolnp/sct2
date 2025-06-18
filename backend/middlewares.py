from fastapi.middleware.cors import CORSMiddleware

def setup_middlewares(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 모든 origin 허용
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
    # 기타 미들웨어 추가 가능 