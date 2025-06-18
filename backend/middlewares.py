from fastapi.middleware.cors import CORSMiddleware

def setup_middlewares(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5500",
            "http://127.0.0.1:5500",
            "https://purunsolsct.netlify.app",
            "https://psysct.netlify.app",
            "https://sct-backend-7epf.onrender.com"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 기타 미들웨어 추가 가능 