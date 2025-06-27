# SCT 자동 해석 시스템

문장완성검사(Sentence Completion Test)를 디지털화하고 AI를 활용한 자동 해석 서비스를 제공하는 웹 애플리케이션입니다.

## 🚀 주요 기능

- **웹 기반 SCT 검사 실시**: 환자가 온라인으로 문장완성검사를 진행할 수 있습니다
- **OpenAI GPT 기반 자동 해석**: AI를 활용하여 검사 결과를 자동으로 해석합니다
- **의사용 관리 시스템**: 의사가 환자 검사를 관리하고 결과를 확인할 수 있습니다
- **보안 인증 시스템**: JWT 기반 인증 및 보안 기능을 제공합니다
- **관리자 기능**: 시스템 관리 및 사용자 관리를 위한 관리자 인터페이스

## 🏗️ 프로젝트 구조

```
sct2/
├── backend/                 # FastAPI 백엔드 서버
│   ├── routers/            # API 라우터들
│   │   ├── auth.py         # 인증 관련 API
│   │   ├── user.py         # 사용자 관리 API
│   │   ├── sct.py          # SCT 검사 관련 API
│   │   ├── gpt.py          # GPT 해석 API
│   │   ├── admin.py        # 관리자 API
│   │   ├── session.py      # 세션 관리 API
│   │   └── health.py       # 헬스체크 API
│   ├── models.py           # 데이터베이스 모델
│   ├── schemas.py          # Pydantic 스키마
│   ├── crud.py             # 데이터베이스 CRUD 작업
│   ├── auth_utils.py       # 인증 유틸리티
│   ├── main.py             # FastAPI 애플리케이션 진입점
│   ├── requirements.txt    # Python 의존성
│   └── Dockerfile          # Docker 설정
├── frontend/               # 정적 웹 프론트엔드
│   ├── index.html          # 메인 페이지
│   ├── login.html          # 로그인 페이지
│   ├── signup.html         # 회원가입 페이지
│   ├── doctor.html         # 의사용 대시보드
│   ├── patient.html        # 환자용 검사 페이지
│   ├── admin.html          # 관리자 페이지
│   └── netlify.toml        # Netlify 배포 설정
└── alembic/                # 데이터베이스 마이그레이션
```

## 🛠️ 기술 스택

### Backend
- **FastAPI**: 고성능 Python 웹 프레임워크
- **SQLAlchemy**: ORM 및 데이터베이스 관리
- **PostgreSQL**: 메인 데이터베이스
- **Alembic**: 데이터베이스 마이그레이션
- **JWT**: 인증 토큰 관리
- **OpenAI GPT**: AI 기반 해석 서비스

### Frontend
- **HTML5/CSS3/JavaScript**: 정적 웹 인터페이스
- **반응형 디자인**: 모바일 및 데스크톱 지원

## 📋 데이터베이스 모델

### 주요 엔티티
- **User**: 의사 및 관리자 사용자 정보
- **SCTSession**: SCT 검사 세션 정보
- **SCTResponse**: 개별 검사 응답 데이터
- **SCTInterpretation**: AI 해석 결과
- **GPTTokenUsage**: GPT API 사용량 추적
- **IPBlock**: 보안을 위한 IP 차단 관리

## 🚀 설치 및 실행

### 1. 백엔드 설정

```bash
# 백엔드 디렉토리로 이동
cd backend

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일에서 데이터베이스 연결 정보 및 OpenAI API 키 설정

# 데이터베이스 마이그레이션
alembic upgrade head

# 서버 실행
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. 프론트엔드 설정

```bash
# 프론트엔드 디렉토리로 이동
cd frontend

# 정적 파일 서버 실행 (선택사항)
python -m http.server 8080
```

### 3. Docker를 사용한 실행

```bash
# 백엔드 Docker 빌드 및 실행
cd backend
docker build -t sct-backend .
docker run -p 8000:8000 sct-backend
```

## 🔧 환경변수 설정

백엔드 실행을 위해 다음 환경변수들을 설정해야 합니다:

```env
# 데이터베이스 설정
DATABASE_URL=postgresql://username:password@localhost:5432/sct_db

# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key

# JWT 설정
SECRET_KEY=your_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# 보안 설정
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=30
```

## 📚 API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔐 보안 기능

- **JWT 기반 인증**: 안전한 토큰 기반 인증
- **비밀번호 해싱**: bcrypt를 사용한 비밀번호 암호화
- **로그인 시도 제한**: 무차별 대입 공격 방지
- **IP 차단**: 의심스러운 IP 주소 차단
- **세션 관리**: 안전한 세션 생성 및 관리

## 👥 사용자 역할

### 의사 (Doctor)
- 환자 검사 세션 생성 및 관리
- 검사 결과 및 AI 해석 확인
- 환자 정보 관리

### 관리자 (Admin)
- 시스템 전체 관리
- 사용자 계정 관리
- 시스템 설정 관리
- GPT API 사용량 모니터링

### 환자 (Patient)
- 웹 기반 SCT 검사 참여
- 검사 완료 후 결과 확인

## 🤝 기여하기

1. 이 저장소를 포크합니다
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

프로젝트에 대한 질문이나 문제가 있으시면 이슈를 생성해 주세요.

---

**SCT 자동 해석 시스템** - AI 기반 문장완성검사 해석 서비스