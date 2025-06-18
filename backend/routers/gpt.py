from fastapi import APIRouter

router = APIRouter()

@router.post("/gpt/interpret")
def gpt_interpret():
    # 실제 GPT 해석 로직은 여기에 구현
    return {"result": "interpretation result"} 