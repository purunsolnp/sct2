from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database_config import get_db
from auth_utils import get_current_user
import openai
import os
import json
from typing import List, Dict, Any
from models import SCTSession, SCTResponse, GPTTokenUsage
from crud import get_session_by_id, get_responses_by_session_id
import logging

# 로거 설정
logger = logging.getLogger(__name__)

router = APIRouter()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# SCT 해석 프롬프트 템플릿
SCT_INTERPRETATION_PROMPT = """당신은 숙련된 임상심리사입니다. 아래의 SCT(문장완성검사) 응답을 바탕으로 임상 해석 보고서를 작성하세요.

[중요 지침]
- 마크다운 헤더(`#`, `##`, `###`, `####` 등)는 절대 사용하지 마세요.
- 각 섹션 제목은 반드시 숫자와 점으로 시작하고, 볼드(굵은 글씨)는 `**`로 감싸서 작성하세요. (예: **1. 검사 개요**)
- 환자를 지칭할 때는 반드시 '{patient_name}님'처럼 이름 뒤에 '님'을 붙여서만 지칭하세요. '그', '그녀', '환자', '내담자' 등으로 지칭하지 마세요.
- 줄바꿈(`\\n\\n`)과 볼드(`**`)만 사용하세요. 리스트, 표, 헤더, 이탤릭 등 다른 마크다운 문법은 사용하지 마세요.
- 각 번호(1, 2, 3...)는 반드시 한 줄 띄우고, 볼드(굵은 글씨)로 작성하세요.
- 각 소제목(2.1, 2.2 등)도 볼드로 감싸세요.
- 각 항목은 반드시 400자 이상으로 작성하세요. 분석은 진단 가능성과 방어기제 수준, 대인 기능, 자아 강도, 성격특성 등을 포함하여 실제 임상에 쓸 수 있을 정도의 깊이로 기술하세요.
- 단순한 문장 요약이 아니라, 해당 응답이 어떤 방어기제(defense), 성격 특성(personality trait), 임상증상(psychopathology)과 연결되는지를 명확히 해석해 주세요.
- 가능한 경우 DSM-5 진단 기준, Vaillant 방어기제 분류, 성격 스펙트럼 개념과 연계해 주세요.
- 응답의 진실성(진단 신뢰도)이나 과장/저반응 가능성에 대한 평가도 포함해 주세요.
- 보고서 말미에는 치료 예후, 치료적 제휴 가능성, 강점과 취약성을 구분해서 서술하고, 재평가 시점과 그 이유도 포함하세요.

다음의 보고서 구조를 반드시 따르세요:

**1. 검사 개요**  
{patient_name}님, 검사일, 검사 협조도, 응답의 전반적 특성, 응답 스타일, 검사 신뢰도 등을 요약해 주세요. 정서의 깊이, 문장 구조의 성실성, 회피나 과장 여부 등도 평가해 주세요.

**2. 주요 심리적 특성 분석**  
**2.1 가족관계 및 애착 패턴**  
**2.2 대인관계 및 사회적 기능**  
**2.3 자아개념 및 정체성**  
**2.4 정서조절 및 스트레스 대처**  
**2.5 성역할 및 이성관계**  
**2.6 미래전망 및 목표지향성**  
**2.7 과거경험 및 현실적응**  
각 항목별로 {patient_name}님의 실제 응답을 구체적으로 인용하며, 임상적으로 해석해 주세요.

**3. 임상적 평가**  
**3.1 주요 방어기제 및 성격특성**  
Vaillant의 방어기제 분류 체계를 기반으로 주요 방어기제를 평가하고, 성격 구조 및 적응 수준을 분석해 주세요.

**3.2 정신병리학적 고려사항**  
우울, 불안, 자기애, 충동성, 관계 회피, 현실 검증 등 심리적 증상 및 기능 저하와 관련된 병리적 요소를 서술해 주세요. DSM-5 기준과 연결 가능하다면 명시적으로 진단 가설을 제시해 주세요.

**4. 치료적 권고사항**  
**4.1 우선 개입 영역**  
정서조절, 관계 문제, 자아통합 등 임상적 개입 우선순위를 제시해 주세요. 치료 목표는 단기-중기-장기로 구분하여 구체화해 주세요.

**4.2 생활관리 및 지원방안**  
일상생활에서 실천 가능한 정서 안정 전략, 사회적 지원, 자기 구조화 기술 등 생활 차원의 개입을 제안해 주세요.

**5. 종합 해석 및 예후**  
{patient_name}님의 응답 전반에서 드러난 심리 구조, 핵심 정서, 반복되는 심리 주제, 방어기제 수준, 성격 특성 간의 연결성을 종합적으로 통합하여 서술해 주세요.  
이 항목은 단순 요약이 아닌 전체 구조적 해석의 핵심입니다.

**5.1 심리적 강점**  
{patient_name}님의 치료에 긍정적으로 작용할 수 있는 자원, 자기 통찰, 회복 탄력성 등을 구체적으로 기술해 주세요.

**5.2 심리적 취약성**  
{patient_name}님의 정서적 취약 영역, 반복되는 갈등 패턴, 방어 실패 지점, 사회적 적응의 제약 요인을 명확히 제시해 주세요.

**5.3 치료적 제휴 형성 가능성**  
치료자와의 관계 형성 가능성을 예측하고, 관계 유지를 위한 고려 요소를 평가해 주세요.

**5.4 재평가 및 추적 관찰 권고**  
재검 권장 시점(예: 초기 개입 3-6개월 후)과 그 이유(예: 정서 조절 변화, 대인 기능 향상 여부 등)를 구체적으로 기술해 주세요.

아래는 {patient_name}님의 실제 응답입니다.
---
{responses_text}
---
보고서는 반드시 위의 지침을 엄격히 지키고, 임상적으로 깊이 있게, 실제 임상 보고서처럼 작성하세요. 말투는 존댓말로 작성해주세요.  
각 항목별로 소제목을 붙이고, {patient_name}님의 실제 응답을 인용해 해석해 주세요.  
불필요한 반복이나 단순 요약은 피하고, 임상적 통찰과 치료적 제안을 충분히 포함하세요."""

def calculate_gpt_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """GPT 토큰 사용량에 따른 비용을 계산합니다."""
    if model == "gpt-4o":
        return (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
    return 0.0

@router.post("/gpt/interpret/{session_id}")
async def gpt_interpret(
    session_id: str, 
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """SCT 세션에 대한 GPT 해석을 생성합니다."""
    try:
        # 사용자 승인 상태 확인
        if not current_user.is_verified:
            raise HTTPException(
                status_code=403, 
                detail="승인되지 않은 계정입니다. 관리자에게 문의하세요."
            )
        
        # 사용자 활성 상태 확인
        if not current_user.is_active:
            raise HTTPException(
                status_code=403, 
                detail="비활성화된 계정입니다. 관리자에게 문의하세요."
            )
        
        # 세션 조회
        session = get_session_by_id(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        # 세션 소유자 확인
        if session.doctor_id != current_user.doctor_id:
            raise HTTPException(status_code=403, detail="해당 세션에 대한 접근 권한이 없습니다.")
        
        # 응답 조회
        responses = get_responses_by_session_id(db, session_id)
        if not responses:
            raise HTTPException(status_code=404, detail="응답을 찾을 수 없습니다.")
        
        # 응답 텍스트 구성
        responses_text = ""
        for response in responses:
            responses_text += f"\n{response['item_no']}. {response['stem']} → {response['answer']}"
        
        # 프롬프트 구성
        prompt = SCT_INTERPRETATION_PROMPT.format(
            patient_name=session.patient_name,
            responses_text=responses_text
        )
        
        # OpenAI API 호출
        if not openai.api_key:
            raise HTTPException(status_code=500, detail="OpenAI API 키가 설정되지 않았습니다.")
        
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 임상심리학 전문가입니다. SCT 응답을 분석하여 전문적이고 객관적인 해석을 제공해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        interpretation = response.choices[0].message.content
        
        # 해석 결과를 세션에 저장
        session.interpretation = interpretation
        db.commit()
        
        # 토큰 사용량 기록
        usage = response.usage
        model = "gpt-4o"
        cost = calculate_gpt_cost(model, usage.prompt_tokens, usage.completion_tokens)
        
        token_usage = GPTTokenUsage(
            doctor_id=session.doctor_id,
            session_id=session_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model=model,
            cost=cost
        )
        db.add(token_usage)
        db.commit()
        
        logger.info(f"✅ GPT 해석 생성 완료: {usage.total_tokens} 토큰 사용 (${cost})")
        
        return {
            "session_id": session_id,
            "patient_name": session.patient_name,
            "interpretation": interpretation,
            "token_usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "model": model,
                "cost": cost
            }
        }
        
    except Exception as e:
        logger.error(f"❌ GPT 해석 생성 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"해석 생성 중 오류가 발생했습니다: {str(e)}")

@router.get("/gpt/interpret/{session_id}")
async def get_interpretation(
    session_id: str, 
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """기존 해석 결과를 조회합니다."""
    session = get_session_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    
    # 세션 소유자 확인
    if session.doctor_id != current_user.doctor_id:
        raise HTTPException(status_code=403, detail="해당 세션에 대한 접근 권한이 없습니다.")
    
    if not session.interpretation:
        raise HTTPException(status_code=404, detail="해석 결과가 없습니다.")
    
    return {
        "session_id": session_id,
        "patient_name": session.patient_name,
        "interpretation": session.interpretation
    } 