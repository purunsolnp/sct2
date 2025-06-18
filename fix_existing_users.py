#!/usr/bin/env python3
"""
기존 사용자 계정들의 비밀번호를 수정하는 스크립트
"""

import requests
import json

def fix_existing_users():
    """기존 사용자 계정들의 비밀번호를 수정합니다."""
    
    base_url = "https://sct-backend-7epf.onrender.com"
    
    # 기존 계정들의 정보 (비밀번호를 다시 설정)
    existing_users = [
        {
            "doctor_id": "doctor1",
            "email": "doctor1@example.com",
            "password": "password123",
            "first_name": "김",
            "last_name": "의사",
            "specialty": "정신건강의학과",
            "hospital": "서울대학교병원",
            "phone": "010-1234-5678",
            "medical_license": "12345"
        },
        {
            "doctor_id": "doctor2", 
            "email": "doctor2@example.com",
            "password": "password123",
            "first_name": "이",
            "last_name": "의사",
            "specialty": "정신건강의학과",
            "hospital": "연세대학교병원",
            "phone": "010-2345-6789",
            "medical_license": "23456"
        },
        {
            "doctor_id": "admin",
            "email": "admin@example.com", 
            "password": "admin123",
            "first_name": "관리",
            "last_name": "자",
            "specialty": "시스템관리",
            "hospital": "SCT 시스템",
            "phone": "010-3456-7890",
            "medical_license": "admin"
        }
    ]
    
    print("🔧 기존 사용자 계정 비밀번호 수정 시작...")
    print(f"서버 주소: {base_url}")
    print("-" * 50)
    
    # 헬스체크 먼저
    try:
        health_response = requests.get(f"{base_url}/health")
        print(f"✅ 헬스체크 성공: {health_response.status_code}")
    except Exception as e:
        print(f"❌ 헬스체크 실패: {e}")
        return
    
    print("-" * 50)
    
    # 각 계정에 대해 로그인 시도 후 실패하면 재등록
    for user_data in existing_users:
        print(f"\n🧪 {user_data['doctor_id']} 계정 확인 중...")
        
        # 먼저 로그인 시도
        try:
            login_response = requests.post(
                f"{base_url}/auth/login",
                headers={"Content-Type": "application/json"},
                json={
                    "doctor_id": user_data["doctor_id"],
                    "password": user_data["password"]
                }
            )
            
            if login_response.status_code == 200:
                print(f"✅ {user_data['doctor_id']} 로그인 성공 - 비밀번호 정상")
                continue
            else:
                print(f"❌ {user_data['doctor_id']} 로그인 실패 - 비밀번호 수정 필요")
                
        except Exception as e:
            print(f"❌ 로그인 요청 오류: {e}")
            continue
        
        # 로그인 실패시 계정 재등록 시도
        print(f"🔄 {user_data['doctor_id']} 계정 재등록 시도...")
        
        try:
            register_response = requests.post(
                f"{base_url}/auth/register",
                headers={"Content-Type": "application/json"},
                json=user_data
            )
            
            if register_response.status_code == 200:
                print(f"✅ {user_data['doctor_id']} 계정 재등록 성공!")
            elif register_response.status_code == 400:
                error_data = register_response.json()
                if "이미 존재하는 사용자입니다" in error_data.get('detail', ''):
                    print(f"⚠️ {user_data['doctor_id']} 계정이 이미 존재합니다")
                    # 기존 계정 삭제 후 재생성하는 로직이 필요할 수 있음
                else:
                    print(f"❌ 계정 재등록 실패: {error_data}")
            else:
                print(f"❌ 계정 재등록 실패: {register_response.status_code}")
                print(f"응답: {register_response.text}")
                
        except Exception as e:
            print(f"❌ 재등록 요청 오류: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 계정 수정 완료!")
    
    # 최종 로그인 테스트
    print("\n🔍 최종 로그인 테스트...")
    print("-" * 30)
    
    test_accounts = [
        {"doctor_id": "doctor1", "password": "password123"},
        {"doctor_id": "doctor2", "password": "password123"},
        {"doctor_id": "admin", "password": "admin123"}
    ]
    
    for account in test_accounts:
        print(f"\n🧪 {account['doctor_id']} 로그인 테스트...")
        
        try:
            response = requests.post(
                f"{base_url}/auth/login",
                headers={"Content-Type": "application/json"},
                json=account
            )
            
            if response.status_code == 200:
                print(f"✅ 로그인 성공!")
            else:
                print(f"❌ 로그인 실패: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 요청 오류: {e}")

if __name__ == "__main__":
    fix_existing_users() 