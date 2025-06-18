#!/usr/bin/env python3
"""
원격 서버에 기본 사용자 계정 생성 스크립트
"""

import requests
import json

def create_remote_users():
    """원격 서버에 기본 사용자 계정들을 생성합니다."""
    
    base_url = "https://sct-backend-7epf.onrender.com"
    
    # 기본 사용자 데이터
    default_users = [
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
    
    print("🚀 원격 서버에 기본 사용자 계정 생성 시작...")
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
    
    # 각 계정 생성
    for user_data in default_users:
        print(f"\n🧪 {user_data['doctor_id']} 계정 생성 시도...")
        
        try:
            response = requests.post(
                f"{base_url}/auth/register",
                headers={"Content-Type": "application/json"},
                json=user_data
            )
            
            print(f"상태 코드: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 계정 생성 성공!")
                print(f"토큰: {data.get('access_token', 'N/A')[:20]}...")
            elif response.status_code == 400:
                error_data = response.json()
                if "이미 존재하는 사용자입니다" in error_data.get('detail', ''):
                    print(f"⚠️ 계정이 이미 존재합니다: {user_data['doctor_id']}")
                else:
                    print(f"❌ 계정 생성 실패: {error_data}")
            else:
                print(f"❌ 계정 생성 실패")
                print(f"응답: {response.text}")
                
        except Exception as e:
            print(f"❌ 요청 오류: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 계정 생성 완료!")
    
    # 생성된 계정으로 로그인 테스트
    print("\n🔍 생성된 계정으로 로그인 테스트...")
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
    create_remote_users() 