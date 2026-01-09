import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import requests # 피드백 전송용

# --------------------------------------------------------------------------------
# 1. 기본 설정 및 "Made by 황오독" 추가
# --------------------------------------------------------------------------------
st.set_page_config(page_title="롯데 자이언츠 승부 예측기", page_icon="⚾", layout="wide")

# [New] 우측 상단 'Made by 황오독' 라벨 (HTML/CSS 활용)
st.markdown(
    """
    <style>
    .made-by {
        position: fixed;
        top: 60px; /* 스트림릿 기본 헤더 아래 위치 */
        right: 20px;
        font-size: 14px;
        font-weight: bold;
        color: #888888;
        background-color: rgba(255, 255, 255, 0.7);
        padding: 5px 10px;
        border-radius: 10px;
        z-index: 9999;
    }
    </style>
    <div class="made-by">Made by 황오독</div>
    """,
    unsafe_allow_html=True
)

# 한글 폰트 설정
import os
if os.name == 'posix': # 리눅스(배포환경)
    plt.rcParams['font.family'] = 'NanumGothic'
else: # 윈도우(로컬환경)
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


@st.cache_data
def load_and_train_model():
    # 데이터 로드
    try:
        df = pd.read_csv('롯데징크스v6.csv')
    except:
        return None, None, None, None, None, None

    # 전처리
    df = df[df['결과'] != '우천취소'].dropna(subset=['결과'])
    df['Target'] = df['결과'].apply(lambda x: 1 if x == '승' else 0)
    
    # 시계열(월) 정보 추출
    df['Date'] = pd.to_datetime(df['일자'])
    df['Month'] = df['Date'].dt.month

    # 결측치 처리
    df['우리팀 선발'] = df['우리팀 선발'].fillna('Unknown')
    df['유니폼'] = df['유니폼'].fillna('Unknown')

    # 인코딩
    le_starter = LabelEncoder()
    df['Starter_Code'] = le_starter.fit_transform(df['우리팀 선발'])
    
    le_opp = LabelEncoder()
    df['Opponent_Code'] = le_opp.fit_transform(df['상대팀'])
    
    le_uni = LabelEncoder()
    df['Uniform_Code'] = le_uni.fit_transform(df['유니폼'])

    # 학습 변수
    df['Foreign_Opp_Pitcher'] = df['상대팀 선발 투수(외국인)'].apply(lambda x: 1 if x == 'O' else 0)
    df['Is_Home'] = df['홈구장'].apply(lambda x: 1 if x == 'O' else 0)
    
    features = [
        'Opponent_Code', 'Foreign_Opp_Pitcher', 'Uniform_Code', 'Is_Home',
        '최근5경기승률', '휴식기간', '이동거리', '상대 전적 승률', '시즌 누적 승률',
        '연승/연패', '홈/원정 구분 승률', '최근 7일 경기수', 'Starter_Code', 'Month'
    ]
    
    X = df[features].fillna(0)
    y = df['Target']

    # 검증 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return model, le_starter, le_opp, le_uni, df, acc

# 모델 로드 실행
model, le_starter, le_opp, le_uni, raw_df, model_acc = load_and_train_model()

if model is None:
    st.error("🚨 '롯데징크스v6.csv' 파일이 필요합니다. 폴더에 파일을 넣어주세요.")
    st.stop()


# --------------------------------------------------------------------------------
# 2. 사이드바 (입력 & 피드백)
# --------------------------------------------------------------------------------
st.sidebar.title("⚾ 예측기 컨트롤타워")

st.sidebar.markdown("### 🤖 모델 신뢰도")
st.sidebar.metric(label="검증 정확도(Accuracy)", value=f"{model_acc * 100:.1f}%", delta="Reliable")
st.sidebar.divider()

st.sidebar.header("1. 경기 정보 입력")
input_date = st.sidebar.date_input("경기 날짜", value=pd.to_datetime("2025-04-
