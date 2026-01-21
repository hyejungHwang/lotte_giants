import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import datetime
import streamlit.components.v1 as components

# -------------------------------------------------------------------------
GA_ID = "G-PNKSFLG8WD" 

ga_js = f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
<script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){{dataLayer.push(arguments);}}
    gtag('js', new Date());

    gtag('config', '{GA_ID}');
</script>
"""


# --------------------------------------------------------------------------------
# 1. 기본 설정 및 "Made by 황오독" 추가
# --------------------------------------------------------------------------------
st.set_page_config(page_title="롯데 자이언츠 승부 예측기", page_icon="⚾", layout="wide")

# [New] 우측 상단 'Made by 황오독' 라벨
st.markdown(
    """
    <style>
    .made-by {
        position: fixed;
        top: 60px;
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

    # 인코더
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
# [수정된 부분] 날짜 입력 코드가 끊기지 않도록 주의하세요!
input_date = st.sidebar.date_input("경기 날짜", value=datetime.date.today())
input_month = input_date.month


pitcher_list = sorted(raw_df['우리팀 선발'].dropna().unique().tolist())

input_starter = st.sidebar.selectbox("우리 팀 선발", pitcher_list, index=0)
input_opponent = st.sidebar.selectbox("상대 팀", le_opp.classes_)
input_home = st.sidebar.radio("경기 장소", ["사직 (홈)", "원정"])
input_uniform = st.sidebar.selectbox("유니폼", le_uni.classes_)
input_opp_foreign = st.sidebar.checkbox("상대 선발 외국인?", value=False)

st.sidebar.header("2. 팀 컨디션")
input_momentum = st.sidebar.slider("최근 5경기 승률", 0.0, 1.0, 0.5)
input_streak = st.sidebar.number_input("연승/연패", value=0)
input_rest = st.sidebar.number_input("휴식일", value=1, min_value=0)
input_games_7d = st.sidebar.slider("최근 7일 경기수", 0, 7, 6)

# --------------------------------------------------------------------------------
# 3. 메인 화면 (예측 결과)
# --------------------------------------------------------------------------------
st.title(f"⚾ 롯데 자이언츠 승부 예측 AI")
st.markdown(f"### {input_month}월의 승부를 예측합니다!")

# 데이터 변환
code_starter = le_starter.transform([input_starter])[0]
code_opp = le_opp.transform([input_opponent])[0]
code_uni = le_uni.transform([input_uniform])[0]
val_is_home = 1 if "홈" in input_home else 0
val_foreign_opp = 1 if input_opp_foreign else 0
val_travel = 0 if val_is_home else 200

# 통계치 자동 계산
avg_h2h = raw_df[raw_df['Opponent_Code'] == code_opp]['상대 전적 승률'].mean()
if np.isnan(avg_h2h): avg_h2h = 0.5
avg_season = raw_df['시즌 누적 승률'].mean()
avg_venue = raw_df[raw_df['Is_Home'] == val_is_home]['홈/원정 구분 승률'].mean()
if np.isnan(avg_venue): avg_venue = 0.5

# 입력 데이터 생성
input_data = pd.DataFrame([[
    code_opp, val_foreign_opp, code_uni, val_is_home,
    input_momentum, input_rest, val_travel, avg_h2h, avg_season,
    input_streak, avg_venue, input_games_7d, code_starter, input_month
]], columns=[
    'Opponent_Code', 'Foreign_Opp_Pitcher', 'Uniform_Code', 'Is_Home',
    '최근5경기승률', '휴식기간', '이동거리', '상대 전적 승률', '시즌 누적 승률',
    '연승/연패', '홈/원정 구분 승률', '최근 7일 경기수', 'Starter_Code', 'Month'
])

if st.button("🔮 승부 예측하기", type="primary"):
    # 1. 떠나간 선수들 명단 (이스터 에그용)
    missing_players = ['감보아', '데이비슨', '반즈', '벨라스케즈']
    
    # 2. 떠난 선수
    if input_starter in missing_players:
        st.markdown(f"""
        <div style='background-color: #F0F2F6; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #888;'>
            <h3 style='color: #555; margin: 0;'>🍂 "만약 {input_starter} 선수가 있었더라면..."</h3>
            <p style='color: #666; font-size: 16px; margin-top: 5px;'>
                지금은 볼 수 없지만, 그가 마운드에 올랐다고 가정한다면?
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    prob = model.predict_proba(input_data)[0][1]
    
    st.divider()
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.subheader("승리 확률")
        st.markdown(f"<h1 style='font-size: 50px; color: #D00F31;'>{prob*100:.1f}%</h1>", unsafe_allow_html=True)
        if prob >= 0.6:
            st.success("승리 유력! (치킨각 🍗)")
        elif prob >= 0.4:
            st.warning("예측불허 접전! (직관 필요)")
        else:
            st.error("고전 예상... (마음의 준비)")
            
    with c2:
        st.subheader("승부처 분석")
        month_msg = "평이한 계절"
        if input_month in [3, 4, 5]: month_msg = "🌸 봄데 버프 (승률↑)"
        elif input_month in [7, 8]: month_msg = "☀️ 한여름 체력 저하 (승률↓)"
        
        st.write(f"📅 **계절:** {month_msg}")
        st.write(f"🏟️ **장소:** {'홈 어드밴티지' if val_is_home else '원정 불리함'}")
        st.write(f"💪 **선발:** {input_starter}")
        
        # 중요도 그래프
        fig, ax = plt.subplots(figsize=(6, 2))
        factors = ['계절', '홈/원정', '선발투수']
        v_month = 60 if input_month in [3,4,5] else (40 if input_month in [7,8] else 50)
        v_home = 80 if val_is_home else 30
        v_start = 85 if input_starter in ['반즈','박세웅'] else 50
        
        ax.barh(factors, [v_month, v_home, v_start], color=['green', 'blue', 'red'])
        ax.set_xlim(0, 100)
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>Designed by 황오독</div>", unsafe_allow_html=True)
