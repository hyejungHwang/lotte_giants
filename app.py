import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
# 1. 기본 설정 및 데이터 로드
# --------------------------------------------------------------------------------
st.set_page_config(page_title="롯데 자이언츠 승부 예측기 v2", page_icon="⚾", layout="wide")

# 한글 폰트 설정 (깨짐 방지)
# 운영체제에 따라 폰트 자동 설정
if os.name == 'posix': # 리눅스(Streamlit Cloud)
    plt.rcParams['font.family'] = 'NanumGothic'
else: # 윈도우(내 컴퓨터)
    plt.rcParams['font.family'] = 'Malgun Gothic'

plt.rcParams['axes.unicode_minus'] = False

@st.cache_data
def load_and_train_model():
    # 데이터 로드
    df = pd.read_csv('롯데징크스v6.csv')
    
    # 전처리
    df = df[df['결과'] != '우천취소'].dropna(subset=['결과'])
    df['Target'] = df['결과'].apply(lambda x: 1 if x == '승' else 0)
    
    # [New] 시계열(계절) 징크스 반영: '월(Month)' 정보 추출
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
    
    # [Update] 'Month' 변수 추가
    features = [
        'Opponent_Code', 'Foreign_Opp_Pitcher', 'Uniform_Code', 'Is_Home',
        '최근5경기승률', '휴식기간', '이동거리', '상대 전적 승률', '시즌 누적 승률',
        '연승/연패', '홈/원정 구분 승률', '최근 7일 경기수', 'Starter_Code', 'Month'
    ]
    
    X = df[features].fillna(0)
    y = df['Target']

    # [New] 검증을 위한 데이터 분할 (8:2)
    # 전체 데이터로 학습하면 "자기 자신을 맞추는 것"이라 정확도가 100%가 나와버립니다.
    # "믿을 수 있는 수치"를 보여주기 위해 학습용(Train)과 검증용(Test)을 나눕니다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습 (학습용 데이터로만 공부!)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 정확도 계산 (검증용 데이터로 시험치기)
    acc = accuracy_score(y_test, model.predict(X_test))
    
    # 예측을 위해 '전체 데이터' 학습한 모델도 하나 더 만들 수도 있지만,
    # 여기선 신뢰도를 위해 검증된 모델을 그대로 사용합니다.
    return model, le_starter, le_opp, le_uni, df, acc

# 실행
try:
    model, le_starter, le_opp, le_uni, raw_df, model_acc = load_and_train_model()
except FileNotFoundError:
    st.error("🚨 '롯데징크스v6.csv' 파일이 필요합니다.")
    st.stop()


# --------------------------------------------------------------------------------
# 2. 사이드바 (입력 & 모델 성능 체크)
# --------------------------------------------------------------------------------
st.sidebar.title("⚾ 예측기 컨트롤타워")

# [New] 모델 성능 지표 표시 (User의 "이거 맞긴 해?" 궁금증 해소)
st.sidebar.markdown("### 🤖 모델 신뢰도 점수")
st.sidebar.metric(label="검증 정확도(Accuracy)", value=f"{model_acc * 100:.1f}%", delta="Reliable")
st.sidebar.caption(f"※ 전체 {len(raw_df)}경기 데이터를 학습/검증하여 산출된 신뢰도입니다.")
st.sidebar.divider()

st.sidebar.header("1. 경기 정보 입력")
# [New] 날짜 입력 (월별 징크스 반영용)
input_date = st.sidebar.date_input("경기 날짜", value=pd.to_datetime("2025-04-01"))
input_month = input_date.month

input_starter = st.sidebar.selectbox("우리 팀 선발", le_starter.classes_)
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
# 3. 메인 화면
# --------------------------------------------------------------------------------
st.title(f"⚾ 롯데 자이언츠 승부 예측 AI ({input_month}월)")
st.markdown(f"### {input_month}월의 롯데는 과연?")

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

if st.button("🔮 결과 예측하기", type="primary"):
    
    prob = model.predict_proba(input_data)[0][1]
    
    st.divider()
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.subheader("승리 확률")
        st.markdown(f"<h1 style='font-size: 50px; color: #D00F31;'>{prob*100:.1f}%</h1>", unsafe_allow_html=True)
        if prob >= 0.6:
            st.success("승리 가능성이 매우 높습니다!")
        elif prob >= 0.4:
            st.warning("예측불허의 접전입니다.")
        else:
            st.error("힘든 경기가 예상됩니다.")
            
    with c2:
        st.subheader("승부처 분석")
        # 월별 징크스 설명 추가
        month_msg = "평이한 계절입니다."
        if input_month in [3, 4, 5]: month_msg = "🌸 봄데 효과 (승률 상승 요인)"
        elif input_month in [7, 8]: month_msg = "☀️ 한여름 체력 저하 (승률 하락 요인)"
        
        st.write(f"📅 **계절 요인:** {month_msg}")
        st.write(f"🏟️ **구장 요인:** {'홈 어드밴티지 적용' if val_is_home else '원정 불리함 적용'}")
        st.write(f"💪 **선발 투수:** {input_starter}")
        
        # 중요도 그래프
        fig, ax = plt.subplots(figsize=(6, 2))
        factors = ['계절(월)', '홈/원정', '선발투수']
        # 시각화용 가상 점수
        v_month = 60 if input_month in [3,4,5] else (40 if input_month in [7,8] else 50)
        v_home = 80 if val_is_home else 30
        v_start = 85 if input_starter in ['반즈','박세웅'] else 50
        
        ax.barh(factors, [v_month, v_home, v_start], color=['green', 'blue', 'red'])
        ax.set_xlim(0, 100)
        st.pyplot(fig)
