import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm

# ✅ 한글 폰트 설정
plt.rcParams['font.family'] = 'D2Coding'   # 설치된 폰트명 정확히 써야 함
plt.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 깨짐 방지

st.title("📊 데이터 업로드 & 시각화")
uploaded_file = st.file_uploader('csv 또는 excel 파일 업로드', type=['csv', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 데이터 미리보기")
    st.dataframe(df.head())

    # 👉 위기단계 컬럼명 (CSV에 맞게 수정)
    단계컬럼명 = '위기단계'

    if '나이' in df.columns and 단계컬럼명 in df.columns:

        if pd.api.types.is_numeric_dtype(df[단계컬럼명]):
            st.subheader("📈 나이별 위기단계 선 그래프")
            fig, ax = plt.subplots()
            ax.plot(df['나이'], df[단계컬럼명], marker='^', linestyle='-')
            ax.set_xlabel('나이')
            ax.set_ylabel('위기단계')
            ax.set_title('나이별 위기단계')
            st.pyplot(fig)

            st.subheader("📊 나이별 위기단계 막대 그래프")
            fig2, ax2 = plt.subplots()
            ax2.bar(df['나이'], df[단계컬럼명], color='skyblue')
            ax2.set_xlabel('나이')
            ax2.set_ylabel('위기단계')
            ax2.set_title('나이별 위기단계')
            st.pyplot(fig2)

        else:
            # ✅ 순서형 카테고리 정의 (요청하신 단계순서 반영)
            단계순서 = [ "학대의심" , "응급" , "관찰필요" , "상담필요" , "정상군" ]
            df[단계컬럼명] = pd.Categorical(
                df[단계컬럼명],
                categories=단계순서,
                ordered=True
            )

            st.subheader("📊 나이별 위기단계 빈도 막대 그래프")
            grouped = df.groupby('나이')[단계컬럼명].value_counts().unstack(fill_value=0)
            grouped = grouped[단계순서]  # 순서대로 정렬

            fig3, ax3 = plt.subplots(figsize=(8, 6))
            grouped.plot(kind='bar', stacked=True, ax=ax3)
            ax3.set_xlabel('나이')
            ax3.set_ylabel('빈도')
            ax3.set_title("나이별 위기단계 분포 (막대 그래프)")
            st.pyplot(fig3)

            st.subheader("📈 나이별 위기단계 선 그래프")
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            grouped.plot(kind='line', marker='o', ax=ax4)
            ax4.set_xlabel('나이')
            ax4.set_ylabel('빈도')
            ax4.set_title("나이별 위기단계 분포 (선 그래프)")
            st.pyplot(fig4)

    else:
        st.warning("⚠️ CSV/Excel에 '나이'와 '위기단계' 컬럼이 있어야 합니다.")
