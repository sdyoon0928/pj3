import streamlit as st
from openai import OpenAI

st.title('OpenAI 텍스트 생성')

api_key = st.text_input('Open API key', type='password')

if api_key:
    client = OpenAI(api_key=api_key)
    prompt = st.text_area('프롬프트를 입력하세요', '커피에 대한 짧은 글을 작성해줘')

    if st.button('텍스트 생성중'):
        with st.spinner('생성중...'):
            response = client.chat.completions.create(
                model = 'gpt-4o-mini',
                messages=[{'role' : 'user', 'content' : prompt}],
                temperature=.7
            )
            result = response.choices[0].message.content
            st.success('성공')
            st.write(result)