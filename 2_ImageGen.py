import streamlit as st
from openai import OpenAI

st.title('OpenAI 이미지 생성')

api_key = st.text_input('OpenAPI Key', type='password')

if api_key:
    client = OpenAI(api_key=api_key)
    prompt = st.text_input('이미지 프롬프트 입력', '귀여운 고양이와 강아지가 각각의 노트북을 쓰는 그림')

    if st.button('이미지 생성'):
        with st.spinner('생성중...'):
            response = client.images.generate(
                model='dall-e-3',
                prompt=prompt,
                size='1024x1024'
            )
            image_url = response.data[0].url
            st.image(image_url, caption='생성된 이미지')



client.close()
