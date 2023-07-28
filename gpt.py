import streamlit as st
import openai

# Configuração das credenciais do OpenAI GPT-3
openai.api_key = "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"


def chat_with_gpt3(prompt, max_tokens=100):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Substitua pelo modelo gpt-3.5-turbo
        prompt=prompt,
        max_tokens=max_tokens,
    )
    return response['choices'][0]['text']


def main():
    # CSS personalizado
    custom_css = """
    <style>
        body {
            background-color: #282C34;  # Substitua com a cor que você deseja
        }
        .stTextInput .stTextInput>div>div>input {
            background-color: #282C34;  # Substitua com a cor que você deseja
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    st.markdown("### Assistente de Análises de Negócios")
    st.write("Bem-vindo ao seu assistente de análises de negócios. Você pode me perguntar sobre várias métricas e análises de negócios, como vendas, lucros, desempenho do produto, análises de mercado e muito mais.")

    user_input = st.text_area("Digite sua pergunta aqui. Por exemplo: 'Como foram nossas vendas no último trimestre?'", value="", height=100,)

    if st.button("Enviar"):
        if user_input:
            st.write("Você:", user_input)
            response = chat_with_gpt3(user_input)
            st.write("ChatGPT:", response)

   

if __name__ == "__main__":
    main()
