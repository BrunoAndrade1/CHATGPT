import streamlit as st
from pathlib import Path
from PIL import Image
import pandas as pd
import webbrowser
#$$$$$$$
import openai
openai.api_key = "z3ydK4TGR8FBqpO4P1JxT3BlbkFJrXmWK5QUYZlDdTfDynh1"
import openai
import streamlit as st
from deta import Deta
from geopy.geocoders import Nominatim
# Configurações da página


#$$$$$$$$$$$$ GPT 
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


#$$$$$$$$$$$$$$$

st.set_page_config(
    page_title='Union It',
    page_icon="🚀",
    layout="centered",
)

# Configuraçoes Estruturais #
diretorio = Path(__file__).parent if "__file__" in locals() else Path.cwd()
arquivo_css = diretorio / 'styles'/'geral.css'
arquivo_img = diretorio /'assets'/ 'images_union.png'

# carregando asset
with open(arquivo_css) as c:
    st.markdown('<style>{}</style>'.format(c.read()), unsafe_allow_html=True)

imagem = Image.open(arquivo_img)

# Definindo três colunas para centralizar a imagem
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image(imagem, width=400)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# onfiguração geral das impormaçoes #


def pagina_apresentacao():
    st.markdown("""
    <style>
    .title {
      text-align: center;
    }
    .content {
      text-align: justify;
      margin: 0 auto;
      max-width: 800px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
    .big-title {
        font-size: 30px;
        text-align: left;
        color:#0000FF ;
    }
    .content {
        font-size: 20px;
        color: #0000FF;  /* Azul escuro */
    }
</style>
<div class="big-title">🏢 A Union IT Digital</div>
<div class="content">
🖥 Union IT Digital é uma empresa especializada em soluções de tecnologia da informação e transformação digital, que tem como objetivo ajudar empresas a melhorar a eficiência operacional e alcançar seus objetivos de negócios.

💡 Com uma combinação de conhecimento técnico avançado e experiência prática, nosso time de dados está preparado para abordar desafios complexos de dados. Eles desenvolvem soluções analíticas abrangentes que incluem, gerenciamento de dados, aplicação de técnicas de aprendizado de máquina, modelagem de séries temporais, análise de risco e avaliação da saúde financeira da empresa.

🏆 Cada membro da equipe traz uma gama diversificada de habilidades e proficiências, o que nos permite criar soluções personalizadas para atender às necessidades específicas de cada projeto. Os membros da equipe têm certificações reconhecidas na indústria em plataformas de nuvem e dados, como Azure e Snowflake, garantindo que estamos atualizados com as melhores práticas e tecnologias mais recentes.

🎯 Nós nos esforçamos para fornecer soluções eficientes e eficazes que possam ajudar sua empresa a tomar decisões embasadas em dados. Nosso objetivo é oferecer insights acionáveis que possam levar a melhorias operacionais significativas e a uma vantagem competitiva sustentável.
</div>
""", unsafe_allow_html=True)
################
# Para usar a função de apresentação
pagina_apresentacao()


##########################################################      Mapa
# Crie um dataframe com os dados de latitude e longitude
data = pd.DataFrame({
    'lat': [-23.61211],  # Latitude 
    'lon': [-46.69377]  # Longitude 
})

# Use a função st.map para exibir o mapa com zoom 15
st.map(data, zoom=15)

#st.button('Botão com Tooltip', help='Clique aqui para fazer algo!')

#%%%%%%%%%%%%%%%%%% Link do site
#st.markdown("Visite nosso [site](https://unionitdigital.com.br/ai-ml/)")

if st.button('Visite nosso site'):
    webbrowser.open_new_tab('https://unionitdigital.com.br/ai-ml/')
#$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Chatboot

openai.api_key = st.secrets["openai"]["api_key"]

# Restante do código ...

def chat_with_gpt3(prompt, max_tokens=100):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Mude para o modelo de chat mais recente.
        messages=[
            {"role": "system", "content": "Você é um assistente útil que responde a perguntas."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message['content']


if __name__ == "__main__":
    main()


