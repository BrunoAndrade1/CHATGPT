import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import openai
openai.api_key = ''
openai.api_key = st.secrets["openai"]["api_key"]
st.set_page_config(layout="wide")

st.title('Indicadores Contábeis')


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Chat gpt #####################################################

def chat_with_gpt3(prompt, max_tokens=100):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "Você é um assistente útil que responde a perguntas."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message['content']

def chat():
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

    user_input = st.text_area("Digite sua pergunta aqui. Por exemplo: 'O que são Indicadores Contábeis?'", value="", height=100,)

    if st.button("Enviar"):
        if user_input:
            st.write("Você:", user_input)
            response = chat_with_gpt3(user_input)
            st.write("ChatGPT:", response)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Suponha que você tenha esses dados
dados = {
    'Mês': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
    'Dinheiro': [2000, 2200, 2300, 2100, 2200, 2400, 2300, 2500, 2400, 2600, 2500, 2800],
    'Contas a Receber': [1500, 1600, 1500, 1700, 1600, 1800, 1700, 1600, 1700, 1800, 1900, 1800],
    'Estoques': [2500, 2400, 2600, 2500, 2400, 2600, 2500, 2400, 2500, 2600, 2400, 2600]
}

df = pd.DataFrame(dados)

df.set_index('Mês', inplace=True)

# Definir o estilo do seaborn
sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(10, 9))
df.plot(kind='bar', stacked=True, ax=ax, color=['steelblue', 'darkorange', 'green'])

plt.title('Ativo Circulante ao Longo do Tempo', fontsize=20, pad=20)
plt.xlabel('Mês', fontsize=14)
plt.ylabel('Ativo Circulante (R$)', fontsize=14)
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

###################### Grafico col2 ##################

dados2 = {
    'Item': ['Dinheiro', 'Contas a Receber', 'Estoques'],
    'Valor': [20000, 30000, 50000],
}

df2 = pd.DataFrame(dados2)

# Define o estilo do Seaborn
sns.set(style="whitegrid")

# Cria o gráfico de pizza com Matplotlib
fig2, ax1 = plt.subplots(figsize=(10, 9))
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']  # Define algumas cores
explode = (0.1, 0, 0)  # "explode" a primeira fatia
ax1.pie(df2['Valor'], explode=explode, labels=df2['Item'], autopct='%1.1f%%', colors=colors, shadow=True, startangle=140)
plt.title('Análise Vertical: Composição do Ativo Circulante', fontsize=16, color='darkblue')


chat()
col1, col2 = st.columns([1,1])
# Mostrar o gráfico no Streamlit
col1.pyplot(fig)
col2.pyplot(fig2)


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ ROW 2###########################

# Gerar dados aleatórios para exemplo
np.random.seed(0)
data2 = np.random.rand(12) * 100

# Criar DataFrame
df5 = pd.DataFrame(data2, columns=['Termometro Kanitz'])
df5['Mes'] = pd.date_range(start='1/1/2023', periods=len(df), freq='M')

# Criar o objeto para o gráfico
fig10 = go.Figure()

# Adicionar a linha para os dados do termômetro Kanitz
fig10.add_trace(go.Scatter(x=df5['Mes'], y=df5['Termometro Kanitz'],
                         mode='lines+markers',
                         name='Termômetro Kanitz',
                         line=dict(color='darkblue', width=2),
                         marker=dict(size=8, color='red')
                        )
             )

# Adicionar título e rótulos dos eixos
fig10.update_layout(title='Termômetro de Kanitz ao Longo do Tempo',
                  xaxis_title='Meses',
                  yaxis_title='Termômetro de Kanitz',
                  title_x=0.1,  # Centraliza o título
                  autosize=False,
                  width=350,
                  height=400,
                  )
fig10.update_xaxes(tickangle=90)

# Mostrar o gráfico

########################################################## col2 row 2
sns.set(style="whitegrid")

# Criar DataFrame
df = pd.DataFrame({'Termometro Kanitz': data2})
df['Mes'] = pd.date_range(start='1/1/2023', periods=len(df), freq='M')

# Usando Plotly para criar gráfico interativo
fig = px.bar(df, x='Mes', y='Termometro Kanitz', 
             labels={'Termometro Kanitz':'Termômetro de Kanitz (%)'}, 
             color='Termometro Kanitz', color_continuous_scale='Reds',
             title='Termômetro de Kanitz ao Longo do Tempo')

fig.update_layout(yaxis_tickformat = '.2f')
# Mostrar o gráfico no Streamlit
fig.update_xaxes(tickangle=90)


col1, col2 = st.columns([1,1])
# Mostrar o gráfico no StreamlitC
col1.plotly_chart(fig10)

col2.plotly_chart(fig)
