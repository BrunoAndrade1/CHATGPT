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

def pagina_1():
    st.title('Página 1')
    # Aqui vai o resto do código para a página 1
    Lucro_liquido = 2300
    Total_investido = 2300
    Despesas_mesal = 3000
    Investimento = 5000


    # https://plotly.com/python/v3/gauge-charts/
    def create_gauge(current_value, min_value, max_value, quadrant_colors, quadrant_text, sensor_text):
        n_quadrants = len(quadrant_colors) - 1
        hand_length = np.sqrt(2) / 4
        hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

        fig3 = go.Figure(
            data=[
                go.Pie(
                    values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
                    rotation=90,
                    hole=0.5,
                    marker_colors=quadrant_colors,
                    text=quadrant_text,
                    textinfo="text",
                    hoverinfo="skip",
                ),
            ],
            layout=go.Layout(
                showlegend=False,
                margin=dict(b=0,t=10,l=10,r=10),
                width=200,
                height=150,
                paper_bgcolor=quadrant_colors[0],
                annotations=[
                    go.layout.Annotation(
                        text=f"<b>{sensor_text}:</b><br>R$ :{current_value}",
                        x=0.5, xanchor="center", xref="paper",
                        y=0.25, yanchor="bottom", yref="paper",
                        showarrow=False,
                    ),
                #paper_bgcolor="White", # aqui você coloca a cor que deseja ****quadrant_colors[0]***
                ],
                shapes=[
                    go.layout.Shape(
                        type="circle",
                        x0=0.48, x1=0.52,
                        y0=0.48, y1=0.52,
                        fillcolor="#333",
                        line_color="#333",
                    ),
                    go.layout.Shape(
                        type="line",
                        x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                        y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                        line=dict(color="#333", width=4)
                    )
                ]
            )
        )
        
        return fig3
    ##_______________________

    configs = [
        {"current_value": Lucro_liquido, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#ffffff", "#f25829", "#f2a529", "#eff229", "#85e043", "#2bad4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Lucro Líquido"},
        {"current_value": Total_investido, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#ffffff", "#f20030", "#f25a00", "#eff200", "#00e043", "#002b4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Total Investido"},
        {"current_value": Despesas_mesal, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#ffffff", "#f20030", "#f25a00", "#eff200", "#00e043", "#002b4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Despesas Mensais"},
        {"current_value": Investimento, "min_value": 0, "max_value": 5000, "quadrant_colors": ["#ffffff", "#f20030", "#f25a00", "#eff200", "#00e043", "#002b4e"], "quadrant_text": ["", "<b style='font-size:10px;'>V high</b>", "<b style='font-size:10px;'>High</b>", "<b style='font-size:10px;'>M</b>", "<b style='font-size:10px;'>L</b>", "<b style='font-size:10px;'>Very l</b>"], "sensor_text": "Investimento"},
    ]



    cols = st.columns(4)  # Define a quantidade de colunas

    # Criar os gráficos de velocímetro e exibi-los
    for i ,config in enumerate(configs):
        fig3 = create_gauge(**config) # ** desempacota o dicionario 
        cols[i].plotly_chart(fig3)  # Adicione a figura à coluna i

    #________________________________________________________________
    #                Outro grafico gauge
    # cols = st.columns(4)
    # for i ,config in enumerate(configs):
    #     fig3 = create_gauge(**config) # ** desempacota o dicionario 
    #     cols[i].plotly_chart(fig3)  # Adicione a figura à coluna i
    ######%%%%%%%%%%%%%

    data = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100) * 100,
        'C': np.random.rand(100) * -100,
        'D': np.random.rand(100) * 50,
    })

    # Definir as colunas
    col1, col2, col3, col4 = st.columns(4)

    # Ajustar a altura e a largura de cada gráfico
    fig, ax = plt.subplots(figsize=(4, 4))  # ajuste o tamanho conforme necessário

    # Gere seu gráfico para a primeira coluna (por exemplo, um gráfico de barras)
    sns.barplot(x=data.index, y='A', data=data, ax=ax)
    col1.pyplot(fig)
    #___________________________________________

    dados = {
        'Mes': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
        'Canal A': [100, 120, 130, 120, 110, 120, 110, 130, 120, 140, 150, 160],
        'Canal B': [100, 100, 120, 90, 130, 100, 120, 130, 120, 110, 110, 120]
    }

    df = pd.DataFrame(dados)
    df.set_index('Mes', inplace=True)

    # Definir o estilo do seaborn
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(4, 4))
    df.plot(kind='bar', stacked=True, ax=ax, color=['steelblue', 'darkorange'])
    plt.title('Novos Clientes ao Longo do Tempo', fontsize=5, pad=5)
    plt.xlabel('Mês', fontsize=14)
    plt.ylabel('Novos Clientes', fontsize=5)
    plt.xticks(fontsize=4, rotation=0)
    plt.yticks(fontsize=4)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Mostrar o gráfico no Streamlit
    col2.pyplot(fig)


    # fig, ax = plt.subplots(figsize=(4, 4))
    # sns.lineplot(x=data.index, y='B', data=data, ax=ax)
    # col2.pyplot(fig)
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    dados = {
        'Mes': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
        'Taxa de Retenção': [85, 83, 82, 80, 78, 79, 81, 82, 80, 78, 76, 77]
    }

    df = pd.DataFrame(dados)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.lineplot(x='Mes', y='Taxa de Retenção', data=df, marker='o',ax=ax,color= 'steelblue',linewidth=2.5)
    plt.title('Taxa de Retenção ao Longo do Tempo')
    plt.xlabel('Mês')
    plt.ylabel('Taxa de Retenção (%)')
    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)
    plt.grid(True, linestyle='--', alpha=0.6)
    col3.pyplot(fig)

    # ##############################################

    satisfacao_cliente = np.random.uniform(80, 100)
    df = pd.DataFrame({
        "Níveis de Satisfação": ["Excelente", "Bom", "Médio", "Ruim"],
        "Valor": [20, 100, 80, 60]
    })
    # Criar gráfico de barras para satisfação do cliente com seaborn
    fig, ax = plt.subplots(figsize=(4, 4))
    #sns.barplot(x=df.index, y="Satisfação do Cliente", data=df, ax=ax, ci=None)
    sns.barplot(x="Valor", y="Níveis de Satisfação", data=df, ax=ax, orient="h", ci=None, palette="viridis")

    ax.set_xlim([0, 100])  # Definir limites do eixo x
    ax.set_title("Níveis de Satisfação do Cliente")  # Adicionar título
    col4.pyplot(fig)  # Exibir gráfico de barras


    #$$$$$$$$$$$$$$$$$$
    # fig, ax = plt.subplots(figsize=(4, 4))
    # sns.scatterplot(x=data.index, y='D', data=data, ax=ax)
    # col4.pyplot(fig)


    ######################################################################




    # Definir as colunas
    col1, col2, col3, col4 = st.columns(4)

    # Calcular KPIs
    vendas_totais = np.random.randint(10000, 20000)
    novos_clientes = np.random.randint(100, 500)
    taxa_retencao = np.random.uniform(0.7, 0.9)
    satisfacao_cliente = np.random.uniform(80, 100)

    # Exibir KPIs
    col1.metric("Vendas Totais", vendas_totais, delta="3%")  # suponha que as vendas aumentaram 3%
    col2.metric("Novos Clientes", novos_clientes, delta="2%")  # suponha que o número de novos clientes aumentou 2%
    col3.metric("Taxa de Retenção", f"{taxa_retencao * 100:.2f}%", delta="-1%")  # suponha que a taxa de retenção diminuiu 1%
    col4.metric("Satisfação do Cliente", f"{satisfacao_cliente:.2f}%", delta="4%")  # suponha que a satisfação do cliente aumentou 4%


    # # Ajustar a altura e a largura de cada gráfico
    # fig, ax = plt.subplots(figsize=(4, 4))  # ajuste o tamanho conforme necessário

    # # Gere seu gráfico para a primeira coluna (por exemplo, um gráfico de distribuição)
    # sns.distplot(data['A'], ax=ax)
    # col1.pyplot(fig)

    ## indicador funcionando falta arruamr tamanho 
    #col1, col2, col3, col4 = st.columns(4)
    # fig1 = go.Figure(go.Indicator(
    #     mode = "number+delta",
    #     value = 492, 
    #     delta = {"reference": 512, "valueformat": ".0f"},
    #     title = {"text": "Indicador de Desempenho<br><span style='font-size:0.8em;color:gray'>Produto X</span><br>"},
    #     domain = {'y': [0, 1], 'x': [0.25, 0.75]}
    # ))

    # col1.plotly_chart(fig1, use_container_width=True)
    #$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#####Obs a pagina tem que chaamr antes 
#***********************************************************************************************************
#**************************** pagina 2****************************************************************
def pagina_2():
    st.title('Página 2')
    st.write('hello') 

paginas = ["Página 1", "Página 2"]

# Criar uma caixa de seleção na barra lateral
pagina_selecionada = st.sidebar.selectbox("Escolha uma página:", paginas)

# Usar um if-elif para determinar qual página exibir
if pagina_selecionada == "Página 1":
    pagina_1()
elif pagina_selecionada == "Página 2":
    pagina_2()
#)))))))))))))))))))))))))))))))))))))))))))))))))))))))
