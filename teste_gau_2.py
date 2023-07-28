import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import calendar
import warnings
warnings.filterwarnings("ignore")

# Create four columns
col1, col2, col3, col4 = st.columns(4)

# Indicador 1
fig1 = go.Figure(go.Indicator(
    mode = "number+delta",
    value = 492, 
    delta = {"reference": 512, "valueformat": ".0f"},
    title = {"text": "Indicador de Desempenho<br><span style='font-size:0.8em;color:gray'>Produto X</span><br>"},
    domain = {'y': [0, 1], 'x': [0.25, 0.75]}
))
col1.plotly_chart(fig1, use_container_width=True)

# Indicador 2
fig2 = go.Figure(go.Indicator(
    mode = "number+delta",
    value = 300, 
    delta = {"reference": 350, "valueformat": ".0f"},
    title = {"text": "Indicador de Desempenho<br><span style='font-size:0.8em;color:gray'>Produto Y</span><br>"},
    domain = {'y': [0, 1], 'x': [0.25, 0.75]}
))
col2.plotly_chart(fig2, use_container_width=True)

# Indicador 3
fig3 = go.Figure(go.Indicator(
    mode = "number+delta",
    value = 600, 
    delta = {"reference": 500, "valueformat": ".0f"},
    title = {"text": "Indicador de Desempenho<br><span style='font-size:0.8em;color:gray'>Produto Z</span><br>"},
    domain = {'y': [0, 1], 'x': [0.25, 0.75]}
))
col3.plotly_chart(fig3, use_container_width=True)

# Indicador 4
fig4 = go.Figure(go.Indicator(
    mode = "number+delta",
    value = 800, 
    delta = {"reference": 900, "valueformat": ".0f"},
    title = {"text": "Indicador de Desempenho<br><span style='font-size:0.8em;color:gray'>Produto W</span><br>"},
    domain = {'y': [0, 1], 'x': [0.25, 0.75]}
))
col4.plotly_chart(fig4, use_container_width=True)

fig1.update_layout(autosize=False, width=500, height=500)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Lucro_liquido = 2300
Total_investido = 2300
Despesas_mesal = 3000
Investimento = 5000

col1, col2, col3, col4 = st.columns(4)
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
for i ,config in enumerate(configs):
    fig3 = create_gauge(**config) # ** desempacota o dicionario 
    cols[i].plotly_chart(fig3)  


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# # KPIs de vendas
# kpi_values = [2000, 150, 10, 500, 20]
# kpi_names = ['Vendas Totais', 'Vendas Online', 'Vendas Offline', 'Novos Clientes', 'Taxa de Churn']

# # Criar o gráfico
# fig = go.Figure(data=[
#     go.Bar(
#         name='KPIs de Vendas', 
#         x=kpi_values, 
#         y=kpi_names, 
#         orientation='h',
#         marker=dict(
#             color='rgba(50, 171, 96, 0.6)',
#             line=dict(color='rgba(50, 171, 96, 1.0)', width=1)
#         ),
#         hoverinfo='x'
#     )
# ])

# # Configurar o layout do gráfico
# fig.update_layout(
#     title_text='KPIs de Vendas', 
#     xaxis=dict(
#         showgrid=False,
#         showline=False,
#         showticklabels=True,
#         zeroline=False,
#     ),
#     yaxis=dict(
#         showgrid=False,
#         showline=False,
#         showticklabels=True,
#         zeroline=False,
#     ),
#     autosize=False,
#     margin=dict(
#         autoexpand=False,
#         l=100,
#         r=20,
#         t=110,
#     ),
#     showlegend=False,
#     plot_bgcolor='white'
# )

# st.plotly_chart(fig)  # Exibir o gráfico no Streamlit








