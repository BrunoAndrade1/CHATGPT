import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
###############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨

kpi_values = [25000, 150, 10] # valores dos KPIs
kpi_names = ['Vendas', 'Clientes Adquiridos', 'Taxa de Churn'] # nomes dos KPIs

fig, ax = plt.subplots()

bars = ax.bar(kpi_names, kpi_values, color=['green', 'blue', 'red']) # Cria o gráfico de barras

# Adiciona os valores de cada barra no topo da mesma
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')




ax.set_title('KPIs') # Define o título do gráfico
ax.set_xlabel('Indicadores') # Define o título do eixo x
ax.set_ylabel('Valores') # Define o título do eixo y

st.pyplot(fig) # Exibe o gráfico com o Streamlit


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# KPIs de vendas
kpi_values = [2000, 150, 10, 500, 20]
kpi_names = ['Vendas Totais', 'Vendas Online', 'Vendas Offline', 'Novos Clientes', 'Taxa de Churn']

# Criar o gráfico
fig = go.Figure(data=[
    go.Bar(
        name='KPIs de Vendas', 
        x=kpi_values, 
        y=kpi_names, 
        orientation='h',
        marker=dict(
            color='rgba(50, 171, 96, 0.6)',
            line=dict(color='rgba(50, 171, 96, 1.0)', width=1)
        ),
        hoverinfo='x'
    )
])

# Configurar o layout do gráfico
fig.update_layout(
    title_text='KPIs de Vendas', 
    xaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
        zeroline=False,
    ),
    autosize=False,
    margin=dict(
        autoexpand=False,
        l=100,
        r=20,
        t=110,
    ),
    showlegend=False,
    plot_bgcolor='white'
)

st.plotly_chart(fig)  # Exibir o gráfico no Streamlit