import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Criação do gráfico de velocímetro
def make_gauge_chart(value=750, title="Active Users"):
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={'axis': {'range': [None, 1500]},
               'steps': [
                   {'range': [0, 500], 'color': "lightgray"},
                   {'range': [500, 1000], 'color': "gray"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 990}}))

# Adiciona espaço em branco para tornar o gráfico efetivamente menor
gauge_chart1 = make_gauge_chart(750, "Active Users")
gauge_chart2 = make_gauge_chart(300, "New Signups")
gauge_chart3 = make_gauge_chart(25, "Churn Rate")
gauge_chart4 = make_gauge_chart(10, "Bounce Rate")

# Cada gráfico tem um tamanho especificado e uma margem direita maior para mover para a esquerda
for chart in [gauge_chart1, gauge_chart2, gauge_chart3, gauge_chart4]:
    chart.update_layout(
        autosize=False,
        width=300,
        height=250,
        margin=dict(
            l=0,# Aumentar a margem esquerda
            r=150,  # Aumentar a margem direita
            b=10,
            t=100,
            pad=10
        ),
    )
st.header("KPIs")

# Primeira linha
col1, col2, col3, col4, = st.columns(4)

css_code = """
<style>
.stMetric .stMetric-value {
    justify-content: flex-end;
    padding-right: 50px;
}
</style>
"""



with col1:
    st.markdown(css_code, unsafe_allow_html=True)
    st.metric(label="Active Users", value="1200", delta="30%", delta_color="off")

with col2:
    st.metric(label="New Signups", value="300", delta="20%", delta_color="inverse")

with col3:
    st.metric(label="Churn Rate", value="25%", delta="5%", delta_color="normal")

with col4:
    st.metric(label="Bounce Rate", value="10%", delta="15%", delta_color="inverse")

# Segunda linha
col5, col6, col7, col8 = st.columns(4)

with col5:
    st.plotly_chart(gauge_chart1)

with col6:
    st.plotly_chart(gauge_chart2)

with col7:
    st.plotly_chart(gauge_chart3)

with col8:
    st.plotly_chart(gauge_chart4)
