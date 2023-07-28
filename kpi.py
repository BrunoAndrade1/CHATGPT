import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Slider
slider_value = st.slider('Select a range of values', 0.0, 100.0, (25.0, 75.0))
st.write('You selected a range: ', slider_value)

# Checkbox
if st.checkbox('Show/Hide'):
    # This code will run if the checkbox is checked
    st.text("Showing or Hiding Widget")

# File uploader
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

# Map
df_map = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])
st.map(df_map)


st.header("More KPIs")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Revenue", value="$12000", delta="10%")
with col2:
    st.metric(label="Cost", value="$5000", delta="-15%")
with col3:
    st.metric(label="Profit", value="$7000", delta="25%")

st.header("Bar Chart")
df = pd.DataFrame({
   'Fruits': ['Apples', 'Oranges', 'Bananas', 'Berries'],
   'Amount': [12, 26, 3, 17]
})

st.bar_chart(df)



#st.title("Streamlit KPI Dashboard")


st.header("Table")
table_data = pd.DataFrame({
  'Avocado': [3, 2, 1, 0],
  'Banana': [10, 20, 30, 40],
  'Cherry': [100, 200, 300, 400]
})

st.table(table_data)


st.title("Dashboard de KPIs")

# KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Receita", value="$150,000", delta="10%", delta_color="inverse")

with col2:
    st.metric(label="Custo", value="$50,000", delta="-15%", delta_color="normal")

with col3:
    st.metric(label="Lucro", value="$100,000", delta="25%", delta_color="inverse")

with col4:
    st.metric(label="Satisfação do Cliente", value="95%", delta="2%", delta_color="inverse")


import streamlit as st

st.title("Dashboard de KPIs de E-commerce")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Visitantes Únicos", value="150K", delta="5% ⬆️", delta_color="inverse")

with col2:
    st.metric(label="Taxa de Conversão", value="12.5%", delta="-1% ⬇️", delta_color="normal")

with col3:
    st.metric(label="Vendas Totais", value="$1.2M", delta="10% ⬆️", delta_color="inverse")

with col4:
    st.metric(label="Ticket Médio", value="$50", delta="2% ⬆️", delta_color="inverse")



st.title("Dashboard de KPIs de Plataforma de Música")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Usuários Ativos", value="250K", delta="3% ⬆️", delta_color="inverse")

with col2:
    st.metric(label="Novos Usuários", value="20K", delta="4% ⬆️", delta_color="inverse")

with col3:
    st.metric(label="Músicas Reproduzidas", value="2.5M", delta="10% ⬆️", delta_color="inverse")

with col4:
    st.metric(label="Artistas Mais Populares", value="The Beatles", delta="Nova entrada", delta_color="info")




st.title("Dashboard de KPIs de Aplicativo de Saúde e Fitness")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Usuários Ativos", value="100K", delta="5% ⬆️", delta_color="inverse")

with col2:
    st.metric(label="Novas Assinaturas", value="3K", delta="3% ⬆️", delta_color="inverse")

with col3:
    st.metric(label="Sessões de Treino", value="500K", delta="15% ⬆️", delta_color="inverse")

with col4:
    st.metric(label="Duração Média da Sessão", value="45 min", delta="2 min ⬆️", delta_color="inverse")