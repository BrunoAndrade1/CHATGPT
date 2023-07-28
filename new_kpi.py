
import streamlit as st
import plotly.graph_objects as go

# KPIs
st.header("KPIs")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Active Users", value="1200", delta="30%", delta_color="off")

with col2:
    st.metric(label="New Signups", value="300", delta="20%", delta_color="inverse")

with col3:
    st.metric(label="Churn Rate", value="25%", delta="5%", delta_color="normal")

with col4:
    st.metric(label="Bounce Rate", value="10%", delta="15%", delta_color="inverse")


st.header("Mais KPIs")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Artistas Mais Populares", value="The Beatles", delta="Nova entrada", delta_color="normal")

with col2:
    st.metric(label="Músicas Mais Tocadas", value="Imagine", delta="2 novas", delta_color="inverse")

with col3:
    st.metric(label="Álbuns Mais Vendidos", value="Thriller", delta="5k vendas", delta_color="off")



# Create random data
import random
data1 = random.randint(500, 1000)
data2 = random.randint(500, 1000)

# Create Plotly figure
fig1 = go.Figure(go.Indicator(
    mode = "number+delta",
    value = data1,
    delta = {'reference': data2, 'relative': True, 'increasing.color': 'Green', 'decreasing.color': 'Red'},
    title = {"text": "KPI 1"}))

fig2 = go.Figure(go.Indicator(
    mode = "number+delta",
    value = data2,
    delta = {'reference': data1, 'relative': True, 'increasing.color': 'Green', 'decreasing.color': 'Red'},
    title = {"text": "KPI 2"}))

# Streamlit app
st.title("KPI Dashboard with Plotly")
st.plotly_chart(fig1)
st.plotly_chart(fig2)
