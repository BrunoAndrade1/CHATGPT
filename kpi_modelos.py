import streamlit as st
import plotly.graph_objects as go
import random
from faker import Faker

fake = Faker()

# Gerando alguns dados
crypto_names = [fake.cryptocurrency_name() for _ in range(5)]
values = [fake.random_number(digits=4) for _ in range(5)]
colors = [fake.rgb_css_color() for _ in range(5)]

# Bar chart
bar_fig = go.Figure(data=[go.Bar(x=crypto_names, y=values, marker_color=colors)])
bar_fig.update_layout(title="Cryptocurrency Bar Chart", xaxis_title="Cryptocurrency", yaxis_title="Value")
st.plotly_chart(bar_fig)

# Gerando alguns dados
x_data = list(range(10))
y_data = [random.randint(1, 100) for _ in range(10)]

# Line chart
line_fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode='lines'))
line_fig.update_layout(title="Random Line Chart", xaxis_title="X", yaxis_title="Y")
st.plotly_chart(line_fig)

# Area chart
area_fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, fill='tozeroy', mode='none'))
area_fig.update_layout(title="Random Area Chart", xaxis_title="X", yaxis_title="Y")
st.plotly_chart(area_fig)

# Gerando alguns dados
labels = [fake.cryptocurrency_name() for _ in range(5)]
values = [random.randint(1, 100) for _ in range(5)]

# Pie chart
pie_fig = go.Figure(data=go.Pie(labels=labels, values=values))
pie_fig.update_layout(title="Random Pie Chart")
st.plotly_chart(pie_fig)

# Gerando alguns dados
x_data = [random.random() for _ in range(10)]
y_data = [random.random() for _ in range(10)]

# Scatter chart
scatter_fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode='markers'))
scatter_fig.update_layout(title="Random Scatter Chart", xaxis_title="X", yaxis_title="Y")
st.plotly_chart(scatter_fig)

# Gerando alguns dados para box plot
y_box = [random.gauss(mu=0, sigma=0.5) for _ in range(200)]

# Box plot
box_fig = go.Figure(data=go.Box(y=y_box))
box_fig.update_layout(title="Random Box Plot", yaxis_title="Value")
st.plotly_chart(box_fig)

# Gerando alguns dados para violin plot
y_violin = [random.gauss(mu=0, sigma=0.5) for _ in range(200)]

# Violin plot
violin_fig = go.Figure(data=go.Violin(y=y_violin))
violin_fig.update_layout(title="Random Violin Plot", yaxis_title="Value")
st.plotly_chart(violin_fig)

# KPIs
st.header("KPIs")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Active Users", value="1200", delta="30%")
with col2:
    st.metric(label="New Signups", value="300", delta="20%")
with col3:
    st.metric(label="Churn Rate", value="25%", delta="5%")

# Caixa de texto
st.header("Important Note")
st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean volutpat ante ac lacus lobortis, a feugiat nulla varius.")

# Caixa de valor
st.header("Special Value")
st.write(f"The special value today is: {random.randint(1, 100)}")

