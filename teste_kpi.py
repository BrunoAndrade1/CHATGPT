import streamlit as st
from streamlit_echarts import st_echarts

def app():
    col1, col2, col3, col4 = st.columns(4)

    option = {
        "series": [
            {
                "type": 'gauge',
                "startAngle": 180,
                "endAngle": 0,
                "center": ['50%', '75%'],
                "radius": '90%',
                "min": 0,
                "max": 1,
                "splitNumber": 8,
                "axisLine": {
                    "lineStyle": {
                        "width": 6,
                        "color": [
                            [0.25, '#FF6E76'],
                            [0.5, '#FDDD60'],
                            [0.75, '#58D9F9'],
                            [1, '#7CFFB2']
                        ]
                    }
                },
                "pointer": {
                    "icon": 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
                    "length": '12%',
                    "width": 20,
                    "offsetCenter": [0, '-60%'],
                    "itemStyle": {
                        "color": 'auto'
                    }
                },
                "axisTick": {
                    "length": 12,
                    "lineStyle": {
                        "color": 'auto',
                        "width": 2
                    }
                },
                "splitLine": {
                    "length": 20,
                    "lineStyle": {
                        "color": 'auto',
                        "width": 5
                    }
                },
                "axisLabel": {
                    "color": '#464646',
                    "fontSize": 20,
                    "distance": -60,
                    "formatter": [
                        {"value": 0.875, "label": "Grade A"},
                        {"value": 0.625, "label": "Grade B"},
                        {"value": 0.375, "label": "Grade C"},
                        {"value": 0.125, "label": "Grade D"},
                        {"value": 0, "label": ""}
                    ]
                },
                "title": {
                    "offsetCenter": [0, '-10%'],
                    "fontSize": 20
                },
                "detail": {
                    "fontSize": 30,
                    "offsetCenter": [0, '-35%'],
                    "valueAnimation": True,
                    "formatter": "{value}%"
                },
                "data": [
                    {
                        "value": 0.7,
                        "name": 'Grade Rating'
                    }
                ]
            }
        ]
    }

    with col1:
        st_echarts(options=option, key="echarts1")

    with col2:
        st_echarts(options=option, key="echarts2")

    with col3:
        st_echarts(options=option, key="echarts3")

    with col4:
        st_echarts(options=option, key="echarts4")

app()
