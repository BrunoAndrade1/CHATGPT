import streamlit as st
from streamlit_echarts import st_pyecharts

from pyecharts import options as opts
from pyecharts.charts import Gauge

# def gauge_base() -> Gauge:
#     c = (
#         Gauge()
#         .add("", [("Score", 87.5)])
#         .set_global_opts(title_opts=opts.TitleOpts(title="Gauge-Base"))
#     )
#     return c

# st_pyecharts(gauge_base())

# import streamlit as st
# from streamlit_echarts import st_echarts

# option1 = {
#     "tooltip": {
#         "formatter": "{a} <br/>{b} : {c}%"
#     },
#     "toolbox": {
#         "feature": {
#             "restore": {},
#             "saveAsImage": {}
#         }
#     },
#     "series": [
#         {
#             "name": 'Business Indicator',
#             "type": 'gauge',
#             "detail": {"formatter": '{value}%'},
#             "data": [{"value": 50, "name": 'tick médio'}]
#         }
#     ]
# }

# st_echarts(options=option1, key="echarts")

import streamlit as st
from streamlit_echarts import st_echarts

def get_label(value):
    if value == 0.875:
        return 'Grade A'
    elif value == 0.625:
        return 'Grade B'
    elif value == 0.375:
        return 'Grade C'
    elif value == 0.125:
        return 'Grade D'
    else:
        return ''

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
            "splitNumber": 4,
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
                "length": '40%',
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
                "formatter": ["D", "C", "B", "A", ""]
            },
            "title": {
                "offsetCenter": [0, '-10%'],
                "fontSize": 15
            },
            "detail": {
                "fontSize": 20,
                "offsetCenter": [0, '-35%'],
                "valueAnimation": True,
                "formatter": "{value}%"
            },
            "data": [
                {
                    "value": 0.5,
                    "name": 'vendas'
                }
            ]
        }
    ]        
}

st_echarts(options=option, key="echarts")
