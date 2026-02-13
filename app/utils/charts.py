import json
import plotly.io as pio

def to_dict(fig): 
    return json.loads(pio.to_json(fig))

def get_chart_layout_update():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        margin=dict(l=60, r=40, t=60, b=120)
    )

def get_personal_chart_layout():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12, color="#a0aec0"),
        title_font=dict(size=18, color="white")
    )
