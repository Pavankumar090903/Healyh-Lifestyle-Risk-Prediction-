import json
import plotly.io as pio


def fig_to_dict(fig):
    """Convert a Plotly figure to a plain dict (for JSON responses)."""
    return json.loads(pio.to_json(fig))


def transparent_layout():
    """Return a layout dict for charts that sit on a dark/transparent background."""
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        margin=dict(l=60, r=40, t=60, b=120),
    )


def clean_chart_layout():
    """Return a layout dict for charts on the personal dashboard (light background)."""
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12, color="#a0aec0"),
        title_font=dict(size=18, color="white"),
    )
