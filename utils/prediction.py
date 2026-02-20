import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


def calculate_risk(input_df, model, encoders, cat_cols):
    """Run the model and return the (score, label) tuple."""
    for col in cat_cols:
        if col not in input_df.columns:
            continue
        try:
            encoder = encoders[col]
            val = input_df.loc[0, col]
            if val in encoder.classes_:
                input_df.loc[0, col] = encoder.transform([val])[0]
            else:
                input_df.loc[0, col] = -1
        except Exception:
            pass

    pred_prob  = model.predict_proba(input_df)[0]
    pred_class = int(np.argmax(pred_prob))
    risk_label = {0: "Low", 1: "Medium", 2: "High"}.get(pred_class, "Low")

    final_score = min(int(pred_class * 33 + pred_prob[pred_class] * 33), 99)
    return final_score, risk_label


def generate_risk_gauge(final_score):
    """Return an HTML donut chart showing the risk score."""
    if final_score < 33:
        color = "#28a745"
    elif final_score < 66:
        color = "#ffc107"
    else:
        color = "#dc3545"

    fig = go.Figure(data=[go.Pie(
        values=[final_score, 100 - final_score],
        labels=['Risk Score', 'Remaining'],
        hole=0.7,
        marker=dict(colors=[color, '#f0f0f0'], line=dict(color='white', width=3)),
        textinfo='none',
        hoverinfo='label+percent',
        direction='clockwise',
        sort=False
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        annotations=[dict(
            text=f'<b>{final_score}%</b>',
            x=0.5, y=0.5,
            font=dict(size=64, color=color, family='Inter'),
            showarrow=False
        )]
    )
    return pio.to_html(fig, full_html=False, config={'displayModeBar': False}, div_id="riskGaugePlot")


def get_recommendations(user_input, risk_label):
    """Return (recs, alerts) lists based on the user's vitals and risk level."""
    recs, alerts = [], []

    if user_input['hba1c'] >= 6.5:
        alerts.append(("🚨", "Critical HbA1c Level", "danger"))
        recs.append(("🚨", "Critical HbA1c", "Level is in diabetes range. Seek medical attention."))
    elif user_input['hba1c'] >= 5.7:
        alerts.append(("⚠️", "Pre-Diabetes Warning", "warning"))
        recs.append(("⚠️", "Pre-Diabetes", "Elevated HbA1c. Lifestyle changes required."))

    if user_input['systolic_bp'] > 140 or user_input['diastolic_bp'] > 90:
        alerts.append(("💓", "High Blood Pressure", "danger"))
        recs.append(("💓", "Hypertension", "Blood pressure is high. Monitor regularly."))

    if user_input['bmi'] >= 30:
        alerts.append(("⚖️", "Obesity Detected", "danger"))
        recs.append(("⚖️", "Weight Management", "BMI > 30. A weight loss plan is recommended."))

    if user_input['cholesterol_total'] > 200:
        alerts.append(("🍔", "High Cholesterol", "warning"))
        recs.append(("🍔", "Cholesterol", "Total cholesterol is high. Consider diet changes."))

    if risk_label == "High":
        recs.append(("👨‍⚕️", "Specialist Review", "High risk detected. Consult an endocrinologist."))
    elif risk_label == "Medium":
        alerts.append(("🟡", "Preventive Watch", "warning"))
        recs.append(("📉", "Risk Reduction", "Moderate risk detected. Adopt preventive lifestyle changes."))
    elif risk_label == "Low":
        alerts.append(("✅", "Optimal Maintenance", "success"))
        recs.append(("🛡️", "Maintenance", "Low risk. Continue your healthy habits to stay safe."))

    wellness_tips = [
        ("🥗", "Balanced Diet",     "Maintain a diet rich in vegetables, whole grains, and lean proteins."),
        ("💧", "Hydration",         "Drink 8-10 glasses of water daily to support metabolism."),
        ("😴", "Sleep Hygiene",     "Aim for 7-9 hours of quality sleep to regulate blood sugar."),
        ("🧘", "Stress Management", "Practice mindfulness or yoga to keep cortisol levels low."),
        ("🩺", "Regular Checkups",  "Schedule annual health screenings to monitor your vitals."),
    ]
    for tip in wellness_tips:
        if len(recs) >= 5:
            break
        recs.append(tip)

    return recs[:5], alerts


def generate_prediction_charts(user_input, hba1c_impact, bmi_impact, bp_impact):
    """Build and return a dict of HTML chart snippets for the prediction results page."""

    # How much each factor contributes to risk
    impact_df = pd.DataFrame([
        {"Factor": "Blood Sugar", "Impact": hba1c_impact},
        {"Factor": "Body Mass",   "Impact": bmi_impact},
        {"Factor": "BP Level",    "Impact": bp_impact},
        {"Factor": "Sedentary",   "Impact": max(0, (150 - user_input['physical_activity_minutes_per_week']) * 0.1)},
    ]).sort_values("Impact", ascending=True)
    impact_fig = px.bar(impact_df, x="Impact", y="Factor", orientation='h',
                        title="Impact on Risk Score", template="plotly_white",
                        color="Impact", color_continuous_scale='Reds')
    impact_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250,
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # Your vitals compared to clinical targets
    v_df = pd.DataFrame([
        {"Metric": "HbA1c (%)", "You": user_input['hba1c'],         "Target": 5.6},
        {"Metric": "Glucose",   "You": user_input['glucose_fasting'], "Target": 100},
        {"Metric": "BMI",       "You": user_input['bmi'],            "Target": 24.9},
    ]).melt(id_vars="Metric", var_name="Type", value_name="Value")
    v_fig = px.bar(v_df, x="Metric", y="Value", color="Type", barmode="group",
                   title="Your Stats vs Targets", template="plotly_white")
    v_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # Wellness dimension scores (0-100 scale)
    dim_df = pd.DataFrame({
        "Dimension": ["Blood Sugar", "Body Weight", "Activity", "Circulation"],
        "Score": [
            100 - min(100, hba1c_impact * 2),
            100 - min(100, bmi_impact * 2),
            min(100, (user_input['physical_activity_minutes_per_week'] / 150) * 100),
            100 - min(100, bp_impact * 5),
        ]
    })
    r_fig = px.bar(dim_df, x="Score", y="Dimension", orientation='h',
                   title="Wellness Dimension Scores (0-100)",
                   template="plotly_white", color="Score", color_continuous_scale="RdYlGn")
    r_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    r_fig.update_coloraxes(showscale=False)

    # Blood pressure zone map
    bp_fig = go.Figure()
    bp_fig.add_shape(type="rect", x0=80,  y0=60, x1=120, y1=80,  fillcolor="green",  opacity=0.2, line_width=0, layer="below")
    bp_fig.add_shape(type="rect", x0=120, y0=80, x1=140, y1=90,  fillcolor="orange", opacity=0.2, line_width=0, layer="below")
    bp_fig.add_shape(type="rect", x0=140, y0=90, x1=180, y1=110, fillcolor="red",    opacity=0.1, line_width=0, layer="below")
    bp_fig.add_trace(go.Scatter(
        x=[user_input['systolic_bp']], y=[user_input['diastolic_bp']],
        mode='markers+text', name="You", text=["YOU"], textposition="top center",
        marker=dict(size=12, color='black', symbol='star')
    ))
    bp_fig.update_layout(title="Blood Pressure Zoning", xaxis_title="Systolic", yaxis_title="Diastolic",
                         height=250, margin=dict(l=20, r=20, t=40, b=20),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # Physical activity vs risk curve
    act_x = np.linspace(0, 300, 50)
    act_y = 100 * np.exp(-act_x / 150)
    c_fig = px.area(x=act_x, y=act_y, title="Activity vs Risk Curve",
                    labels={'x': 'Minutes/Week', 'y': 'Risk Influence'}, template="plotly_white")
    user_act = user_input['physical_activity_minutes_per_week']
    c_fig.add_scatter(x=[user_act], y=[100 * np.exp(-user_act / 150)],
                      mode='markers', name="Your Spot", marker=dict(size=10, color='red'))
    c_fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    no_toolbar = {'displayModeBar': False}
    return {
        "impact": pio.to_html(impact_fig, full_html=False, config=no_toolbar),
        "vitals": pio.to_html(v_fig,      full_html=False, config=no_toolbar),
        "radar":  pio.to_html(r_fig,      full_html=False, config=no_toolbar),
        "bp_map": pio.to_html(bp_fig,     full_html=False, config=no_toolbar),
        "curve":  pio.to_html(c_fig,      full_html=False, config=no_toolbar),
    }
