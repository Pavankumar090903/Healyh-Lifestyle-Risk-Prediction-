import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*google-auth.*")
warnings.filterwarnings("ignore", message=".*google.generativeai.*")

import os
import re
import json
import random
from datetime import datetime
from functools import wraps

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import google.generativeai as genai

from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

# Load .env for local development (no effect in production)
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_default_key_replace_in_production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Hugging Face Spaces iframe session fix
app.config.update(
    SESSION_COOKIE_SAMESITE='None',
    SESSION_COOKIE_SECURE=True
)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# The super admin email — only this account can access the admin panel
SUPER_ADMIN_EMAIL = "mulagiripavankumar886@gmail.com"

# Load the trained model and its supporting files.
# All .pkl files must be in the same folder as this script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model    = joblib.load(os.path.join(BASE_DIR, "diabetes_risk_model.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))
num_cols = joblib.load(os.path.join(BASE_DIR, "num_cols.pkl"))
cat_cols = joblib.load(os.path.join(BASE_DIR, "cat_cols.pkl"))

# Set up Gemini AI if an API key is available
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)
    HAS_GEMINI = True
else:
    HAS_GEMINI = False
    app.logger.warning("Gemini API key not set. Chatbot will use local fallback responses.")


# --- Database Models ---

class User(UserMixin, db.Model):
    id                  = db.Column(db.Integer, primary_key=True)
    email               = db.Column(db.String(100), unique=True, nullable=False)
    password            = db.Column(db.String(100), nullable=False)
    name                = db.Column(db.String(1000))
    role                = db.Column(db.String(20), default='patient', nullable=False)
    admin_request_status = db.Column(db.String(20), default='none')
    records             = db.relationship('MedicalRecord', backref='patient', lazy=True)


class MedicalRecord(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age             = db.Column(db.Integer)
    gender          = db.Column(db.String(10))
    bmi             = db.Column(db.Float)
    hba1c           = db.Column(db.Float)
    glucose         = db.Column(db.Integer)
    systolic_bp     = db.Column(db.Integer)
    diastolic_bp    = db.Column(db.Integer)
    cholesterol     = db.Column(db.Integer)
    activity_minutes = db.Column(db.Integer)
    family_history  = db.Column(db.Integer)
    risk_score      = db.Column(db.Integer)
    risk_label      = db.Column(db.String(20))
    timestamp       = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Create tables on startup if they don't exist yet
with app.app_context():
    db.create_all()


# --- Access Control Decorators ---

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if current_user.email != SUPER_ADMIN_EMAIL:
            flash("Access denied. This section is reserved for the primary Administrator.", "danger")
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function


def patient_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if current_user.role != 'patient':
            flash('This feature is for patients only.', 'warning')
            return redirect(url_for('admin_dashboard'))
        return f(*args, **kwargs)
    return decorated_function


# --- Page Routes ---

@app.route("/")
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    login_role = session.get('login_role', current_user.role)
    if login_role == 'admin' and current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    return render_template("home.html")


@app.route("/about")
@login_required
def about():
    return render_template("about.html")


@app.route("/services")
@login_required
def services():
    return render_template("services.html")


@app.route("/contact")
@login_required
def contact():
    return render_template("contact.html")


@app.route("/dashboard")
@login_required
def dashboard():
    # Chart data is loaded via AJAX calls from the frontend
    return render_template("dashboard.html")


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    return render_template("admin_dashboard.html")


@app.route("/admin/patients")
@admin_required
def admin_patients():
    return render_template("admin_patients.html")


# --- Auth Routes ---

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email")
        password = request.form.get("password")
        remember = bool(request.form.get("remember"))

        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password, password):
            return render_template("login.html", error="Please check your login details and try again.")

        selected_role = request.form.get("role")

        if selected_role == 'admin' and email != SUPER_ADMIN_EMAIL:
            return render_template("login.html", error="Access denied. Only patient access is available for this account.")

        login_user(user, remember=remember)
        session['login_role'] = selected_role

        if selected_role == 'admin':
            return redirect(url_for("admin_dashboard"))

        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email         = request.form.get("email")
        name          = request.form.get("name")
        password      = request.form.get("password")
        selected_role = request.form.get("role")

        if selected_role == 'admin' and email != SUPER_ADMIN_EMAIL:
            return render_template("signup.html", error="Administrator registration is restricted. Please register as a Patient.")

        if len(password) < 8:
            return render_template("signup.html", error="Password must be at least 8 characters long.")
        if not re.search(r"\d", password):
            return render_template("signup.html", error="Password must contain at least one number.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return render_template("signup.html", error="Password must contain at least one special character.")

        if User.query.filter_by(email=email).first():
            return render_template("signup.html", error="Email address already exists.")

        final_role = 'admin' if email == SUPER_ADMIN_EMAIL else 'patient'

        new_user = User(
            email=email,
            name=name,
            password=generate_password_hash(password, method="pbkdf2:sha256"),
            role=final_role,
            admin_request_status='approved' if final_role == 'admin' else 'none'
        )
        db.session.add(new_user)
        db.session.commit()

        return render_template("login.html", success="Registration successful! Please login.")

    return render_template("signup.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# --- Risk Prediction ---

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method != "POST":
        return render_template("predict.html", result=None, gauge=None)

    try:
        # Read the form values with sensible defaults
        user_input = {
            'age':                                 int(request.form.get('age', 30)),
            'gender':                              request.form.get('gender', 'Male'),
            'bmi':                                 float(request.form.get('bmi', 25.0)),
            'hba1c':                               float(request.form.get('hba1c', 5.5)),
            'glucose_fasting':                     int(request.form.get('glucose_fasting', 100)),
            'systolic_bp':                         int(request.form.get('systolic_bp', 120)),
            'diastolic_bp':                        int(request.form.get('diastolic_bp', 80)),
            'cholesterol_total':                   int(request.form.get('cholesterol_total', 200)),
            'physical_activity_minutes_per_week':  int(request.form.get('physical_activity_minutes_per_week', 150)),
            'family_history_diabetes':             int(request.form.get('family_history_diabetes', 0))
        }

        # Basic sanity checks — clamp obviously bad values back to safe defaults
        if not (0 < user_input['age'] < 120):    user_input['age']   = 30
        if not (10 < user_input['bmi'] < 60):    user_input['bmi']   = 25
        if not (3 < user_input['hba1c'] < 20):   user_input['hba1c'] = 5.5

        # Build the full input DataFrame the model expects.
        # For columns not collected from the user, we fall back to 0 (numerical) or "Unknown" (categorical).
        input_data = {}
        for col in num_cols:
            input_data[col] = user_input.get(col, 0)
        for col in cat_cols:
            input_data[col] = user_input.get(col, "Unknown")

        input_df = pd.DataFrame([input_data])

        # Encode categorical columns using the saved label encoders
        for col in cat_cols:
            if col in input_df.columns:
                try:
                    encoder = encoders[col]
                    val = input_df.loc[0, col]
                    if val in encoder.classes_:
                        input_df.loc[0, col] = encoder.transform([val])[0]
                    else:
                        input_df.loc[0, col] = -1  # unknown category
                except Exception:
                    pass

        # Run the model
        pred_prob  = model.predict_proba(input_df)[0]
        pred_class = int(np.argmax(pred_prob))
        risk_label = {0: "Low", 1: "Medium", 2: "High"}.get(pred_class, "Low")

        # Map the class + confidence to a 0–99 score for the gauge
        final_score = min(int(pred_class * 33 + pred_prob[pred_class] * 33), 99)

        # Pick a color based on the risk band
        if final_score < 33:
            bar_color = "#28a745"   # green
        elif final_score < 66:
            bar_color = "#ffc107"   # amber
        else:
            bar_color = "#dc3545"   # red

        # Build the donut gauge chart
        gauge_fig = go.Figure(data=[go.Pie(
            values=[final_score, 100 - final_score],
            labels=['Risk Score', 'Remaining'],
            hole=0.7,
            marker=dict(colors=[bar_color, '#f0f0f0'], line=dict(color='white', width=3)),
            textinfo='none',
            hoverinfo='label+percent',
            direction='clockwise',
            sort=False
        )])
        gauge_fig.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            height=320,
            annotations=[dict(
                text=f'<b>{final_score}%</b>',
                x=0.5, y=0.5,
                font=dict(size=64, color=bar_color, family='Inter'),
                showarrow=False
            )]
        )
        gauge_html = pio.to_html(gauge_fig, full_html=False, config={'displayModeBar': False}, div_id="riskGaugePlot")

        # Calculate impact scores for each key metric
        hba1c_impact = max(0, (user_input['hba1c'] - 5.7) * 20)
        bmi_impact   = max(0, (user_input['bmi'] - 25) * 5)
        bp_impact    = max(0, (user_input['systolic_bp'] - 120) * 0.5)
        act_impact   = max(0, (150 - user_input['physical_activity_minutes_per_week']) * 0.1)

        # Impact on risk score bar chart
        impact_df  = pd.DataFrame([
            {"Factor": "Blood Sugar", "Impact": hba1c_impact},
            {"Factor": "Body Mass",   "Impact": bmi_impact},
            {"Factor": "BP Level",    "Impact": bp_impact},
            {"Factor": "Sedentary",   "Impact": act_impact},
        ]).sort_values("Impact", ascending=True)
        impact_fig = px.bar(impact_df, x="Impact", y="Factor", orientation='h',
                            title="Impact on Risk Score", template="plotly_white",
                            color="Impact", color_continuous_scale='Reds')
        impact_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250,
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        impact_html = pio.to_html(impact_fig, full_html=False, config={'displayModeBar': False})

        # Your stats vs clinical targets
        v_df  = pd.DataFrame([
            {"Metric": "HbA1c (%)", "You": user_input['hba1c'],        "Target": 5.6},
            {"Metric": "Glucose",   "You": user_input['glucose_fasting'], "Target": 100},
            {"Metric": "BMI",       "You": user_input['bmi'],           "Target": 24.9},
        ]).melt(id_vars="Metric", var_name="Type", value_name="Value")
        v_fig = px.bar(v_df, x="Metric", y="Value", color="Type", barmode="group",
                       title="Your Stats vs Targets", template="plotly_white")
        v_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        v_html = pio.to_html(v_fig, full_html=False, config={'displayModeBar': False})

        # Wellness dimension scores
        dim_df  = pd.DataFrame({
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
        r_html = pio.to_html(r_fig, full_html=False, config={'displayModeBar': False})

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
        bp_html = pio.to_html(bp_fig, full_html=False, config={'displayModeBar': False})

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
        c_html = pio.to_html(c_fig, full_html=False, config={'displayModeBar': False})

        # Build recommendations and alerts
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

        # Pad with general wellness tips if we have fewer than 5 entries
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

        # Simple health summary based on risk level
        if risk_label == "High":
            summary = "Your analysis indicates a high priority for health improvements."
        elif risk_label == "Medium":
            summary = "Your results show some areas that need attention to prevent future risks."
        else:
            summary = "You are currently in a good health range! Consistency is your best friend."

        suggestions = []
        if hba1c_impact > 10: suggestions.append("Consider reducing sugar intake and monitoring HbA1c.")
        if bmi_impact   > 10: suggestions.append("Aim for a balanced diet to reach a healthier weight range.")
        if user_act < 150:    suggestions.append("Try to add 15-20 minutes of walking to your daily routine.")
        if not suggestions:    suggestions.append("Keep maintaining your excellent daily health habits.")

        result = {
            "risk":          risk_label,
            "score":         final_score,
            "recs":          recs[:5],
            "alerts":        alerts,
            "timestamp":     datetime.now().strftime("%B %d, %Y at %I:%M %p"),
            "summary_title": "Overall Assessment",
            "summary_text":  summary,
            "suggestions":   suggestions,
            "charts": {
                "impact": impact_html,
                "vitals": v_html,
                "radar":  r_html,
                "bp_map": bp_html,
                "curve":  c_html,
            }
        }

        # Save the record to the database
        db.session.add(MedicalRecord(
            user_id=current_user.id,
            age=user_input['age'],
            gender=user_input['gender'],
            bmi=user_input['bmi'],
            hba1c=user_input['hba1c'],
            glucose=user_input['glucose_fasting'],
            systolic_bp=user_input['systolic_bp'],
            diastolic_bp=user_input['diastolic_bp'],
            cholesterol=user_input['cholesterol_total'],
            activity_minutes=user_input['physical_activity_minutes_per_week'],
            family_history=user_input['family_history_diabetes'],
            risk_score=final_score,
            risk_label=risk_label
        ))
        db.session.commit()

        flash("Clinical analysis complete!", "success")
        return render_template("predict.html", result=result, gauge=gauge_html)

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        result = {"risk": "Error", "score": 0, "recs": [("❌", "Error", "Please check your inputs and try again.")]}
        return render_template("predict.html", result=result, gauge=None)


# --- AI Chatbot ---

@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data     = request.json
    user_msg = data.get("message", "").lower()

    # Grab the user's latest health record to personalise the reply
    latest_record     = MedicalRecord.query.filter_by(user_id=current_user.id).order_by(MedicalRecord.timestamp.desc()).first()
    vitals_context = ""
    if latest_record:
        vitals_context = (
            f"Note: The user's current health metrics are: "
            f"Age: {latest_record.age}, Gender: {latest_record.gender}, BMI: {latest_record.bmi}, "
            f"HbA1c: {latest_record.hba1c}%, Fasting Glucose: {latest_record.glucose} mg/dL, "
            f"Blood Pressure: {latest_record.systolic_bp}/{latest_record.diastolic_bp} mmHg, "
            f"Risk Score: {latest_record.risk_score}%, Risk Category: {latest_record.risk_label}."
        )

    # Keyword-based local expert responses (works offline)
    expert_knowledge = [
        (["risk", "calculate", "score"],      "Our Risk Prediction tool uses the CatBoost algorithm trained on 100,000 clinical records. Based on your inputs, we calculate a percentage-based risk score."),
        (["dashboard", "track", "history"],   "The Dashboard visualizes your health trends, metabolic scores, and how different factors like Activity and BMI impact your overall risk."),
        (["hba1c", "hb1ac", "a1c", "hb a1c"], "HbA1c measures your average blood sugar over 3 months. Normal: <5.7%, Pre-diabetes: 5.7%-6.4%, Diabetes: 6.5%+. Monitoring this is crucial for long-term health."),
        (["glucose", "sugar", "fasting"],     "Fasting glucose should be 70-99 mg/dL. Values above 126 mg/dL often indicate diabetes. Always confirm with clinical tests."),
        (["diet", "food", "eat"],             "Focus on high-fiber, low-glycemic foods. Leafy greens, nuts, seeds, and whole grains help stabilize blood sugar. Limit refined sugars and processed carbs."),
        (["exercise", "activity", "walk", "gym"], "Aim for 150 mins of moderate activity weekly. Exercise improves insulin sensitivity and helps manage weight effectively."),
        (["bmi", "weight", "height"],         "BMI is height-to-weight ratio. Healthy: 18.5-24.9. Higher BMI often correlates with increased insulin resistance."),
        (["symptoms", "signs", "thirsty", "urination"], "Keep an eye out for extreme thirst, frequent urination, fatigue, and blurred vision. See a doctor if these persist."),
        (["analyze", "status", "report", "my risk"], "To analyze your health status, please use the 'Prediction' tab first. Once you have a score, I can help explain what the numbers mean for your wellness journey."),
        (["hello", "hi", "hey"],              "Hello! I'm your HealthRisk AI. I can analyze your vitals, explain diabetes risk, or offer lifestyle tips. How can I assist you today?"),
        (["thanks", "thank you"],             "You're very welcome! Stay proactive about your health."),
    ]

    for keys, reply in expert_knowledge:
        if any(k in user_msg for k in keys):
            # Let deeper queries fall through to Gemini when available
            if any(k in user_msg for k in ["analyze", "status", "report"]) and HAS_GEMINI:
                continue
            return {"reply": reply}

    # Fall back to Gemini if available
    if not HAS_GEMINI:
        fallback = "I've matched your question to my local medical database, but for complex queries I need the Gemini AI key connected."
        if vitals_context and ("analyze" in user_msg or "status" in user_msg):
            fallback += f" Based on your latest record (Risk: {latest_record.risk_label}), focus on the lifestyle tips from your Prediction results."
        return {"reply": fallback}

    try:
        system_prompt = (
            "You are the Advanced AI Brain of HealthRisk AI, a specialized medical analytics assistant. "
            "Be data-driven, clinical yet empathetic. "
            f"{vitals_context} "
            "Only answer health, wellness, and medical analytics questions. "
            "If the user shares vitals, explain their clinical significance. "
            "Always remind the user that you are an AI and they should consult a doctor for diagnosis. "
            "If no vitals are available, ask them to run the Risk Prediction tool first."
        )
        gemini_model = genai.GenerativeModel('gemini-flash-latest')
        response     = gemini_model.generate_content(f"{system_prompt}\n\nUser: {data.get('message', '')}\nAssistant:")
        return {"reply": response.text}
    except Exception as e:
        app.logger.error(f"Gemini chat error: {e}")
        return {"reply": "I'm having trouble connecting to the cloud right now. I can still answer basic questions about diet, exercise, and vitals!"}


@app.route("/api/analyze_result", methods=["POST"])
@login_required
def analyze_result():
    if not HAS_GEMINI:
        return {"error": "Advanced AI (Gemini) is not connected. Please contact the administrator."}, 400

    result_data = request.json.get("result")
    if not result_data:
        return {"error": "No result data provided."}, 400

    try:
        prompt = (
            "You are a clinical health analyst. Analyze the following diabetes risk assessment results "
            "and provide actionable insights in 3 short paragraphs: "
            "1. Clinical Significance: what do the numbers (HbA1c, Glucose, BP) mean together? "
            "2. Critical Priorities: what is the single most important thing this user should change? "
            "3. Long-term Outlook: what is the likely trajectory if changes are (or aren't) made? "
            f"\n\nResults: {result_data}"
            "\n\nBe professional, data-focused, and empathetic. Include a medical disclaimer."
        )
        gemini_model = genai.GenerativeModel('gemini-flash-latest')
        response     = gemini_model.generate_content(prompt)
        return {"analysis": response.text}
    except Exception as e:
        app.logger.error(f"Gemini analysis error: {e}")
        return {"error": "Failed to generate AI analysis. Please try again later."}, 500


# --- Admin API ---

@app.route("/api/admin/patients")
@admin_required
def api_admin_patients():
    all_patients = User.query.filter_by(role='patient').all()
    patients     = [p for p in all_patients if MedicalRecord.query.filter_by(user_id=p.id).first()]

    patient_list          = []
    risk_counts           = {"High": 0, "Medium": 0, "Low": 0}
    age_groups            = {"0-20": 0, "21-40": 0, "41-60": 0, "61+": 0}
    gender_counts         = {"Male": 0, "Female": 0, "Other": 0, "Unknown": 0}
    family_history_counts = {"Yes": 0, "No": 0}
    hba1c_values, glucose_values, bp_systolic_values, activity_values = [], [], [], []

    for p in patients:
        rec = MedicalRecord.query.filter_by(user_id=p.id).order_by(MedicalRecord.timestamp.desc()).first()
        risk_label = rec.risk_label if rec else "No Data"
        if risk_label in risk_counts:
            risk_counts[risk_label] += 1

        if rec:
            age = rec.age
            if age <= 20:   age_groups["0-20"]  += 1
            elif age <= 40: age_groups["21-40"] += 1
            elif age <= 60: age_groups["41-60"] += 1
            else:           age_groups["61+"]   += 1

            family_history_counts["Yes" if rec.family_history == 1 else "No"] += 1

            hba1c_values.append(rec.hba1c)
            glucose_values.append(rec.glucose)
            bp_systolic_values.append(rec.systolic_bp)
            activity_values.append(rec.activity_minutes)

            gender = rec.gender if rec.gender else "Unknown"
            if gender in gender_counts:
                gender_counts[gender] += 1
            else:
                gender_counts["Other"] += 1

        patient_list.append({
            "id":           p.id,
            "name":         p.name,
            "email":        p.email,
            "latest_risk":  risk_label,
            "latest_score": rec.risk_score if rec else None,
            "last_visit":   rec.timestamp.strftime("%Y-%m-%d") if rec else "N/A",
        })

    total_count    = len(patients)
    high_risk_count = risk_counts["High"]
    scores         = [p['latest_score'] for p in patient_list if p['latest_score'] is not None]
    avg_risk_val   = round(np.mean(scores), 1) if scores else 0
    avg_hba1c      = round(np.mean(hba1c_values), 1) if hba1c_values else 0

    if total_count == 0:
        system_status, status_class = "N/A", "secondary"
    else:
        high_risk_pct = (high_risk_count / total_count) * 100
        if high_risk_pct > 30:
            system_status, status_class = "Critical (Bad)", "danger"
        elif high_risk_pct > 15:
            system_status, status_class = "Warning (Moderate)", "warning"
        else:
            system_status, status_class = "Optimal (Good)", "success"

    # Collect clinical alerts for the most at-risk patients
    clinical_alerts = []
    for p in patients:
        rec = MedicalRecord.query.filter_by(user_id=p.id).order_by(MedicalRecord.timestamp.desc()).first()
        if not rec:
            continue
        if rec.risk_label == "High":
            clinical_alerts.append({"patient": p.name, "type": "High Risk",    "value": f"{rec.risk_score}%", "class": "danger"})
        elif rec.glucose > 180:
            clinical_alerts.append({"patient": p.name, "type": "High Glucose", "value": str(rec.glucose),    "class": "warning"})
        elif rec.systolic_bp > 140:
            clinical_alerts.append({"patient": p.name, "type": "High BP",      "value": str(rec.systolic_bp), "class": "warning"})

    return {
        "patients": patient_list,
        "stats": {
            "total_patients": total_count,
            "high_risk_count": high_risk_count,
            "avg_risk":       avg_risk_val,
            "avg_hba1c":      avg_hba1c,
            "system_status":  system_status,
            "status_class":   status_class,
        },
        "charts": {
            "risk_dist":            {"labels": list(risk_counts.keys()),           "values": list(risk_counts.values())},
            "gender_dist":          {"labels": list(gender_counts.keys()),         "values": list(gender_counts.values())},
            "age_dist":             {"labels": list(age_groups.keys()),            "values": list(age_groups.values())},
            "family_history":       {"labels": list(family_history_counts.keys()), "values": list(family_history_counts.values())},
            "registration_trend":   {"labels": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                                     "values": [2, 5, 3, 8, 4, max(1, len(patients)//2), len(patients)]},
            "clinical_distributions": {
                "glucose":     glucose_values,
                "bp_systolic": bp_systolic_values,
                "activity":    activity_values,
                "hba1c":       hba1c_values,
            },
        },
        "alerts": clinical_alerts[:5],
    }


@app.route("/api/community/stats")
@login_required
def api_community_stats():
    all_patients = User.query.filter_by(role='patient').all()
    patients     = [p for p in all_patients if MedicalRecord.query.filter_by(user_id=p.id).first()]

    risk_counts           = {"High": 0, "Medium": 0, "Low": 0}
    age_groups            = {"0-20": 0, "21-40": 0, "41-60": 0, "61+": 0}
    family_history_counts = {"Yes": 0, "No": 0}
    hba1c_values, glucose_values, bp_systolic_values, activity_values = [], [], [], []

    for p in patients:
        rec = MedicalRecord.query.filter_by(user_id=p.id).order_by(MedicalRecord.timestamp.desc()).first()
        if not rec:
            continue

        if rec.risk_label in risk_counts:
            risk_counts[rec.risk_label] += 1

        age = rec.age
        if age <= 20:   age_groups["0-20"]  += 1
        elif age <= 40: age_groups["21-40"] += 1
        elif age <= 60: age_groups["41-60"] += 1
        else:           age_groups["61+"]   += 1

        family_history_counts["Yes" if rec.family_history == 1 else "No"] += 1

        hba1c_values.append(rec.hba1c)
        glucose_values.append(rec.glucose)
        bp_systolic_values.append(rec.systolic_bp)
        activity_values.append(rec.activity_minutes)

    total_count     = len(patients)
    high_risk_count = risk_counts["High"]
    scores          = [MedicalRecord.query.filter_by(user_id=p.id).order_by(MedicalRecord.timestamp.desc()).first().risk_score for p in patients]
    avg_risk_val    = round(np.mean(scores), 1) if scores else 0
    avg_hba1c       = round(np.mean(hba1c_values), 1) if hba1c_values else 0

    if total_count == 0:
        system_status, status_class = "N/A", "secondary"
    else:
        high_risk_pct = (high_risk_count / total_count) * 100
        if high_risk_pct > 30:
            system_status, status_class = "Critical", "danger"
        elif high_risk_pct > 15:
            system_status, status_class = "Warning", "warning"
        else:
            system_status, status_class = "Excellent", "success"

    return {
        "stats": {
            "total_patients": total_count,
            "high_risk_count": high_risk_count,
            "avg_risk":       avg_risk_val,
            "avg_hba1c":      avg_hba1c,
            "system_status":  system_status,
            "status_class":   status_class,
        },
        "charts": {
            "risk_dist":      {"labels": list(risk_counts.keys()),            "values": list(risk_counts.values())},
            "age_dist":       {"labels": list(age_groups.keys()),             "values": list(age_groups.values())},
            "family_history": {"labels": list(family_history_counts.keys()),  "values": list(family_history_counts.values())},
            "clinical_distributions": {
                "glucose":     glucose_values,
                "bp_systolic": bp_systolic_values,
                "activity":    activity_values,
            },
        },
    }


@app.route("/admin/patient/<int:user_id>")
@admin_required
def admin_patient_detail(user_id):
    patient = User.query.get_or_404(user_id)
    if patient.role != 'patient':
        flash('Requested user is not a patient.', 'warning')
        return redirect(url_for('admin_dashboard'))
    return render_template("admin_patient_detail.html", patient=patient)


@app.route("/api/admin/patient/<int:user_id>/history")
@admin_required
def api_admin_patient_history(user_id):
    records = MedicalRecord.query.filter_by(user_id=user_id).order_by(MedicalRecord.timestamp.asc()).all()
    history = [{
        "timestamp":  r.timestamp.strftime("%Y-%m-%d %H:%M"),
        "risk_score": r.risk_score,
        "risk_label": r.risk_label,
        "bmi":        r.bmi,
        "hba1c":      r.hba1c,
        "glucose":    r.glucose,
        "bp":         f"{r.systolic_bp}/{r.diastolic_bp}",
    } for r in records]
    return {"history": history}


@app.route("/api/personal-history")
@login_required
def personal_history():
    records = MedicalRecord.query.filter_by(user_id=current_user.id).order_by(MedicalRecord.timestamp.asc()).all()

    if not records:
        return {"has_data": False}

    history = []
    for r in records:
        if r.risk_score > 50:
            risk_cat = "High"
        elif r.risk_score > 20:
            risk_cat = "Medium"
        else:
            risk_cat = "Low"

        history.append({
            "timestamp":      r.timestamp.strftime("%H:%M"),
            "full_timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M"),
            "time_only":      r.timestamp.strftime("%H:%M"),
            "risk_score":     r.risk_score,
            "risk_cat":       risk_cat,
            "bmi":            r.bmi,
            "hba1c":          r.hba1c,
            "glucose":        r.glucose,
            "systolic":       r.systolic_bp,
            "diastolic":      r.diastolic_bp,
            "cholesterol":    r.cholesterol,
            "activity":       r.activity_minutes,
        })

    df = pd.DataFrame(history)
    latest = history[-1]

    # Shared layout for all charts — transparent background, clean look
    chart_layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=13, color="#212529"),
        title_font=dict(size=16, color="#212529"),
        margin=dict(l=60, r=30, t=70, b=60),
        xaxis=dict(showgrid=True, gridcolor='#f1f3f5', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#f1f3f5', zeroline=False),
    )

    # Risk score trend with up/down signal markers
    df['diff'] = df['risk_score'].diff().fillna(0)
    f_risk = go.Figure()
    f_risk.add_trace(go.Scatter(
        x=df['timestamp'], y=df['risk_score'],
        mode='lines', line=dict(color='#adb5bd', width=2, dash='dot'), name='Trend'
    ))
    up_moves   = df[df['diff'] > 0]
    down_moves = df[df['diff'] < 0]
    f_risk.add_trace(go.Scatter(
        x=up_moves['timestamp'], y=up_moves['risk_score'], mode='markers',
        marker=dict(symbol='triangle-up', size=16, color='#198754', line=dict(width=1, color='white')),
        name='Increase'
    ))
    f_risk.add_trace(go.Scatter(
        x=down_moves['timestamp'], y=down_moves['risk_score'], mode='markers',
        marker=dict(symbol='triangle-down', size=16, color='#dc3545', line=dict(width=1, color='white')),
        name='Decrease'
    ))
    f_risk.add_hrect(y0=0,  y1=20,  fillcolor="#198754", opacity=0.05, layer="below", line_width=0)
    f_risk.add_hrect(y0=20, y1=50,  fillcolor="#ffc107", opacity=0.05, layer="below", line_width=0)
    f_risk.add_hrect(y0=50, y1=100, fillcolor="#dc3545", opacity=0.05, layer="below", line_width=0)
    f_risk.update_layout(title="Risk Volatility Signal Ticker", showlegend=False, **chart_layout)

    # Glucose bar chart
    f_glucose = px.bar(df, x="timestamp", y="glucose", title="Glucose Levels")
    f_glucose.update_traces(marker_color='#fd7e14', marker_line_width=0)
    f_glucose.update_layout(**chart_layout)

    # Blood pressure line chart
    f_bp = px.line(df, x="timestamp", y=["systolic", "diastolic"], title="Blood Pressure Analysis")
    f_bp.update_traces(line=dict(width=4))
    f_bp.data[0].line.color = '#dc3545'
    f_bp.data[1].line.color = '#ffc107'
    f_bp.update_layout(legend=dict(orientation="h", y=1.1, x=0), **chart_layout)

    # Activity bar chart
    f_activity = px.bar(df, x="activity", y="timestamp", orientation='h', title="Activity Metrics")
    f_activity.update_traces(marker_color='#198754', marker_line_width=0)
    f_activity.update_layout(**chart_layout)

    # Cholesterol bar chart
    f_cholesterol = px.bar(df, x="timestamp", y="cholesterol", title="Cholesterol Markers")
    f_cholesterol.update_traces(marker_color='#6f42c1', marker_line_width=0)
    f_cholesterol.update_layout(**chart_layout)

    # Risk category breakdown donut
    risk_counts_df = df["risk_cat"].value_counts().reset_index()
    risk_counts_df.columns = ["Category", "Count"]
    f_pie = px.pie(
        risk_counts_df, values="Count", names="Category", hole=0.5,
        title="Predictive Stability Summary",
        color="Category",
        color_discrete_map={"High": "#dc3545", "Medium": "#ffc107", "Low": "#198754"}
    )
    f_pie.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1), **chart_layout)

    return {
        "has_data": True,
        "latest_stats": {
            "score":           latest["risk_score"],
            "glucose":         latest["glucose"],
            "bmi":             latest["bmi"],
            "hba1c":           latest["hba1c"],
            "prediction_time": latest["time_only"],
        },
        "charts": {
            "risk_trend":       json.loads(pio.to_json(f_risk)),
            "metabolic_trend":  json.loads(pio.to_json(f_glucose)),
            "bp_trend":         json.loads(pio.to_json(f_bp)),
            "activity_trend":   json.loads(pio.to_json(f_activity)),
            "cholesterol_trend": json.loads(pio.to_json(f_cholesterol)),
            "risk_pie":         json.loads(pio.to_json(f_pie)),
        },
    }


if __name__ == "__main__":
    app.run(debug=True)
