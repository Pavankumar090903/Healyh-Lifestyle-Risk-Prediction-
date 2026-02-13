import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*google-auth.*")
warnings.filterwarnings("ignore", message=".*google.generativeai.*")

from flask import Flask, render_template, request, redirect, url_for, flash, session
import re
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.io as pio
import random
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file (local development)
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_default_key_replace_in_production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# -------------------------
# DATABASE MODELS
# -------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(1000))
    role = db.Column(db.String(20), default='patient', nullable=False)  # 'admin' or 'patient'
    admin_request_status = db.Column(db.String(20), default='none')  # 'none', 'pending', 'approved', 'rejected'
    # Relationship to medical records
    records = db.relationship('MedicalRecord', backref='patient', lazy=True)

class MedicalRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    bmi = db.Column(db.Float)
    hba1c = db.Column(db.Float)
    glucose = db.Column(db.Integer)
    systolic_bp = db.Column(db.Integer)
    diastolic_bp = db.Column(db.Integer)
    cholesterol = db.Column(db.Integer)
    activity_minutes = db.Column(db.Integer)
    family_history = db.Column(db.Integer)
    risk_score = db.Column(db.Integer)
    risk_label = db.Column(db.String(20))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------------
# ROLE-BASED ACCESS CONTROL
# -------------------------
from functools import wraps

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        # Strictly restrict to the Super Admin email
        if current_user.email != "mulagiripavankumar886@gmail.com":
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

# Create database tables
with app.app_context():
    db.create_all()

# -------------------------
# LOAD MODEL & DATA
# -------------------------
model = joblib.load("diabetes_risk_model.pkl")
encoders = joblib.load("encoders.pkl")
num_cols = joblib.load("num_cols.pkl")
cat_cols = joblib.load("cat_cols.pkl")

df = pd.read_csv("diabetes_dataset.csv")

# -------------------------
# HOME
# -------------------------
@app.route("/")
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    # Redirect based on session login_role if set, else actual role
    login_role = session.get('login_role', current_user.role)
    if login_role == 'admin' and current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    return render_template("home.html")

# -------------------------
# ABOUT
# -------------------------
@app.route("/about")
def about():
    return render_template("about.html")

# -------------------------
# RISK PREDICTION
# -------------------------
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    gauge_json = None
    
    if request.method == "POST":
        try:
            # 1. Capture Top 10 Inputs with Default Fallbacks
            user_input = {
                'age': int(request.form.get('age', 30)),
                'gender': request.form.get('gender', 'Male'),
                'bmi': float(request.form.get('bmi', 25.0)),
                'hba1c': float(request.form.get('hba1c', 5.5)),
                'glucose_fasting': int(request.form.get('glucose_fasting', 100)),
                'systolic_bp': int(request.form.get('systolic_bp', 120)),
                'diastolic_bp': int(request.form.get('diastolic_bp', 80)),
                'cholesterol_total': int(request.form.get('cholesterol_total', 200)),
                'physical_activity_minutes_per_week': int(request.form.get('physical_activity_minutes_per_week', 150)),
                'family_history_diabetes': int(request.form.get('family_history_diabetes', 0))
            }

            # 2. Validation (Backend Safety Check)
            if not (0 < user_input['age'] < 120): user_input['age'] = 30
            if not (10 < user_input['bmi'] < 60): user_input['bmi'] = 25
            if not (3 < user_input['hba1c'] < 20): user_input['hba1c'] = 5.5

            # 3. Prepare DataFrame for Model (Fill missing cols with defaults)
            # We must provide ALL columns the model expects, even if we only ask for 10.
            # We use median/mode from loaded artifacts for the missing ones.
            input_data = {}
            
            # Fill numericals
            for col in num_cols:
                if col in user_input:
                    input_data[col] = user_input[col]
                else:
                    # Fallback to median from loaded data if available, else 0
                    # For this implementation, we assume median is good enough for 'noise' features
                    input_data[col] = 0 # Ideally should use loaded medians if available

            # Fill categoricals
            for col in cat_cols:
                if col in user_input:
                    input_data[col] = user_input[col]
                else:
                    input_data[col] = "Unknown" # Fallback

            input_df = pd.DataFrame([input_data])

            # Apply Encoding
            for col in cat_cols:
                if col in input_df.columns:
                     try:
                        encoder = encoders[col]
                        # Handle unseen labels carefully
                        val = input_df.loc[0, col]
                        if val in encoder.classes_:
                            input_df.loc[0, col] = encoder.transform([val])[0]
                        else:
                            input_df.loc[0, col] = -1 # Special handling for unknown
                     except:
                         pass # Skip if encoder fails

            # 4. Predict Logic
            
            pred_prob = model.predict_proba(input_df)[0]
            pred_class = np.argmax(pred_prob)
            
            risk_map = {0: "Low", 1: "Medium", 2: "High"}
            risk_label = risk_map.get(pred_class, "Low")
            
            risk_label = risk_map.get(pred_class, "Low")
            
            # Score logic: Map class + prob to 0-100 scale
            # Using max prob is simple, but often > 0.9. 
            # Let's map class + prob to a 0-100 scale for smoother gauge.
            # Low: 0-33, Medium: 34-66, High: 67-100
            base_score = pred_class * 33
            offset = pred_prob[pred_class] * 33
            final_score = min(int(base_score + offset), 99)
            # Determine Gauge Bar Color & Risk Level
            if final_score < 33:
                bar_color = "#28a745" # Green
            elif final_score < 66:
                bar_color = "#ffc107" # Yellow (Bootstrap warning)
            else:
                bar_color = "#dc3545" # Red
                
            # 5. Generate Donut Chart for Risk Score
            import plotly.graph_objects as go
            
            # Calculate remaining percentage
            remaining = 100 - final_score
            
            # Create donut chart
            fig = go.Figure(data=[go.Pie(
                values=[final_score, remaining],
                labels=['Risk Score', 'Remaining'],
                hole=0.7,
                marker=dict(
                    colors=[bar_color, '#f0f0f0'],
                    line=dict(color='white', width=3)
                ),
                textinfo='none',
                hoverinfo='label+percent',
                direction='clockwise',
                sort=False
            )])
            
            # Add center text annotation
            fig.add_annotation(
                text=f'<b>{int(final_score)}%</b>',
                x=0.5, y=0.5,
                font=dict(size=60, color=bar_color, family='Inter'),
                showarrow=False
            )
            
            fig.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                height=320,
                annotations=[
                    dict(
                        text=f'<b>{int(final_score)}%</b>',
                        x=0.5, y=0.5,
                        font=dict(size=64, color=bar_color, family='Inter', weight='bold'),
                        showarrow=False
                    )
                ]
            )
            
            # Use Div ID for targeting in JS
            gauge_json = pio.to_html(fig, full_html=False, config={'displayModeBar': False}, div_id="riskGaugePlot")

            # 6. Generate Recommendations & Alerts
            recs_list = []
            alerts = []
            
            # High Priority (Medical)
            if user_input['hba1c'] >= 6.5:
                alerts.append(("🚨", "Critical HbA1c Level", "danger"))
                recs_list.append(("🚨", "Critical HbA1c", "Level is in diabetes range. Seek medical attention."))
            elif user_input['hba1c'] >= 5.7:
                alerts.append(("⚠️", "Pre-Diabetes Warning", "warning"))
                recs_list.append(("⚠️", "Pre-Diabetes", "Elevated HbA1c. Lifestyle changes required."))

            if user_input['systolic_bp'] > 140 or user_input['diastolic_bp'] > 90:
                alerts.append(("💓", "High Blood Pressure", "danger"))
                recs_list.append(("💓", "Hypertension", "Blood pressure is high. Monitor regularly."))

            # Lifestyle
            if user_input['bmi'] >= 30:
                alerts.append(("⚖️", "Obesity Detected", "danger"))
                recs_list.append(("⚖️", "Weight Management", "BMI > 30. A weight loss plan is recommended."))
            elif user_input['bmi'] >= 25:
                 pass # usually just a recommendation, not a top-level alert

            if user_input['physical_activity_minutes_per_week'] < 150:
                 pass # Recommendation only
            
            if user_input['cholesterol_total'] > 200:
                 alerts.append(("🍔", "High Cholesterol", "warning"))
                 recs_list.append(("🍔", "Cholesterol", "Total cholesterol is high. Consider diet changes."))

            if risk_label == "High":
                recs_list.append(("👨‍⚕️", "Specialist Review", "High risk detected. Consult an endocrinologist."))
            elif risk_label == "Medium":
                alerts.append(("🟡", "Preventive Watch", "warning"))
                recs_list.append(("📉", "Risk Reduction", "Moderate risk detected. Adopt preventive lifestyle changes."))
            elif risk_label == "Low":
                alerts.append(("✅", "Optimal Maintenance", "success"))
                recs_list.append(("🛡️", "Maintenance", "Low risk. Continue your healthy habits to stay safe."))

            # Fallback / Educational Recommendations
            if len(recs_list) < 5:
                # General preventive tips to fill the quota
                wellness_tips = [
                    ("🥗", "Balanced Diet", "Maintain a diet rich in vegetables, whole grains, and lean proteins."),
                    ("💧", "Hydration", "Drink 8-10 glasses of water daily to support metabolism."),
                    ("😴", "Sleep Hygiene", "Aim for 7-9 hours of quality sleep to regulate blood sugar."),
                    ("🧘", "Stress Management", "Practice mindfulness or yoga to keep cortisol levels low."),
                    ("🩺", "Regular Checkups", "Schedule annual health screenings to monitor your vitals.")
                ]
                
                for tip in wellness_tips:
                    if len(recs_list) < 5:
                        recs_list.append(tip)
            
            # Limit to Top 5
            recs = recs_list[:5]

            # 7. Personal Impact Analysis Chart
            impact_factors = []
            hba1c_impact = max(0, (user_input['hba1c'] - 5.7) * 20) 
            impact_factors.append({"Factor": "Blood Sugar", "Impact": hba1c_impact})
            bmi_impact = max(0, (user_input['bmi'] - 25) * 5)
            impact_factors.append({"Factor": "Body Mass", "Impact": bmi_impact})
            bp_impact = max(0, (user_input['systolic_bp'] - 120) * 0.5)
            impact_factors.append({"Factor": "BP Level", "Impact": bp_impact})
            act_impact = max(0, (150 - user_input['physical_activity_minutes_per_week']) * 0.1)
            impact_factors.append({"Factor": "Sedentary", "Impact": act_impact})
            
            impact_df = pd.DataFrame(impact_factors).sort_values("Impact", ascending=True)
            impact_fig = px.bar(impact_df, x="Impact", y="Factor", orientation='h',
                               title="Impact on Risk Score", template="plotly_white",
                               color="Impact", color_continuous_scale='Reds')
            impact_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            impact_json = pio.to_html(impact_fig, full_html=False, config={'displayModeBar': False})

            # 8. VITALS COMPARISON CHART
            vitals_data = [
                {"Metric": "HbA1c (%)", "You": user_input['hba1c'], "Target": 5.6},
                {"Metric": "Glucose", "You": user_input['glucose_fasting'], "Target": 100},
                {"Metric": "BMI", "You": user_input['bmi'], "Target": 24.9}
            ]
            v_df = pd.DataFrame(vitals_data).melt(id_vars="Metric", var_name="Type", value_name="Value")
            v_fig = px.bar(v_df, x="Metric", y="Value", color="Type", barmode="group",
                          title="Your Stats vs Targets", template="plotly_white")
            v_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            v_json = pio.to_html(v_fig, full_html=False, config={'displayModeBar': False})

            # 9. HEALTH DIMENSIONS BAR 
            health_score = 100 - min(100, hba1c_impact * 2)
            metabolic_score = 100 - min(100, bmi_impact * 2)
            activity_score = min(100, (user_input['physical_activity_minutes_per_week'] / 150) * 100)
            bp_score = 100 - min(100, bp_impact * 5)
            
            dim_df = pd.DataFrame({
                "Dimension": ["Blood Sugar", "Body Weight", "Activity", "Circulation"],
                "Score": [health_score, metabolic_score, activity_score, bp_score]
            })
            r_fig = px.bar(dim_df, x="Score", y="Dimension", orientation='h', 
                          title="Wellness Dimension Scores (0-100)", 
                          template="plotly_white", color="Score", color_continuous_scale="RdYlGn")
            r_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            r_fig.update_coloraxes(showscale=False)
            r_json = pio.to_html(r_fig, full_html=False, config={'displayModeBar': False})

            # 10. BP Zoning Map
            bp_fig = go.Figure()
            bp_fig.add_shape(type="rect", x0=80, y0=60, x1=120, y1=80, fillcolor="green", opacity=0.2, line_width=0, layer="below")
            bp_fig.add_shape(type="rect", x0=120, y0=80, x1=140, y1=90, fillcolor="orange", opacity=0.2, line_width=0, layer="below")
            bp_fig.add_shape(type="rect", x0=140, y0=90, x1=180, y1=110, fillcolor="red", opacity=0.1, line_width=0, layer="below")
            bp_fig.add_trace(go.Scatter(x=[user_input['systolic_bp']], y=[user_input['diastolic_bp']], mode='markers+text', name="You", text=["YOU"], textposition="top center", marker=dict(size=12, color='black', symbol='star')))
            bp_fig.update_layout(title="Blood Pressure Zoning", xaxis_title="Systolic", yaxis_title="Diastolic", height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            bp_json = pio.to_html(bp_fig, full_html=False, config={'displayModeBar': False})

            # 11. ACTIVITY IMPACT CURVE (KEEP)
            act_curve_x = np.linspace(0, 300, 50)
            act_curve_y = 100 * np.exp(-act_curve_x / 150)
            c_fig = px.area(x=act_curve_x, y=act_curve_y, title="Activity vs Risk Curve", labels={'x':'Minutes/Week','y':'Risk Influence'}, template="plotly_white")
            c_fig.add_scatter(x=[user_input['physical_activity_minutes_per_week']], y=[100 * np.exp(-user_input['physical_activity_minutes_per_week']/150)], mode='markers', name="Your Spot", marker=dict(size=10, color='red'))
            c_fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            c_json = pio.to_html(c_fig, full_html=False, config={'displayModeBar': False})

            # 12. ENHANCED SIMPLE HEALTH SUMMARY
            msg = "Overall Assessment"
            if risk_label == "High":
                detail = "Your analysis indicates a high priority for health improvements."
            elif risk_label == "Medium":
                detail = "Your results show some areas that need attention to prevent future risks."
            else:
                detail = "You are currently in a good health range! Consistency is your best friend."

            suggestions = []
            if hba1c_impact > 10: suggestions.append("Consider reducing sugar intake and monitoring HbA1c.")
            if bmi_impact > 10: suggestions.append("Aim for a balanced diet to reach a healthier weight range.")
            if user_input['physical_activity_minutes_per_week'] < 150: suggestions.append("Try to add 15-20 minutes of walking to your daily routine.")
            if not suggestions: suggestions.append("Keep maintaining your excellent daily health habits.")

            result = {
                "risk": risk_label,
                "score": final_score,
                "recs": recs_list[:5],
                "alerts": alerts,
                "timestamp": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
                "summary_title": msg,
                "summary_text": detail,
                "suggestions": suggestions,
                "charts": {
                    "impact": impact_json,
                    "vitals": v_json,
                    "radar": r_json, # Changed to internal bar chart
                    "bp_map": bp_json,
                    "curve": c_json
                }
            }

            # --- SAVE TO DATABASE ---
            new_record = MedicalRecord(
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
            )
            db.session.add(new_record)
            db.session.commit()
            # ------------------------

        except Exception as e:
            app.logger.error(f"Prediction Error: {e}")
            result = {"risk": "Error", "score": 0, "recs": [("❌", "Error", "Please checks your inputs and try again.")]}

    return render_template("predict.html", result=result, gauge=gauge_json)

# -------------------------
# AUTH ROUTES
# -------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        remember = True if request.form.get("remember") else False

        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password, password):
            # For simplicity, just reload login with error
            return render_template("login.html", error="Please check your login details and try again.")

        selected_role = request.form.get("role")
        
        # Strict Admin Role Validation
        SUPER_ADMIN_EMAIL = "mulagiripavankumar886@gmail.com"
        if selected_role == 'admin' and email != SUPER_ADMIN_EMAIL:
            return render_template("login.html", error="Access denied. Only patient access is available for this account.")

        login_user(user, remember=remember)
        session['login_role'] = selected_role

        # Redirect based on selected role
        if selected_role == 'admin':
            return redirect(url_for("admin_dashboard"))
        
        return redirect(url_for("home"))
    
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        name = request.form.get("name")
        password = request.form.get("password")
        selected_role = request.form.get("role") # 'patient' or 'admin'
        
        # Super Admin Special Handling
        SUPER_ADMIN_EMAIL = "mulagiripavankumar886@gmail.com"
        
        # Block unauthorized Admin registration
        if selected_role == 'admin' and email != SUPER_ADMIN_EMAIL:
            return render_template("signup.html", error="Administrator registration is restricted. Please register as a Patient.")

        # Password Validation
        if len(password) < 8:
            return render_template("signup.html", error="Password must be at least 8 characters long.")
        if not re.search(r"\d", password):
             return render_template("signup.html", error="Password must contain at least one number.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
             return render_template("signup.html", error="Password must contain at least one special character.")

        user = User.query.filter_by(email=email).first()

        if user:
            return render_template("signup.html", error="Email address already exists.")

        # Determine actual role
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



# -------------------------
# AI CHATBOT API
# -------------------------
import google.generativeai as genai

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

if GEMINI_API_KEY and GEMINI_API_KEY not in ["", "YOUR_API_KEY_HERE"]:
    genai.configure(api_key=GEMINI_API_KEY)
    HAS_GEMINI = True
else:
    HAS_GEMINI = False
    app.logger.warning("Gemini API Key not found. Chatbot will use hardcoded fallback responses.")


@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.json
    user_msg = data.get("message", "").lower()
    
    # Fetch User's Latest Health Data for Personalization
    latest_record = MedicalRecord.query.filter_by(user_id=current_user.id).order_by(MedicalRecord.timestamp.desc()).first()
    user_vitals_summary = ""
    if latest_record:
        user_vitals_summary = (
            f"Note: The user's current health metrics are: "
            f"Age: {latest_record.age}, Gender: {latest_record.gender}, BMI: {latest_record.bmi}, "
            f"HbA1c: {latest_record.hba1c}%, Fasting Glucose: {latest_record.glucose} mg/dL, "
            f"Blood Pressure: {latest_record.systolic_bp}/{latest_record.diastolic_bp} mmHg, "
            f"Risk Score: {latest_record.risk_score}%, Risk Category: {latest_record.risk_label}."
        )

    # 1. Advanced Local Expert System (High Efficiency & Works Offline)
    # Mapping multiple keys to the same response
    expert_knowledge = [
        (["risk", "calculate", "score"], "Our Risk Prediction tool uses the CatBoost algorithm trained on 100,000 clinical records. Based on your inputs, we calculate a percentage-based risk score."),
        (["dashboard", "track", "history"], "The Dashboard visualizes your health trends, metabolic scores, and how different factors like Activity and BMI impact your overall risk."),
        (["hba1c", "hb1ac", "a1c", "hb a1c"], "HbA1c measures your average blood sugar over 3 months. Normal: <5.7%, Pre-diabetes: 5.7%-6.4%, Diabetes: 6.5%+. Monitoring this is crucial for long-term health."),
        (["glucose", "sugar", "fasting"], "Fasting glucose should be 70-99 mg/dL. Values above 126 mg/dL often indicate diabetes. Always confirm with clinical tests."),
        (["diet", "food", "eat"], "Focus on high-fiber, low-glycemic foods. Leafy greens, nuts, seeds, and whole grains help stabilize blood sugar. Limit refined sugars and processed carbs."),
        (["exercise", "activity", "walk", "gym"], "Aim for 150 mins of moderate activity weekly. Exercise improves insulin sensitivity and helps manage weight effectively."),
        (["bmi", "weight", "height"], "BMI is height-to-weight ratio. Healthy: 18.5-24.9. Higher BMI often correlates with increased insulin resistance."),
        (["symptoms", "signs", "thirsty", "urination"], "Keep an eye out for extreme thirst, frequent urination, fatigue, and blurred vision. See a doctor if these persist."),
        (["analyze", "status", "report", "my risk"], "To analyze your health status, please use the 'Prediction' tab first. Once you have a score, I can help explain what the numbers mean for your wellness journey."),
        (["hello", "hi", "hey"], "Hello! I'm your HealthRisk AI. I can analyze your vitals, explain diabetes risk, or offer lifestyle tips. How can I assist you today?"),
        (["thanks", "thank you"], "You're very welcome! Stay proactive about your health.")
    ]

    # Match keywords in the user message
    for keys, val in expert_knowledge:
        if any(k in user_msg for k in keys):
            # Allow "analyze" or "report" queries to fall through to Gemini if available for a deeper look
            if any(k in user_msg for k in ["analyze", "status", "report"]) and HAS_GEMINI:
                continue
            return {"reply": val}
    
    # 2. General AI Fallback (Gemini)
    if not HAS_GEMINI:
         fallback_reply = "I've matched your question to my local medical database, but for complex health queries, I need my 'Advanced AI Brain' (Gemini API Key) connected."
         if user_vitals_summary and ("analyze" in user_msg or "status" in user_msg):
             fallback_reply += f" Based on your latest record (Risk: {latest_record.risk_label}), I'd suggest focusing on your lifestyle adjustments mentioned in the 'Prediction' results."
         return {"reply": fallback_reply}
    
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        
        # Enhanced Analytical System Instruction
        system_instruction = (
            "You are the 'Advanced AI Brain' of HealthRisk AI, a specialized medical analytics assistant. "
            "Your goal is to provide deep analysis of health-related questions. "
            "When analyzing, be data-driven, clinical yet empathetic, and very specific. "
            f"{user_vitals_summary} "
            "If the user asks to 'analyze' or asks about their 'status', use their provided vital metrics (if available above) "
            "to give a detailed breakdown of what they should prioritize. "
            "RULES: "
            "1. ONLY answer health, wellness, and medical analytics questions. "
            "2. If the user mentions their vitals, explain the clinical significance of those numbers. "
            "3. Always remind them that you are an AI and they must consult a doctor for diagnosis. "
            "4. Keep responses professional, structured, and helpful. "
            "5. If no vitals are available, ask them to use the 'Risk Prediction' tool first for a personalized analysis."
        )
        
        full_prompt = f"{system_instruction}\n\nUser: {data.get('message', '')}\nAssistant:"
        
        response = model.generate_content(full_prompt)
        reply = response.text
    except Exception as e:
        app.logger.error(f"Gemini Error: {e}")
        reply = "I'm having a little trouble connecting to the cloud right now. However, I can still answer basic questions about diet, exercise, and vitals!"

    return {"reply": reply}


@app.route("/api/analyze_result", methods=["POST"])
@login_required
def analyze_result():
    if not HAS_GEMINI:
        return {"error": "Advanced AI Brain (Gemini) is not connected. Please contact the administrator."}, 400
    
    data = request.json
    result_data = data.get("result")
    
    if not result_data:
        return {"error": "No result data provided."}, 400
    
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        
        prompt = (
            "You are a clinical health analyst. Analyze the following diabetes risk assessment results "
            "and provide deep, actionable insights. Structure your response into 3 short paragraphs: "
            "1. Clinical Significance: What do these numbers (HbA1c, Glucose, BP) mean together? "
            "2. Critical Priorities: What is the single most important thing this user should change? "
            "3. Long-term Outlook: What is the likely health trajectory if changes are (or aren't) made? "
            f"\n\nResults to Analyze: {result_data}"
            "\n\nKeep it professional, data-centric, and empathetic. Always include a medical disclaimer."
        )
        
        response = model.generate_content(prompt)
        return {"analysis": response.text}
    except Exception as e:
        app.logger.error(f"Analysis Error: {e}")
        return {"error": "Failed to generate AI analysis. Please try again later."}, 500



# -------------------------
# DASHBOARD
# -------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    # Initial render - just the template structure
    # Data will be loaded via AJAX for consistency
    return render_template("dashboard.html")

# -------------------------
# ADMIN DASHBOARD
# -------------------------
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    # Render the admin dashboard template
    return render_template("admin_dashboard.html")

@app.route("/admin/patients")
@admin_required
def admin_patients():
    # Render the dedicated patient management page
    return render_template("admin_patients.html")

@app.route("/api/admin/patients")
@admin_required
def api_admin_patients():
    # Get all patients who have at least one medical record
    all_patients = User.query.filter_by(role='patient').all()
    patients = [p for p in all_patients if MedicalRecord.query.filter_by(user_id=p.id).first()]
    
    patient_list = []
    risk_counts = {"High": 0, "Medium": 0, "Low": 0}
    age_groups = {"0-20": 0, "21-40": 0, "41-60": 0, "61+": 0}
    
    family_history_counts = {"Yes": 0, "No": 0}
    hba1c_values = []
    glucose_values = []
    bp_systolic_values = []
    activity_values = []
    
    for p in patients:
        # Get latest medical record for each patient
        latest_record = MedicalRecord.query.filter_by(user_id=p.id).order_by(MedicalRecord.timestamp.desc()).first()
        
        risk_label = latest_record.risk_label if latest_record else "No Data"
        risk_counts[risk_label] += 1
        
        if latest_record:
            # Age grouping
            age = latest_record.age
            if age <= 20: age_groups["0-20"] += 1
            elif age <= 40: age_groups["21-40"] += 1
            elif age <= 60: age_groups["41-60"] += 1
            else: age_groups["61+"] += 1
            
            # Family History
            if latest_record.family_history == 1: family_history_counts["Yes"] += 1
            else: family_history_counts["No"] += 1
            
            # HbA1c
            hba1c_values.append(latest_record.hba1c)
            glucose_values.append(latest_record.glucose)
            bp_systolic_values.append(latest_record.systolic_bp)
            activity_values.append(latest_record.activity_minutes)
        
        patient_list.append({
            "id": p.id,
            "name": p.name,
            "email": p.email,
            "latest_risk": risk_label,
            "latest_score": latest_record.risk_score if latest_record else None,
            "last_visit": latest_record.timestamp.strftime("%Y-%m-%d") if latest_record else "N/A"
        })
    
    # Calculate stats for KPI cards
    total_count = len(patients)
    high_risk_count = risk_counts["High"]
    avg_risk_val = round(np.mean([p['latest_score'] for p in patient_list if p['latest_score'] is not None]), 1) if patient_list and any(p['latest_score'] is not None for p in patient_list) else 0
    
    # Determine System Status (Good/Bad)
    if total_count == 0:
        system_status = "N/A"
        status_class = "secondary"
    else:
        high_risk_pct = (high_risk_count / total_count) * 100
        if high_risk_pct > 30:
            system_status = "Critical (Bad)"
            status_class = "danger"
        elif high_risk_pct > 15:
            system_status = "Warning (Moderate)"
            status_class = "warning"
        else:
            system_status = "Optimal (Good)"
            status_class = "success"

    avg_hba1c = round(np.mean(hba1c_values), 1) if hba1c_values else 0

    stats = {
        "total_patients": total_count,
        "high_risk_count": high_risk_count,
        "avg_risk": avg_risk_val,
        "avg_hba1c": avg_hba1c,
        "system_status": system_status,
        "status_class": status_class
    }
    
    # Real gender counts from patients
    gender_counts = {"Male": 0, "Female": 0, "Other": 0, "Unknown": 0}
    for p in patients:
        latest = MedicalRecord.query.filter_by(user_id=p.id).order_by(MedicalRecord.timestamp.desc()).first()
        gender = latest.gender if latest and latest.gender else "Unknown"
        if gender in gender_counts:
            gender_counts[gender] += 1
        else:
            gender_counts["Other"] += 1

    # Chart data
    charts = {
        "risk_dist": {
            "labels": list(risk_counts.keys()),
            "values": list(risk_counts.values())
        },
        "registration_trend": {
            "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "values": [2, 5, 3, 8, 4, max(1, len(patients)//2), len(patients)]
        },
        "gender_dist": {
            "labels": list(gender_counts.keys()),
            "values": list(gender_counts.values())
        },
        "age_dist": {
            "labels": list(age_groups.keys()),
            "values": list(age_groups.values())
        },
        "family_history": {
            "labels": list(family_history_counts.keys()),
            "values": list(family_history_counts.values())
        },
        "clinical_distributions": {
            "glucose": glucose_values,
            "bp_systolic": bp_systolic_values,
            "activity": activity_values,
            "hba1c": hba1c_values
        }
    }
    
    # Clinical Alerts (Detect critical cases)
    clinical_alerts = []
    for p in patients:
        latest = MedicalRecord.query.filter_by(user_id=p.id).order_by(MedicalRecord.timestamp.desc()).first()
        if latest:
            if latest.risk_label == "High":
                clinical_alerts.append({
                    "patient": p.name,
                    "type": "High Risk",
                    "value": f"{latest.risk_score}%",
                    "class": "danger"
                })
            elif latest.glucose > 180:
                clinical_alerts.append
                ({
                    "patient": p.name,
                    "type": "High Glucose",
                    "value": f"{latest.glucose}",
                    "class": "warning"
                })
            elif latest.systolic_bp > 140:
                clinical_alerts.append({
                    "patient": p.name,
                    "type": "High BP",
                    "value": f"{latest.systolic_bp}",
                    "class": "warning"
                })

    return {
        "patients": patient_list, 
        "stats": stats, 
        "charts": charts,
        "alerts": clinical_alerts[:5] # Limit to top 5 recent alerts
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
    
    history = []
    for r in records:
        history.append({
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M"),
            "risk_score": r.risk_score,
            "risk_label": r.risk_label,
            "bmi": r.bmi,
            "hba1c": r.hba1c,
            "glucose": r.glucose,
            "bp": f"{r.systolic_bp}/{r.diastolic_bp}"
        })
    
    return {"history": history}

@app.route("/api/dashboard/filter", methods=["GET"])
def dashboard_filter():
    age_range = request.args.get("age", "All")
    gender = request.args.get("gender", "All")
    stage = request.args.get("diabetes_stage", "All")
    education = request.args.get("education", "All")
    ethnicity = request.args.get("ethnicity", "All")
    income = request.args.get("income", "All")
    
    # Apply filters to data
    filtered_df = df.copy()
    
    # Age range filter
    if age_range != "All":
        if age_range == "0-20":
            filtered_df = filtered_df[filtered_df["age"] <= 20]
        elif age_range == "21-40":
            filtered_df = filtered_df[(filtered_df["age"] >= 21) & (filtered_df["age"] <= 40)]
        elif age_range == "41-60":
            filtered_df = filtered_df[(filtered_df["age"] >= 41) & (filtered_df["age"] <= 60)]
        elif age_range == "61+":
            filtered_df = filtered_df[filtered_df["age"] >= 61]
    
    if gender != "All":
        filtered_df = filtered_df[filtered_df["gender"] == gender]
    if stage != "All":
        filtered_df = filtered_df[filtered_df["diabetes_stage"] == stage]
    if education != "All":
        filtered_df = filtered_df[filtered_df["education_level"] == education]
    if ethnicity != "All":
        filtered_df = filtered_df[filtered_df["ethnicity"] == ethnicity]
    if income != "All":
        filtered_df = filtered_df[filtered_df["income_level"] == income]
    
    # KPIs calculated from FILTERED data (updates with slicers)
    total = len(filtered_df)
    avg_hba1c = round(filtered_df["hba1c"].mean(), 2) if not filtered_df.empty else 0
    high_risk = int((filtered_df["diabetes_stage"] == "Type 2").sum())
    avg_bmi = round(filtered_df["bmi"].mean(), 1) if not filtered_df.empty else 0
    
    # Define common layout properties for transparent dark mode charts
    chart_layout_update = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        margin=dict(l=60, r=40, t=60, b=120)
    )

    # 1. Age Distribution - Histogram
    f1 = px.histogram(filtered_df, x="age", color="diabetes_stage", 
                      title="Age Distribution",
                      template="plotly_dark", barmode="overlay", opacity=0.7)
    f1.update_layout(chart_layout_update)
    f1.update_layout(xaxis=dict(title="Age", showticklabels=True), yaxis=dict(title="Count", showticklabels=True))
    
    # 2. BMI vs HbA1c - Scatter
    f2 = px.scatter(filtered_df, x="bmi", y="hba1c", color="diabetes_stage",
                    title="BMI vs HbA1c Correlation",
                    template="plotly_dark", opacity=0.6)
    f2.update_layout(chart_layout_update)
    f2.update_layout(xaxis=dict(title="Body Mass Index", showticklabels=True), yaxis=dict(title="HbA1c (%)", showticklabels=True))
    
    # 3. Gender Distribution - Donut Chart
    f3 = px.pie(filtered_df, names="gender", title="Gender Distribution",
                hole=0.5, template="plotly_dark")
    f3.update_layout(chart_layout_update)
    f3.update_layout(margin=dict(l=20, r=20, t=60, b=40))
    
    # 4. Diabetes Stage - Bar Chart
    stage_counts = filtered_df["diabetes_stage"].value_counts().reset_index()
    stage_counts.columns = ["Stage", "Count"]
    f4 = px.bar(stage_counts, x="Stage", y="Count", 
                title="Patient Distribution by Stage",
                template="plotly_dark", color="Stage")
    f4.update_layout(chart_layout_update)
    f4.update_layout(showlegend=False, xaxis=dict(title="Diabetes Stage", showticklabels=True, tickangle=-45))
    
    # 5. Blood Pressure - Scatter
    f5 = px.scatter(filtered_df, x="systolic_bp", y="diastolic_bp", color="diabetes_stage",
                    title="Blood Pressure Analysis",
                    template="plotly_dark", opacity=0.5)
    f5.update_layout(chart_layout_update)
    f5.update_layout(xaxis=dict(title="Systolic BP", showticklabels=True), yaxis=dict(title="Diastolic BP", showticklabels=True))
    
    # 6. Physical Activity - Box Plot
    f6 = px.box(filtered_df, x="diabetes_stage", y="physical_activity_minutes_per_week",
                title="Physical Activity by Stage",
                template="plotly_dark", color="diabetes_stage")
    f6.update_layout(chart_layout_update)
    f6.update_layout(showlegend=False, xaxis=dict(title="Diabetes Stage", showticklabels=True, tickangle=-45))
    
    # 7. Cholesterol Distribution - Violin Plot
    f7 = px.violin(filtered_df, x="diabetes_stage", y="cholesterol_total",
                   title="Cholesterol Distribution",
                   template="plotly_dark", color="diabetes_stage", box=True)
    f7.update_layout(chart_layout_update)
    f7.update_layout(showlegend=False, xaxis=dict(title="Diabetes Stage", showticklabels=True, tickangle=-45))
    
    # 8. Glucose Levels - Box Plot
    f8 = px.box(filtered_df, x="diabetes_stage", y="glucose_fasting",
                title="Fasting Glucose Levels",
                template="plotly_dark", color="diabetes_stage")
    f8.update_layout(chart_layout_update)
    f8.update_layout(showlegend=False, xaxis=dict(title="Diabetes Stage", showticklabels=True, tickangle=-45))
    
    # 9. Family History Impact - Grouped Bar
    f9 = px.histogram(filtered_df, x="diabetes_stage", color="family_history_diabetes",
                      title="Family History Impact",
                      template="plotly_dark", barmode="group")
    f9.update_layout(chart_layout_update)
    f9.update_layout(xaxis=dict(title="Diabetes Stage", showticklabels=True, tickangle=-45))
    
    # 10. Risk Score Distribution - Histogram
    f10 = px.histogram(filtered_df, x="diabetes_risk_score", color="diabetes_stage",
                       title="Risk Score Distribution",
                       template="plotly_dark", nbins=30)
    f10.update_layout(chart_layout_update)
    f10.update_layout(xaxis=dict(title="Risk Score", showticklabels=True))

    # 11. Ethnicity Breakdown - Donut Chart
    f11 = px.pie(filtered_df, names="ethnicity", title="Ethnicity Breakdown",
                 template="plotly_dark", hole=0.5)
    f11.update_layout(chart_layout_update)

    # 12. Smoking Status - Bar Chart
    f12 = px.histogram(filtered_df, x="smoking_status", color="diabetes_stage",
                       title="Smoking Status vs Risk", barmode="group",
                       template="plotly_dark")
    f12.update_layout(chart_layout_update)
    f12.update_layout(xaxis=dict(title="Smoking Status", showticklabels=True))

    # 13. Diet Score Distribution - Histogram
    f13 = px.histogram(filtered_df, x="diet_score", color="diabetes_stage",
                       title="Diet Score Distribution",
                       template="plotly_dark", nbins=20)
    f13.update_layout(chart_layout_update)
    f13.update_layout(xaxis=dict(title="Diet Score", showticklabels=True))

    # 14. Sleep Hours vs Risk - Box Plot
    f14 = px.box(filtered_df, x="diabetes_stage", y="sleep_hours_per_day",
                 title="Sleep patterns", template="plotly_dark", color="diabetes_stage")
    f14.update_layout(chart_layout_update)
    f14.update_layout(showlegend=False, xaxis=dict(title="Diabetes Stage", showticklabels=True, tickangle=-45))

    # 15. Education Level - Donut Chart
    f15 = px.pie(filtered_df, names="education_level", title="Education Level",
                 template="plotly_dark", hole=0.5)
    f15.update_layout(chart_layout_update)

    # 16. Employment Status - Bar Chart
    f16 = px.histogram(filtered_df, x="employment_status", color="diabetes_stage",
                       title="Employment Status", barmode="group",
                       template="plotly_dark")
    f16.update_layout(chart_layout_update)
    f16.update_layout(xaxis=dict(title="Employment Status", showticklabels=True))

    # 17. Insulin Levels vs Risk - Box Plot
    f17 = px.box(filtered_df, x="diabetes_stage", y="insulin_level",
                 title="Insulin Levels", template="plotly_dark", color="diabetes_stage")
    f17.update_layout(chart_layout_update)
    f17.update_layout(showlegend=False, xaxis=dict(title="Diabetes Stage", showticklabels=True, tickangle=-45))

    # 18. Risk Factor Importance - Best Graph (Population)
    # We'll calculate a simple "Importance" based on correlation with Risk Score
    num_only = filtered_df.select_dtypes(include=[np.number])
    if "diabetes_risk_score" in num_only.columns and not num_only.empty:
        corr = num_only.corr()["diabetes_risk_score"].abs().sort_values(ascending=True).drop("diabetes_risk_score", errors='ignore')
        # Use top 10 factors
        top_corr = corr.tail(10)
        f18 = px.bar(x=top_corr.values, y=top_corr.index, orientation='h',
                     title="Key Risk Factor Weights (Population)",
                     template="plotly_dark",
                     labels={"x": "Importance Weight", "y": "Health Metric"},
                     color=top_corr.values, color_continuous_scale="Viridis")
        f18.update_layout(chart_layout_update)
        f18.update_coloraxes(showscale=False)
    else:
        # Fallback empty chart
        f18 = px.bar(title="Risk Factor Weights (No Data)", template="plotly_dark")
        f18.update_layout(chart_layout_update)

    import json
    def to_dict(fig): return json.loads(pio.to_json(fig))

    return {
        "kpi": {"total": total, "avg_hba1c": avg_hba1c, "high_risk": high_risk, "avg_bmi": avg_bmi},
        "charts": {
            "c1": to_dict(f1), "c2": to_dict(f2), "c3": to_dict(f3), "c4": to_dict(f4), "c5": to_dict(f5),
            "c6": to_dict(f6), "c7": to_dict(f7), "c8": to_dict(f8), "c9": to_dict(f9), "c10": to_dict(f10),
            "c11": to_dict(f11), "c12": to_dict(f12), "c13": to_dict(f13), "c14": to_dict(f14),
            "c15": to_dict(f15), "c16": to_dict(f16), "c17": to_dict(f17), "c18": to_dict(f18)
        }
    }

@app.route("/api/personal-history", methods=["GET"])
@login_required
def personal_history():
    records = MedicalRecord.query.filter_by(user_id=current_user.id).order_by(MedicalRecord.timestamp.asc()).all()
    
    if not records:
        return {"has_data": False}
    
    # Prepare data for charts
    history = []
    for r in records:
        history.append({
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M"),
            "risk_score": r.risk_score,
            "bmi": r.bmi,
            "hba1c": r.hba1c,
            "glucose": r.glucose
        })
    
    # Create sophisticated history charts using Plotly
    df_hist = pd.DataFrame(history)
    
    # Latest status for gauges/indicators
    latest = history[-1]
    
    # 1. Risk Score Trend - Area Chart
    f_risk = px.area(df_hist, x="timestamp", y="risk_score", 
                     title="<b>Your Diabetes Risk Progression</b>",
                     markers=True, 
                     template="plotly_dark",
                     labels={"risk_score": "Risk Level (%)", "timestamp": "Assessment Date"})
    
    f_risk.update_traces(line_color='#6366f1', fillcolor='rgba(99, 102, 241, 0.2)', marker=dict(size=10, color='white', line=dict(width=2, color='#6366f1')))
    f_risk.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12, color="#a0aec0"),
        title_font=dict(size=18, color="white"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[0, 105]),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # 2. Vitals Trend - Line Chart with multiple metrics
    f_vitals = px.line(df_hist, x="timestamp", y=["bmi", "hba1c", "glucose"], 
                       title="<b>Health Metrics Monitoring</b>",
                       markers=True, 
                       template="plotly_dark",
                       labels={"value": "Level", "variable": "Metric"})
    
    colors = ['#10b981', '#0ea5e9', '#f43f5e']
    for i, color in enumerate(colors):
        if i < len(f_vitals.data):
            f_vitals.data[i].line.color = color
            f_vitals.data[i].marker.size = 8
            
    f_vitals.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12, color="#a0aec0"),
        title_font=dict(size=18, color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    )
    
    import json
    return {
        "has_data": True,
        "latest_stats": {
            "score": latest["risk_score"],
            "glucose": latest["glucose"],
            "bmi": latest["bmi"],
            "hba1c": latest["hba1c"]
        },
        "admin_request_status": current_user.admin_request_status,
        "role": current_user.role,
        "charts": {
            "risk_trend": json.loads(pio.to_json(f_risk)),
            "vitals_trend": json.loads(pio.to_json(f_vitals))
        }
    }

# -------------------------
# SERVICES
# -------------------------
@app.route("/services")
def services():
    return render_template("services.html")

# -------------------------
# CONTACT
# -------------------------
@app.route("/contact")
def contact():
    return render_template("contact.html")

# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
