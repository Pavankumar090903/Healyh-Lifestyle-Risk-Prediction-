<p align="center">
  <h1 align="center">🩺 Health Risk AI</h1>
  <p align="center">
    <strong>AI-Powered Diabetes Risk Prediction & Wellness Analytics Platform</strong>
  
</p>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [How It Works](#-how-it-works)
- [Screenshots](#-screenshots)
- [API Endpoints](#-api-endpoints)
- [Deployment](#-deployment)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 About the Project

**Health Risk AI** is a full-stack web application that uses **Machine Learning** to predict diabetes risk based on patient health metrics. It combines a trained **CatBoost classifier** with **Google's Gemini AI** chatbot to provide personalized health insights, risk assessments, and actionable wellness recommendations.

### 🌟 What Makes It Special?

- **Not just a prediction tool** — it provides detailed factor-by-factor health analysis
- **Interactive dashboard** with real-time Plotly charts and analytics
- **AI-powered chatbot** for personalized health conversations
- **Role-based access** — separate views for Patients and Administrators
- **Cloud-deployed** and accessible from anywhere

> 🔗 **Live Demo:** [health-project-health-risk-ai.hf.space](https://health-project-health-risk-ai.hf.space/)

---

## ✨ Key Features

### 🔐 Authentication & Security
- Secure login/signup with **Werkzeug password hashing** (PBKDF2 + SHA-256)
- Role-based access control (Patient / Admin)
- Session management with Flask-Login
- CSRF protection and input sanitization

### 🧠 ML Risk Prediction
- **10 health parameters** analyzed: Age, BMI, Blood Pressure, Cholesterol (HDL/LDL), Blood Glucose, Smoking, Alcohol, Physical Activity, Family History
- **CatBoost Gradient Boosting** classifier with probability scoring
- **Risk categorization**: Low (0–30%), Moderate (30–60%), High (60–100%)
- **Per-factor impact analysis** with clinical threshold comparison

### 📊 Interactive Dashboard
- **6 Plotly.js charts**: Risk gauge, BMI distribution, health parameters spider chart, glucose trends, blood pressure analysis, cholesterol breakdown
- Live-updating date/time display
- Prediction history tracking
- KPI summary cards

### 🤖 AI Health Chatbot
- Powered by **Google Gemini 2.0 Flash**
- Context-aware conversations using patient's health data
- Voice interaction with **Web Speech API**
- Draggable floating chat widget

### 👨‍💼 Admin Panel
- Patient management dashboard
- View all registered patients and their health records
- Approve/reject admin access requests
- System health monitoring

### 📱 Responsive Design
- Mobile-first Bootstrap 5 layout
- Glassmorphism UI with 3D tilt cards
- Smooth CSS animations and micro-interactions
- Dark gradient themes

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | Flask 3.0 | Web framework, routing, API |
| **ML Model** | CatBoost 1.2.5 | Gradient boosting classifier |
| **AI Chatbot** | Google Gemini 2.0 Flash | Natural language health assistant |
| **Database** | SQLite + SQLAlchemy | User & prediction data storage |
| **Frontend** | Bootstrap 5, HTML5, CSS3 | Responsive UI components |
| **Charts** | Plotly.js | Interactive data visualizations |
| **Auth** | Flask-Login + Werkzeug | Session management, password hashing |
| **Server** | Gunicorn | Production WSGI server |
| **Container** | Docker | Deployment containerization |
| **Hosting** | Hugging Face Spaces | Cloud deployment platform |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                      │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌───────────────┐  │
│  │ Login/  │ │ Predict  │ │ Dashboard │ │ Admin Panel   │  │
│  │ Signup  │ │   Form   │ │  Charts   │ │ (Patients)    │  │
│  └────┬────┘ └────┬─────┘ └─────┬─────┘ └──────┬────────┘  │
│       │           │             │               │            │
│  Bootstrap 5 + Plotly.js + Vanilla JS + Web Speech API      │
└───────┼───────────┼─────────────┼───────────────┼────────────┘
        │           │             │               │
   HTTP │     POST  │       AJAX  │         AJAX  │
        ▼           ▼             ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                    SERVER (Flask + Gunicorn)                  │
│                                                              │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐  │
│  │ Flask-Login  │  │  CatBoost ML  │  │  Gemini AI API   │  │
│  │ Auth System  │  │  Prediction   │  │  Chatbot Engine  │  │
│  │              │  │  Engine       │  │                  │  │
│  │ • Login      │  │ • predict_    │  │ • /api/chat      │  │
│  │ • Signup     │  │   proba()     │  │ • Context-aware  │  │
│  │ • Sessions   │  │ • Risk Score  │  │ • Health advice  │  │
│  └──────┬───────┘  │ • Impact Calc │  └────────┬─────────┘  │
│         │          └───────┬───────┘           │            │
│         ▼                  ▼                   │            │
│  ┌─────────────────────────────────────┐      │            │
│  │         SQLite Database             │      │            │
│  │  ┌─────────┐  ┌──────────────────┐  │      │            │
│  │  │  Users  │  │ Medical Records  │  │      │            │
│  │  │ Table   │  │     Table        │  │      │            │
│  │  └─────────┘  └──────────────────┘  │      │            │
│  └─────────────────────────────────────┘      │            │
└───────────────────────────────────────────────┼────────────┘
                                                │
                                    ┌───────────▼───────────┐
                                    │  Google Gemini API    │
                                    │  (External Service)   │
                                    └───────────────────────┘
```

---

## 📁 Project Structure

```
health-risk-ai/
│
├── Dockerfile                    # Docker build instructions (Python 3.9 + Gunicorn)
├── README.md                     # This file
├── .gitignore                    # Files excluded from Git
│
├── app/                          # Main application code
│   ├── app.py                    # Flask backend (956 lines) — routes, ML, chatbot, auth
│   ├── requirements.txt          # Python dependencies (13 packages)
│   │
│   ├── diabetes_risk_model.pkl   # Trained CatBoost classifier model
│   ├── encoders.pkl              # LabelEncoders for categorical features
│   ├── num_cols.pkl              # Numerical column names
│   ├── cat_cols.pkl              # Categorical column names
│   ├── diabetes_dataset.csv      # Training dataset (100K+ records)
│   │
│   ├── templates/                # Jinja2 HTML templates (12 files)
│   │   ├── base.html             # Base layout — navbar, footer, chatbot widget
│   │   ├── login.html            # Login page with animations
│   │   ├── signup.html           # Registration with role selection
│   │   ├── home.html             # Landing page — hero, stats, features
│   │   ├── predict.html          # Prediction form + 5 result charts + alerts
│   │   ├── dashboard.html        # Patient analytics — 6 Plotly charts
│   │   ├── about.html            # About page — architecture & team
│   │   ├── services.html         # Services overview
│   │   ├── contact.html          # Contact form
│   │   ├── admin_dashboard.html  # Admin overview + system stats
│   │   ├── admin_patients.html   # Patient list management
│   │   └── admin_patient_detail.html  # Individual patient profile
│   │
│   └── static/                   # Static assets
       └── images/               # Background images
           ├── login_bg.png
           └── welcome_bg.png
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+** installed
- **Google Gemini API key** (free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/health-risk-ai.git
   cd health-risk-ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   cd app
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file inside the `app/` folder:
   ```env
   SECRET_KEY=your_random_secret_key_here
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open in browser**
   ```
   http://localhost:5000
   ```

---

## ⚙️ How It Works

### 1. User Input
Patient enters 10 health metrics through the prediction form:

| # | Parameter | Type | Example |
|---|-----------|------|---------|
| 1 | Age | Numeric | 45 |
| 2 | BMI | Numeric | 28.5 |
| 3 | Blood Pressure (Systolic) | Numeric | 130 |
| 4 | Cholesterol — HDL | Numeric | 55 |
| 5 | Cholesterol — LDL | Numeric | 140 |
| 6 | Blood Glucose | Numeric | 110 |
| 7 | Smoking Status | Categorical | Non-smoker |
| 8 | Alcohol Consumption | Categorical | Moderate |
| 9 | Physical Activity | Categorical | Active |
| 10 | Family History | Categorical | Yes |

### 2. ML Prediction
```python
# CatBoost model predicts probability of diabetes risk
risk_probability = model.predict_proba(input_data)[0][1]  # 0.0 to 1.0
risk_percentage = risk_probability * 100                    # 0% to 100%

# Risk categorization
if risk_percentage < 30:    → "Low Risk"      (Green)
elif risk_percentage < 60:  → "Moderate Risk"  (Yellow)
else:                       → "High Risk"      (Red)
```

### 3. Factor Impact Analysis
Each health metric is compared against clinical thresholds to calculate its individual contribution:

```python
# Example: BMI Impact Calculation
clinical_threshold = 25.0  # Normal BMI upper limit
patient_bmi = 32.0
difference = 32.0 - 25.0 = +7.0
impact_score = min((7.0 / 25.0) * 100, 100) = 28 points  → "Above Optimal"
```

### 4. Results Display
- **Risk Gauge Chart** — Plotly semicircular gauge (0–100%)
- **Factor Analysis Table** — Each metric with impact score
- **Health Alerts** — Actionable recommendations
- **Summary Card** — Overall risk assessment

---

## 🔌 API Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `GET` | `/` | Home page | ❌ |
| `GET/POST` | `/login` | User login | ❌ |
| `GET/POST` | `/signup` | User registration | ❌ |
| `GET` | `/home` | Landing page | ✅ |
| `GET/POST` | `/predict` | Risk prediction form + results | ✅ |
| `GET` | `/dashboard` | Patient analytics dashboard | ✅ |
| `GET` | `/about` | About page | ✅ |
| `GET` | `/services` | Services page | ✅ |
| `GET` | `/contact` | Contact page | ✅ |
| `POST` | `/api/chat` | AI chatbot (Gemini) | ✅ |
| `GET` | `/api/personal_history` | Prediction history JSON | ✅ |
| `GET` | `/admin/dashboard` | Admin overview | ✅ 👨‍💼 |
| `GET` | `/admin/patients` | Patient list | ✅ 👨‍💼 |
| `GET` | `/admin/patient/<id>` | Patient detail | ✅ 👨‍💼 |
| `POST` | `/admin/approve_admin/<id>` | Approve admin request | ✅ 👨‍💼 |
| `POST` | `/admin/reject_admin/<id>` | Reject admin request | ✅ 👨‍💼 |
| `GET` | `/logout` | User logout | ✅ |

---

## ☁️ Deployment

### Deployed on Hugging Face Spaces

The application is deployed using **Docker** on **Hugging Face Spaces**.

**Live URL:** [health-project-health-risk-ai.hf.space](https://health-project-health-risk-ai.hf.space/)

#### Deployment Files:
| File | Purpose |
|------|---------|
| `Dockerfile` | Python 3.9 container, installs dependencies, runs Gunicorn on port 7860 |
| `README.md` | HF Spaces metadata (`sdk: docker`) |
| `.gitignore` | Excludes `.env`, `*.db`, `__pycache__/` |

#### How to Deploy Your Own:

1. Create a Space on [huggingface.co/new-space](https://huggingface.co/new-space) → Select **Docker** SDK
2. Upload files using `huggingface_hub`:
   ```python
   from huggingface_hub import HfApi
   api = HfApi()
   api.upload_folder(
       folder_path="./health-risk-ai",
       repo_id="YOUR_USERNAME/health-risk-ai",
       repo_type="space"
   )
   ```
3. Set secrets in Space Settings:
   - `SECRET_KEY` — any random string
   - `GEMINI_API_KEY` — your Google AI key

---

## 🗄️ Database Schema

```sql
-- Users Table
CREATE TABLE user (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        VARCHAR(100) NOT NULL,
    email       VARCHAR(120) UNIQUE NOT NULL,
    password    VARCHAR(200) NOT NULL,      -- PBKDF2+SHA256 hashed
    role        VARCHAR(20) DEFAULT 'patient',
    admin_approved BOOLEAN DEFAULT FALSE,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Medical Records Table
CREATE TABLE medical_record (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER NOT NULL,   -- FK → user.id
    age                 FLOAT,
    bmi                 FLOAT,
    blood_pressure      FLOAT,
    cholesterol_hdl     FLOAT,
    cholesterol_ldl     FLOAT,
    blood_glucose       FLOAT,
    smoking_status      VARCHAR(50),
    alcohol_consumption VARCHAR(50),
    physical_activity   VARCHAR(50),
    family_history      VARCHAR(50),
    risk_score          FLOAT,
    risk_level          VARCHAR(20),
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user(id)
);
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** this repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m "Add amazing feature"`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.


