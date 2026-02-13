import google.generativeai as genai
import os

def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key and api_key not in ["", "YOUR_API_KEY_HERE"]:
        genai.configure(api_key=api_key)
        return True
    return False

def get_expert_response(user_msg):
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
    for keys, val in expert_knowledge:
        if any(k in user_msg.lower() for k in keys):
            return val
    return None

def generate_ai_response(prompt, system_instruction=""):
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        full_prompt = f"{system_instruction}\n\n{prompt}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return None
