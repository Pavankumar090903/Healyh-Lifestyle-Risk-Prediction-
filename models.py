from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

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
