from functools import wraps
from flask import redirect, url_for, flash
from flask_login import current_user

SUPER_ADMIN_EMAIL = "mulagiripavankumar886@gmail.com"


def admin_required(f):
    """Only allow the super admin through. Everyone else gets redirected."""
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
    """Restrict a route to patients only."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
        if current_user.role != 'patient':
            flash('This feature is for patients only.', 'warning')
            return redirect(url_for('admin_dashboard'))
        return f(*args, **kwargs)
    return decorated_function
