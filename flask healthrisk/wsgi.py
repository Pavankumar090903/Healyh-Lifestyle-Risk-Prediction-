import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "flask healthrisk", "app"))
from app import app as application
