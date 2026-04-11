from datetime import datetime, timedelta
from gradio_client import Client
from PIL import Image
from io import BytesIO
import os
import numpy as np
import base64
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import requests

db = SQLAlchemy()

# ================= DATABASE ================= #

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    password = db.Column(db.String(100))
    centre_name = db.Column(db.String(200))
    address = db.Column(db.String(300))
    contact_no = db.Column(db.String(20))
    radiologist_name = db.Column(db.String(200))
    license_no = db.Column(db.String(100))
    logo = db.Column(db.String(200))
    signature = db.Column(db.String(200))
    role = db.Column(db.String(20), default="doctor")
    email = db.Column(db.String(120))
    website = db.Column(db.String(120))
    subscription_start = db.Column(db.DateTime, default=datetime.utcnow)
    subscription_end = db.Column(db.DateTime)
    subscription_plan = db.Column(db.String(50))  # 30d, 3m, 6m, 12m


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    result = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    patient_name = db.Column(db.String(100))
    sex = db.Column(db.String(10))
    age = db.Column(db.String(10))
    mobile = db.Column(db.String(20))
    ref_by = db.Column(db.String(100))
    image = db.Column(db.String(200))
    segmented_image = db.Column(db.String(200))
    date = db.Column(db.String(50))
    patient_id = db.Column(db.String(50))
    tumor_area = db.Column(db.Float)

# ================= CLEAN FUNCTION ================= #
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_for_json(v) for v in obj)
    elif isinstance(obj, Image.Image):
        return None
    else:
        return obj

#============Api Calling==========#

def call_hf_api(image_path):
    try:
        print("CALLING HF API")

        client = Client("sdivyanshu1508/brain-tumor-api")

        result = client.predict(
            image=Image.open(image_path),
            api_name="/predict"
        )

        print("HF RAW RESULT:", result)
        print("TYPE:", type(result))

        # ✅ Normalize structure
        if isinstance(result, tuple):
            result = result[1]

        if isinstance(result, list):
            result = result[0]

        # ✅ FIX: handle image safely
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, Image.Image):
                    buffered = BytesIO()
                    value.save(buffered, format="PNG")
                    result[key] = base64.b64encode(buffered.getvalue()).decode()

        result = clean_for_json(result)

        print("FINAL RESPONSE:", result)

        return result

    except Exception as e:
        print("HF API ERROR:", str(e))
        raise e