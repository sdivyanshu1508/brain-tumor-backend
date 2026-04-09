from flask import Flask, request, jsonify, session, send_from_directory
from flask_session import Session
from flask_cors import CORS
from datetime import datetime
from functools import wraps
import os
import uuid
import random
import tensorflow as tf
from models import db, User, Prediction, segment_tumor, detect_tumor, load_models
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
from datetime import datetime, timedelta
import numpy as np
import razorpay

RAZORPAY_KEY_ID = os.environ.get("rzp_test_SYwfzTGxdSXCdN")
RAZORPAY_KEY_SECRET = os.environ.get("BEPVfY6MPcNpLWoCl0uu6tp1")

client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# ================= APP INIT ================= #
app = Flask(__name__)
app.secret_key = "secretkey"
CORS(app)
# ================= CONFIG ================= #

CORS(app, supports_credentials=True,
     resources={r"/*": {"origins": "http://localhost:3000"}})

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
app.config["SESSION_FILE_DIR"] = "./flask_session"
app.config["SESSION_COOKIE_NAME"] = "neuroscan_session"
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False

Session(app)
db.init_app(app)
# ✅ CREATE TABLES (ADD HERE TEMPORARILY)
with app.app_context():
    db.create_all()

# ================= FILE STORAGE ================= #

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =================Plans ================= #
PLANS = {
    "30days": 30,
    "3months": 90,
    "6months": 180,
    "12months": 365
}
# ================= AUTH DECORATOR ================= #

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session or session.get("role") != "admin":
            return jsonify({"error": "Admin only"}), 403
        return f(*args, **kwargs)
    return wrapper

# ================= PREPROCESS ================= #

def preprocess_cnn(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ================= Subscription end ================= #
def check_subscription():
    user = User.query.get(session['user_id'])

    if not user.subscription_end:
        return False

    if user.subscription_end < datetime.utcnow():
        return False

    return True

def subscription_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not check_subscription():
            return jsonify({"error": "Subscription expired"}), 403
        return f(*args, **kwargs)
    return wrapper

# ================= ADMIN ================= #

@app.route("/admin/users")
@admin_required
def get_users():
    users = User.query.all()
    return jsonify([{
        "id": u.id,
        "username": u.username,
        "centre_name": u.centre_name,
        "role": u.role
        } for u in users])


@app.route("/admin/predictions")
@admin_required
def get_all_predictions():
    data = Prediction.query.all()
    return jsonify([{
        "id": d.id,
        "user_id": d.user_id,
        "patient_name": d.patient_name,
        "result": d.result,
        "confidence": d.confidence
    } for d in data])

@app.route("/admin/user_predictions/<int:user_id>")
@admin_required
def get_user_predictions(user_id):
    data = Prediction.query.filter_by(user_id=user_id).all()

    return jsonify([{
        "id": d.id,
        "patient_name": d.patient_name,
        "result": d.result,
        "confidence": d.confidence,
        "image": d.image,
        "segmented_image": d.segmented_image,
        "tumor_area": d.tumor_area
    } for d in data])


@app.route("/admin/delete_user/<int:id>", methods=["DELETE"])
@admin_required
def delete_user(id):
    user = User.query.get(id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "User deleted"})

@app.route("/admin/delete_prediction/<int:id>", methods=["DELETE"])
@admin_required
def delete_prediction(id):
    prediction = Prediction.query.get(id)

    if not prediction:
        return jsonify({"error": "Prediction not found"}), 404

    # 🧹 Delete image files (optional but recommended)
    try:
        if prediction.image:
            os.remove(os.path.join(app.config["UPLOAD_FOLDER"], prediction.image))
        if prediction.segmented_image:
            os.remove(os.path.join(app.config["UPLOAD_FOLDER"], prediction.segmented_image))
    except:
        pass  # ignore if file not found

    db.session.delete(prediction)
    db.session.commit()

    return jsonify({"message": "Prediction deleted successfully"})

# ================= AUTH ================= #

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    print("REGISTER DATA:", data) 
    plan = data.get("plan")
    days = PLANS.get(plan, 0)
    now = datetime.utcnow()

    required_fields = [
        "username",
        "password",
        "centreName",
        "address",
        "contactNo",
        "radiologistName",
        "licenseNo",
        "email",
    ]

    # ✅ Check missing fields
    for field in required_fields:
        if not data.get(field):
            return jsonify({"error": f"{field} is required"}), 400

    # Check existing user
    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"error": "User exists"}), 400

    # 🔐 Get payment info from frontend
    price = data.get("price", 1000)
    payment_id = data.get("payment_id")

    # ✅ Payment validation
    if price > 0 and not payment_id:
          return jsonify({"error": "Payment required"}), 403

    # ✅ Create user
    user = User(
        username=data["username"],
        password=generate_password_hash(data["password"]),
        centre_name=data["centreName"],
        address=data["address"],
        contact_no=data["contactNo"],
        radiologist_name=data["radiologistName"],
        license_no=data["licenseNo"],
        email=data["email"],
        website=data["website"],
        role=data.get("role", "doctor")
    )
    
    if plan and days > 0:
     user.subscription_plan = plan
     user.subscription_start = now
     user.subscription_end = now + timedelta(days=days)

    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "Registered successfully"})

@app.route('/subscribe', methods=['POST'])
def subscribe():
    data = request.json
    plan = data.get("plan")
    coupon = data.get("coupon")

    user = User.query.get(session['user_id'])

    days = PLANS.get(plan)

    if not days:
        return jsonify({"error": "Invalid plan"}), 400

    # 🏷️ Apply coupon
    discount = 0
    if coupon == "SAVE50":
        discount = 0.5

    # 🔁 If already subscribed → extend
    now = datetime.utcnow()
    if user.subscription_end and user.subscription_end > now:
        user.subscription_end += timedelta(days=days)
    else:
        user.subscription_start = now
        user.subscription_end = now + timedelta(days=days)

    user.subscription_plan = plan

    db.session.commit()

    return jsonify({"message": "Subscription updated successfully"})

@app.route('/subscription-status')
def subscription_status():
    user = User.query.get(session['user_id'])

    if not user.subscription_end:
        return jsonify({
            "plan": None,
            "remaining_days": 0
        })

    remaining = (user.subscription_end - datetime.utcnow()).days

    return jsonify({
        "plan": user.subscription_plan,
        "remaining_days": max(remaining, 0)
    })

@app.route("/apply-coupon", methods=["POST"])
def apply_coupon():
    data = request.json
    code = data.get("coupon")

    coupons = {
        "meriwalialaghai": 0,
        "13bhikatega": 1,
        "DISC50": 500,
        "DISC20": 800
    }

    if code in coupons:
        return jsonify({"price": coupons[code]})
    else:
        return jsonify({"error": "Invalid coupon"}), 400

@app.route("/create-order", methods=["POST"])
def create_order():
    data = request.json
    amount = int(data.get("amount")) * 100  # paise

    order = client.order.create({
        "amount": amount,
        "currency": "INR",
        "payment_capture": 1
    })

    return jsonify({
        "order_id": order["id"],
        "amount": amount,
        "key": RAZORPAY_KEY_ID
    })

@app.route("/verify-payment", methods=["POST"])
def verify_payment():
    data = request.json

    try:
        client.utility.verify_payment_signature({
            "razorpay_order_id": data["razorpay_order_id"],
            "razorpay_payment_id": data["razorpay_payment_id"],
            "razorpay_signature": data["razorpay_signature"]
        })

        return jsonify({"status": "success"})
    except:
        return jsonify({"status": "failed"}), 400

@app.route("/login", methods=["POST"])
def login():
    session.clear()

    data = request.json
    user = User.query.filter_by(username=data["username"]).first()

    if user and check_password_hash(user.password, data["password"]):
        session["user_id"] = user.id
        session["role"] = user.role

        return jsonify({"message": "Login success", "role": user.role})

    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})

# ================= ACCOUNT ================= #

@app.route("/account", methods=["GET"])
def account():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user = User.query.get(session["user_id"])

    return jsonify({
        "centreName": user.centre_name,
        "address": user.address,
        "contactNo": user.contact_no,
        "radiologistName": user.radiologist_name,
        "email": user.email,
        "website": user.website,
        "licenseNo": user.license_no
    })


@app.route("/update-account", methods=["POST"])
def update_account():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user = User.query.get(session["user_id"])

    # Update fields
    user.centre_name = request.form.get("centreName")
    user.address = request.form.get("address")
    user.contact_no = request.form.get("contactNo")
    user.radiologist_name = request.form.get("radiologistName")
    user.email = request.form.get("email")
    user.website = request.form.get("website")
    user.license_no = request.form.get("licenseNo")

    # Password update
    if request.form.get("password"):
        user.password = generate_password_hash(request.form.get("password"))

    # File uploads
    for field in ["logo", "signature"]:
        if field in request.files:
            file = request.files[field]
            filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            setattr(user, field, filename)

    db.session.commit()
    return jsonify({"message": "Updated"})

# ================= PREDICT ================= #

@app.route("/predict", methods=["POST"])
@subscription_required
def predict_route():
    try:
        if "user_id" not in session:
            return jsonify({"error": "Unauthorized"}), 401

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        # ===== CLASSIFICATION =====
        img_cnn = preprocess_cnn(path)
        has_tumor, result, confidence = detect_tumor(img_cnn)

        segmented_filename = None
        tumor_area = 0

        # ===== SEGMENTATION =====
        if has_tumor:
            segmented_path, tumor_area = segment_tumor(path)
            segmented_filename = os.path.basename(segmented_path)

        # ===== SAVE =====
        new_prediction = Prediction(
            user_id=session["user_id"],
            patient_name=request.form.get("patient_name"),
            result=result,
            confidence=float(confidence),
            image=filename,
            segmented_image=segmented_filename,
            tumor_area=int(tumor_area),
            date=str(datetime.now())
        )

        db.session.add(new_prediction)
        db.session.commit()

        return jsonify({
            "id": new_prediction.id,
            "result": result,
            "confidence": confidence,
            "image": filename,
            "segmented_image": segmented_filename,
            "tumor_area": tumor_area,
            "tumor": has_tumor
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500
    
#=================Save======================#
@app.route("/save", methods=["POST"])
@subscription_required
def save():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()

    print("SAVE DATA:", data)

    try:
        prediction = Prediction.query.get(data.get("id"))

        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404

        # ✅ UPDATE instead of new entry
        prediction.patient_name = data.get("patient_name")
        prediction.sex = data.get("sex")
        prediction.age = data.get("age")
        prediction.mobile = data.get("mobile")
        prediction.ref_by = data.get("ref_by")
        prediction.patient_id = f"NS-{random.randint(1000,9999)}"
        prediction.date = datetime.now().strftime("%d %B %Y")

        db.session.commit()

        return jsonify({"message": "Saved successfully"})

    except Exception as e:
        print("SAVE ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

# ================= HISTORY ================= #

@app.route("/history")
def history():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = Prediction.query.filter_by(user_id=session["user_id"]).all()

    return jsonify([{
        "id": d.id,
        "patient_name": d.patient_name,
        "result": d.result,
        "confidence": d.confidence,
        "image": d.image,
        "segmented_image": d.segmented_image,
        "tumor_area": d.tumor_area
    } for d in data])

# ================= FILE SERVE ================= #

@app.route("/uploads/<filename>")
@subscription_required
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ================= REPORT ================= #

@app.route("/report/<int:id>")
def get_report(id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    report = Prediction.query.get(id)
    user = User.query.get(session["user_id"])

    if not report:
        return jsonify({"error": "Not found"}), 404

    return jsonify({
        "centreName": user.centre_name,
        "address": user.address,
        "contactNo": user.contact_no,
        "email": user.email,
        "website": user.website,
        "radiologistName": user.radiologist_name,
        "licenseNo": user.license_no,
        "logo": user.logo,
        "signature": user.signature,

        "patient_name": report.patient_name,
        "patient_id": report.patient_id,
        "date": report.date,
        "sex": report.sex,
        "age": report.age,
        "mobile": report.mobile,
        "ref_by": report.ref_by,
        "result": report.result,
        "confidence": report.confidence,
        "image": report.image,
        "segmented_image": report.segmented_image,
        "tumor_area": report.tumor_area,

        "tumor": True if report.tumor_area and report.tumor_area > 0 else False
    })

# ================= HEALTH CHECK ================= #

@app.route("/check")
def check():
    return "SERVER RUNNING"

@app.route("/")
def home():
    return "Backend Running"

# ================= RUN ================= #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)