from flask import Flask, render_template, request, redirect, url_for
import os
from post_processing import analyze_tumor_region

app = Flask(__name__)

# ===============================
# Configuration
# ===============================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ===============================
# Routes
# ===============================

# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Choice Page
@app.route("/choice")
def choice():
    return render_template("choice.html")


# About Page
@app.route("/about")
def about():
    return render_template("about.html")


# Upload Page
@app.route("/upload")
def upload():
    return render_template("upload.html")


# ===============================
# Prediction Route
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    # Get patient details
    patient_name = request.form.get("name")
    age = request.form.get("age")
    gender = request.form.get("gender")

    # Get uploaded file
    file = request.files["image"]

    if file.filename == "":
        return redirect(url_for("upload"))

    # Save file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # ===============================
    # Model Prediction
    # ===============================
    result = analyze_tumor_region(filepath)

    # Add image name for frontend display
    result["image_name"] = file.filename

    # ===============================
    # Render Result Page
    # ===============================
    return render_template(
        "result.html",
        result=result,
        patient_name=patient_name,
        age=age,
        gender=gender
    )


# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
