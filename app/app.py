
from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "runs",
    "detect",
    "results",
    "aquavision_yolov8",
    "weights",
    "best.pt"
)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Load trained YOLO model
model = YOLO(MODEL_PATH)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    detections = []
    error = None

    if request.method == "POST":

        if "image" not in request.files:
            error = "Please select an image."
            return render_template(
                "index.html",
                error=error
            )

        file = request.files["image"]

        if file.filename == "":
            error = "Please select an image."
            return render_template(
                "index.html",
                error=error
            )

        if not allowed_file(file.filename):
            error = "Please upload a JPG, JPEG, PNG or WEBP image."
            return render_template(
                "index.html",
                error=error
            )

        # Generate a unique filename
        filename = str(uuid.uuid4()) + "." + file.filename.rsplit(".", 1)[1].lower()
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        file.save(image_path)

        # Run object detection
        predictions = model.predict(
            source=image_path,
            conf=0.25,
            imgsz=320,
            save=False
        )

        prediction = predictions[0]

        # Save annotated image
        output_filename = "result_" + filename
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)

        annotated_image = prediction.plot()
        import cv2
        cv2.imwrite(output_path, annotated_image)

        result_image = output_filename

        # Extract detected objects
        for box in prediction.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            detections.append({
                "class_name": class_name,
                "confidence": round(confidence * 100, 2)
            })

    return render_template(
        "index.html",
        result_image=result_image,
        detections=detections,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)

