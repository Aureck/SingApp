import os
from flask import Flask, request
from werkzeug.utils import secure_filename
from evaluate_model import evaluate_model

app = Flask(__name__)
TMP_DIR = os.path.join(os.getcwd(), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return "sing App API"

@app.route("/upload_video", methods=["POST"])
def upload():
    f = request.files["video"]
    fname = secure_filename(f.filename)
    path = os.path.join(TMP_DIR, fname)
    f.save(path)
    sentence = evaluate_model(src=path)
    return " - ".join(sentence[::-1])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
