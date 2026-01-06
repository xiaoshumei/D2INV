from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import json
from api.app import d2inv
import os

app = Flask(__name__)
CORS(app)
app.static_folder = "web"


@app.route("/api/d2inv_stream", methods=["GET"])
def d2inv_stream():
    try:
        dataset_name = request.args.get("dataset_name")
        if dataset_name:
            unique_filename = dataset_name
        else:
            return jsonify({"error": "No dataset provided"}), 400

        def generate():
            for result in d2inv(unique_filename):
                yield f"data: {json.dumps(result)}\n\n"

        return Response(generate(), mimetype="text/event-stream")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/list_datasets", methods=["GET"])
def list_datasets():
    """get all available datasets"""
    try:
        import os

        datasets_dir = "./datasets"
        if os.path.exists(datasets_dir):
            files = os.listdir(datasets_dir)
            return jsonify(files)
        else:
            return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    file_path = os.path.join("web", filename)
    if os.path.exists(file_path):
        return send_from_directory("web", filename)
    else:
        return send_from_directory("web", "index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
