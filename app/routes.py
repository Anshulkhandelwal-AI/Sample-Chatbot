import os
from flask import Blueprint, request, jsonify, render_template
from app.ragpipeline import RAGPipeline
from werkzeug.utils import secure_filename

bp = Blueprint("routes", __name__)
UPLOAD_FOLDER = 'app/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

pipeline = RAGPipeline(
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
    pc_api_key=os.getenv("PINECONE_API_KEY"),
    pc_env=os.getenv("PINECONE_ENV")
)

@bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@bp.route("/upload", methods=["POST"])
def upload_files():
    if 'file' not in request.files:
        return jsonify({"Error": "No file part"}), 400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"Error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        pipeline.Load_content(file_path)
        pipeline.Embedding_Upsert()
        return jsonify({'message': "PDF embedded successfully."}), 200
    except Exception as e:
        return jsonify({"Error": str(e)}), 500

@bp.route("/query", methods=["POST"])
def handle_query():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"Error": "Query is required."}), 400

    try:
        contexts = pipeline.Retriever(query)
        answer = pipeline.Generate_answer(query, contexts)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"Error": str(e)}), 500
