import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from services.qa_service import QASystem


def create_app() -> Flask:
    app = Flask(__name__)

    # Allow cross-origin requests (Vercel frontend)
    cors_origins = os.getenv("CORS_ORIGINS", "*")
    CORS(app, resources={r"/api/*": {"origins": cors_origins}}, supports_credentials=False)

    # Initialize Q&A system
    kb_path = os.getenv("KNOWLEDGE_BASE_PATH", "dataset.csv")
    qa_system = QASystem(kb_path)

    @app.get("/api/health")
    def health() -> tuple:
        return jsonify({"status": "ok"}), 200

    @app.post("/api/chat")
    def chat() -> tuple:
        payload = request.get_json(silent=True) or {}
        user_message = (payload.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "message is required"}), 400
        try:
            answer, meta = qa_system.get_response(user_message)
            return jsonify({"response": answer, "meta": meta}), 200
        except Exception as exc:  # Defensive: never expose stack trace
            return jsonify({"error": "internal_error", "detail": str(exc)}), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port)


