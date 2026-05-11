import io
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from faster_whisper import WhisperModel
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename


app = Flask(__name__)

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
}

_model = None
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
SUBTITLE_MODE = os.getenv("SUBTITLE_MODE", "segment").lower()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
JOB_RETENTION_SECONDS = int(os.getenv("JOB_RETENTION_SECONDS", "3600"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
JOBS = {}
JOBS_LOCK = threading.Lock()


def ensure_job(job_id, filename):
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "error": None,
            "created_at": time.time(),
            "filename": filename,
            "srt_path": None,
            "work_dir": None,
        }


def cleanup_old_jobs():
    now = time.time()
    stale_ids = []
    with JOBS_LOCK:
        for job_id, job in JOBS.items():
            age = now - job["created_at"]
            if age > JOB_RETENTION_SECONDS:
                stale_ids.append(job_id)

        for job_id in stale_ids:
            job = JOBS.pop(job_id, None)
            if not job:
                continue
            work_dir = job.get("work_dir")
            if work_dir and os.path.exists(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)


def run_transcription_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["status"] = "processing"
        input_path = job["input_path"]
        output_path = job["output_path"]

    try:
        create_srt(input_path, output_path)
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job["status"] = "done"
            job["srt_path"] = output_path
    except Exception as exc:
        app.logger.exception("Subtitle generation failed")
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job["status"] = "error"
            job["error"] = str(exc)


def get_model():
    global _model
    if _model is None:
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
    return _model


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def to_hinglish(text):
    # Prefer high-quality transliteration if optional packages are available.
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    except Exception:
        pass

    # Fallback transliteration path available in many Python setups.
    try:
        from unidecode import unidecode

        return unidecode(text)
    except Exception:
        return text


def create_srt(video_path, output_srt):
    model = get_model()
    use_word_timestamps = SUBTITLE_MODE == "word"
    segments, info = model.transcribe(
        video_path,
        word_timestamps=use_word_timestamps,
        beam_size=1,
        best_of=1,
        task="transcribe",
        language="hi",
        initial_prompt="Yeh Hindi audio hai. Kripya Hindi mein transcribe karo.",
    )
    detected_language = (info.language or "hi").lower()

    with open(output_srt, "w", encoding="utf-8") as srt_file:
        counter = 1
        for segment in segments:
            if use_word_timestamps and segment.words:
                for word in segment.words:
                    text = word.word.strip()
                    if not text:
                        continue
                    if detected_language.startswith("hi"):
                        text = to_hinglish(text).strip()
                        if not text:
                            continue

                    start = format_time(word.start)
                    end = format_time(word.end)
                    srt_file.write(f"{counter}\n")
                    srt_file.write(f"{start} --> {end}\n")
                    srt_file.write(f"{text}\n\n")
                    counter += 1
            else:
                text = (segment.text or "").strip()
                if not text:
                    continue
                if detected_language.startswith("hi"):
                    text = to_hinglish(text).strip()
                    if not text:
                        continue

                start = format_time(segment.start)
                end = format_time(segment.end)
                srt_file.write(f"{counter}\n")
                srt_file.write(f"{start} --> {end}\n")
                srt_file.write(f"{text}\n\n")
                counter += 1


@app.get("/")
def index():
    return send_from_directory(".", "ui.html")


@app.post("/upload")
def upload_video():
    cleanup_old_jobs()
    if "video" not in request.files:
        return jsonify({"error": "No video file found in request"}), 400

    uploaded_file = request.files["video"]
    if uploaded_file.filename == "":
        return jsonify({"error": "Please select a video file"}), 400

    filename = secure_filename(uploaded_file.filename)
    if not allowed_file(filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        job_id = uuid.uuid4().hex
        temp_dir = tempfile.mkdtemp(prefix=f"subtitle_job_{job_id}_")
        input_path = os.path.join(temp_dir, filename)
        stem = Path(filename).stem
        output_path = os.path.join(temp_dir, f"{stem}.srt")
        uploaded_file.save(input_path)

        ensure_job(job_id, filename)
        with JOBS_LOCK:
            JOBS[job_id]["input_path"] = input_path
            JOBS[job_id]["output_path"] = output_path
            JOBS[job_id]["work_dir"] = temp_dir

        worker = threading.Thread(target=run_transcription_job, args=(job_id,), daemon=True)
        worker.start()
        return jsonify({"job_id": job_id, "status": "queued"}), 202
    except Exception as exc:
        app.logger.exception("Subtitle generation failed")
        return jsonify({"error": str(exc)}), 500


@app.get("/job/<job_id>")
def get_job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found or expired"}), 404
        payload = {"status": job["status"]}
        if job["status"] == "error":
            payload["error"] = job["error"] or "Transcription failed"
        if job["status"] == "done":
            payload["download_url"] = f"/download/{job_id}"
        return jsonify(payload), 200


@app.get("/download/<job_id>")
def download_job_result(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found or expired"}), 404
        if job["status"] != "done" or not job.get("srt_path"):
            return jsonify({"error": "Subtitle file is not ready yet"}), 409
        srt_path = job["srt_path"]
        filename = Path(job["filename"]).stem + ".srt"

    if not os.path.exists(srt_path):
        return jsonify({"error": "Generated file no longer exists"}), 410

    with open(srt_path, "rb") as srt_file:
        srt_bytes = srt_file.read()

    return send_file(
        io.BytesIO(srt_bytes),
        as_attachment=True,
        download_name=filename,
        mimetype="application/x-subrip",
    )


@app.errorhandler(413)
def request_entity_too_large(_error):
    return (
        jsonify(
            {
                "error": f"File too large. Max allowed size is {MAX_UPLOAD_MB} MB on free hosting."
            }
        ),
        413,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)