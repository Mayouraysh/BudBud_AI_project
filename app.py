import gc
import io
import json
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
    # Video formats (kept for backward compatibility / fallback)
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    # Audio formats (browser-extracted audio)
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".m4a",
    ".aac",
    ".weba",
}

_model = None
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
SUBTITLE_MODE = os.getenv("SUBTITLE_MODE", "segment").lower()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "1536"))
JOB_RETENTION_SECONDS = int(os.getenv("JOB_RETENTION_SECONDS", "3600"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
JOBS = {}
UPLOADS = {}
JOBS_LOCK = threading.Lock()
STATE_DIR = os.path.join(tempfile.gettempdir(), "subtitle_jobs_state")
os.makedirs(STATE_DIR, exist_ok=True)


def job_state_path(job_id):
    return os.path.join(STATE_DIR, f"{job_id}.json")


def save_job_state(job_id, job):
    payload = {
        "status": job.get("status"),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "filename": job.get("filename"),
        "srt_path": job.get("srt_path"),
    }
    with open(job_state_path(job_id), "w", encoding="utf-8") as state_file:
        json.dump(payload, state_file)


def load_job_state(job_id):
    state_path = job_state_path(job_id)
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as state_file:
            return json.load(state_file)
    except Exception:
        return None


def ensure_job(job_id, filename):
    with JOBS_LOCK:
        job = {
            "status": "queued",
            "error": None,
            "created_at": time.time(),
            "filename": filename,
            "srt_path": None,
            "work_dir": None,
        }
        JOBS[job_id] = job
        save_job_state(job_id, job)


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
            state_path = job_state_path(job_id)
            if os.path.exists(state_path):
                os.remove(state_path)

        stale_upload_ids = []
        for upload_id, upload in UPLOADS.items():
            age = now - upload["created_at"]
            if age > JOB_RETENTION_SECONDS:
                stale_upload_ids.append(upload_id)

        for upload_id in stale_upload_ids:
            upload = UPLOADS.pop(upload_id, None)
            if not upload:
                continue
            temp_dir = upload.get("temp_dir")
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)


def run_transcription_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["status"] = "processing"
        input_path = job["input_path"]
        output_path = job["output_path"]
        save_job_state(job_id, job)

    try:
        create_srt(input_path, output_path)

        # Delete the input file immediately to free disk space
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except OSError:
            pass

        # Force garbage collection to free RAM on low-memory hosts
        gc.collect()

        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job["status"] = "done"
            job["srt_path"] = output_path
            save_job_state(job_id, job)
    except Exception as exc:
        app.logger.exception("Subtitle generation failed")

        # Clean up input file even on failure
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except OSError:
            pass
        gc.collect()

        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job["status"] = "error"
            job["error"] = str(exc)
            save_job_state(job_id, job)


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


@app.post("/upload/init")
def upload_init():
    cleanup_old_jobs()
    data = request.get_json(silent=True) or {}
    filename = secure_filename(data.get("filename", "video.mp4"))
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400
    if not allowed_file(filename):
        return jsonify({"error": "Unsupported file type"}), 400

    upload_id = uuid.uuid4().hex
    temp_dir = tempfile.mkdtemp(prefix=f"chunk_upload_{upload_id}_")
    input_path = os.path.join(temp_dir, filename)
    with open(input_path, "wb"):
        pass

    with JOBS_LOCK:
        UPLOADS[upload_id] = {
            "created_at": time.time(),
            "filename": filename,
            "temp_dir": temp_dir,
            "input_path": input_path,
            "bytes_written": 0,
        }

    return jsonify({"upload_id": upload_id}), 201


@app.post("/upload/chunk/<upload_id>")
def upload_chunk(upload_id):
    cleanup_old_jobs()
    chunk_file = request.files.get("chunk")
    if chunk_file is None:
        return jsonify({"error": "Missing chunk file"}), 400

    with JOBS_LOCK:
        upload = UPLOADS.get(upload_id)
        if not upload:
            return jsonify({"error": "Upload session not found or expired"}), 404
        input_path = upload["input_path"]

    chunk_bytes = chunk_file.read()
    with open(input_path, "ab") as target:
        target.write(chunk_bytes)

    with JOBS_LOCK:
        upload = UPLOADS.get(upload_id)
        if upload:
            upload["bytes_written"] += len(chunk_bytes)

    return jsonify({"status": "ok", "received_bytes": len(chunk_bytes)}), 200


@app.post("/upload/complete/<upload_id>")
def upload_complete(upload_id):
    cleanup_old_jobs()
    with JOBS_LOCK:
        upload = UPLOADS.pop(upload_id, None)
        if not upload:
            return jsonify({"error": "Upload session not found or expired"}), 404

        filename = upload["filename"]
        input_path = upload["input_path"]
        temp_dir = upload["temp_dir"]
        stem = Path(filename).stem
        output_path = os.path.join(temp_dir, f"{stem}.srt")
        job_id = uuid.uuid4().hex

        # Create job inline to avoid nested lock deadlock.
        JOBS[job_id] = {
            "status": "queued",
            "error": None,
            "created_at": time.time(),
            "filename": filename,
            "srt_path": None,
            "work_dir": temp_dir,
            "input_path": input_path,
            "output_path": output_path,
        }
        save_job_state(job_id, JOBS[job_id])

    worker = threading.Thread(target=run_transcription_job, args=(job_id,), daemon=True)
    worker.start()
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.get("/job/<job_id>")
def get_job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            restored = load_job_state(job_id)
            if restored:
                JOBS[job_id] = {
                    "status": restored.get("status", "error"),
                    "error": restored.get("error"),
                    "created_at": restored.get("created_at", time.time()),
                    "filename": restored.get("filename"),
                    "srt_path": restored.get("srt_path"),
                    "work_dir": None,
                }
                job = JOBS[job_id]
                if job["status"] == "processing":
                    job["status"] = "error"
                    job["error"] = "Server restarted during processing. Please retry this upload."
                    save_job_state(job_id, job)
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