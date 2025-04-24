import os
from flask import Flask, redirect, url_for, flash
import subprocess
from importjson import extract_layoutlm_data

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "uploaded_jsons"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return """
        <h1>LayoutLM Trainer</h1>
        <a href="/train-model">🤖 Entrenar modelo LayoutLM</a><br><br>
        <form action="/upload-json" method="post" enctype="multipart/form-data">
            <p>Upload JSON File for Training:</p>
            <input type="file" name="json_file" accept=".json">
            <input type="submit" value="Upload">
        </form>
    """



#UPLOAD_FOLDER = "uploaded_jsons"
#os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload-json", methods=["POST"])
def upload_json():
    if 'json_file' not in request.files:
        flash("No file part in the request.", "danger")
        return redirect(url_for("index"))

    file = request.files['json_file']
    if file.filename == '':
        flash("No selected file.", "danger")
        return redirect(url_for("index"))

    if file and file.filename.endswith('.json'):
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)
        flash(f"✅ File uploaded successfully: {save_path}", "success")

        # Optionally, trigger processing here:
        try:
            label = "Invoice"   # You can make this dynamic if needed
            output = extract_layoutlm_data(save_path, label)
            flash(f"Processed and generated data at: {output}", "success")
        except Exception as e:
            flash(f"Error during processing: {str(e)}", "danger")

        return redirect(url_for("index"))
    else:
        flash("❌ Only .json files are allowed.", "danger")
        return redirect(url_for("index"))

@app.route("/train-model")
def train_model():
    try:
        subprocess.run(["python", "train_layoutlm.py"], check=True)
        flash("✅ Modelo entrenado exitosamente.", "success")
    except subprocess.CalledProcessError as e:
        flash(f"❌ Error al entrenar el modelo: {str(e)}", "danger")

    return redirect(url_for("index"))

@app.route('/generate-train-jsonl')
def generate_train_jsonl():
    try:
        base_dir = "output/sgd-results/511965/analysis/"   # Cambia esto a la ruta correcta
        file_name = "511965_factura-comercial_1_2025-04-17_140637.pdf_async_analysis.json"
        json_path = os.path.join(base_dir, file_name)
        label = "Invoice"
        output = extract_layoutlm_data(json_path, label)
        flash(f"Archivo generado correctamente en: {output}", "success")
    except Exception as e:
        flash(f"Error: {str(e)}", "danger")

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
