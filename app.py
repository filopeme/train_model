import os
import json
from flask import Flask,request, redirect, url_for, flash,get_flashed_messages
import subprocess
from importjson import extract_layoutlm_data
from importjson import predict_document
from importjson import extract_key_value_pairs
from importjson import extract_tables_from_textract

UPLOAD_FOLDER = "/app/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app = Flask(__name__)
app.secret_key = "supersecretkey"

#UPLOAD_FOLDER = "uploaded_jsons"
#os.makedirs(UPLOAD_FOLDER, exist_ok=True)



@app.route("/")
def index():
    messages = get_flashed_messages(with_categories=True)
    message_html = ""

    for category, msg in messages:
        # Basic styling per category
        color = {
            "success": "green",
            "info": "blue",
            "warning": "orange",
            "danger": "red"
        }.get(category, "black")

        message_html += f"<div style='color:{color}; margin:10px 0;'><strong>{msg}</strong></div>"

    return f"""
        <h1>LayoutLM Trainer & Predictor</h1>
        {message_html}

        <a href="/train-model">🤖 Train Model Manually</a><br><br>

        <form action="/upload-json" method="post" enctype="multipart/form-data">
            <p><strong>Upload JSON File:</strong></p>
            <input type="file" name="json_file" accept=".json" required><br><br>

            <label for="action">Choose Action:</label><br>
            <input type="radio" id="train" name="action" value="train" checked>
            <label for="train">➕ Add to Training Data</label><br>

            <input type="radio" id="predict" name="action" value="predict">
            <label for="predict">🔎 Predict Document Type</label><br><br>

            <!-- Label selection if training -->
            <div id="labelSelect">
                <label for="label">Select Label for Training:</label>
                <select name="label">
                    <option value="Invoice">Invoice</option>
                    <option value="Not Invoice">Not Invoice</option>
                </select><br><br>
            </div>

            <input type="submit" value="Submit">
        </form>

        <script>
            // JS to hide label selector when Predict is chosen
            document.querySelectorAll('input[name="action"]').forEach((elem) => {{
                elem.addEventListener("change", function() {{
                    const labelDiv = document.getElementById("labelSelect");
                    if (this.value === "predict") {{
                        labelDiv.style.display = "none";
                    }} else {{
                        labelDiv.style.display = "block";
                    }}
                }});
            }});
        </script>
    """

#UPLOAD_FOLDER = "uploaded_jsons"
#os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route("/upload-json", methods=["POST"])
def upload_json():
    if 'json_file' not in request.files:
        flash("No file part in the request.", "danger")
        print("No file part in the request.", "danger")
        return redirect(url_for("index"))

    file = request.files['json_file']
    if file.filename == '':
        flash("No selected file.", "danger")
        return redirect(url_for("index"))

    if file and file.filename.endswith('.json'):
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        print(f"Saving file to {save_path}")    
        file.save(save_path)
        flash(f"✅ File uploaded successfully: {save_path}", "success")

        try:
            # 🔍 Step 0: Extract key-value pairs from Textract JSON
            with open(save_path, 'r') as f:
                textract_data = json.load(f)
            kv_result = extract_key_value_pairs(textract_data)

            # Optional: show results in console or flash summary
            print("🧾 Extracted Fields:")
            for k, v in kv_result.items():
                print(f"{k}: {v}")
            flash(f"✅ Extracted {len(kv_result)} field-value pairs.", "success")
            # Save key-value pairs as JSON
            output_json_path = os.path.join(app.config['UPLOAD_FOLDER'], "extracted_fields.json")
            with open(output_json_path, "w", encoding="utf-8") as outfile:
                json.dump(kv_result, outfile, indent=2, ensure_ascii=False)
            flash(f"📁 Saved extracted fields to: {output_json_path}", "info")
            # Store it if needed for display in template or session
            # session["extracted_fields"] = kv_result

            # Step 1: Process file for LayoutLM
            label = request.form.get("label", "Invoice")
            jsonl_path = extract_layoutlm_data(save_path, label)
            flash(f"Processed and saved for training: {jsonl_path}", "success")
            # Extract table
            df_table = extract_tables_from_textract(textract_data)

            # Save as CSV (optional)
            df_table.to_csv("extracted_invoice_table.csv", index=False)
            # Step 2: If 'predict' option is selected, run prediction
            if request.form.get("action") == "predict":
                predicted_label, confidence = predict_document(jsonl_path)
                flash(f"🔎 Prediction: {predicted_label} ({confidence:.8%} confidence)", "info")

        except Exception as e:
            flash(f"❌ Error during processing: {str(e)}", "danger")

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



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
