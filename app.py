from flask import Flask, redirect, url_for, flash
import subprocess

app = Flask(__name__)
app.secret_key = "supersecretkey"

@app.route("/")
def index():
    return """
        <h1>LayoutLM Trainer</h1>
        <a href="/train-model">🤖 Entrenar modelo LayoutLM</a>
    """

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
