from flask import Flask, render_template
import os

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))

app = Flask(__name__, template_folder=template_dir)

@app.route('/')
def index():
    return render_template('reviews.html')

def iniciar_servidor_flask():
    app.run(debug=False, use_reloader=False)