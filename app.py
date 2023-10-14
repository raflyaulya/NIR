from flask import Flask, render_template, request, jsonify
from analysis_customer_reviews import format_airline_name           # Ganti dengan modul analisis kamu

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    airline_name = request.form['airline_name']
    result = format_airline_name(airline_name)  # Panggil fungsi analisis Python kamu
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
