from flask import Flask, render_template, jsonify, url_for, send_from_directory
from dotenv import load_dotenv

from util import decode_vin_corgi, decode_vin_vininfo, decode_vin_b95


load_dotenv()

app = Flask(__name__)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                          'favicon.ico',mimetype='image/vnd.microsoft.icon')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/corgi/<vin>')
def api_corgi(vin):
    result = decode_vin_corgi(vin)
    return jsonify(result)

@app.route('/api/vininfo/<vin>')
def api_vininfo(vin):
    result = decode_vin_vininfo(vin)
    return jsonify(result)

@app.route('/api/b95/<vin>')
def api_b95(vin):
    result = decode_vin_b95(vin)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=8080, debug=True, host='0.0.0.0')