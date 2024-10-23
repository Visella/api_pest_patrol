from flask import Flask, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)

app.wsgi_app = ProxyFix(app.wsgi_app)

@app.route('/')
def index():
    return jsonify({"message": "Hello from Flask on Vercel!"})

# For Vercel to handle the app
def handler(event, context):
    return app(event, context)
