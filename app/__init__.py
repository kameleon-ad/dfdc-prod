from flask import Flask, send_from_directory

from app.api import api_blueprint


def create_app(config=None):
    app = Flask(__name__)

    app.register_blueprint(api_blueprint, url_prefix='/api')

    @app.get('/')
    def root():
        return send_from_directory('static', 'index.html')

    return app
