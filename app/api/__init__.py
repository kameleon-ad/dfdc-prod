import time
from pathlib import Path

from flask import Blueprint, jsonify, request

from app.extensions import DETECTOR

api_blueprint = Blueprint('api', __name__)


@api_blueprint.get('test')
def api_test():
    return jsonify({'status': 'API is working'})


@api_blueprint.post('reality_video')
def api_reality_video():
    video_file = request.files.get('video')
    if video_file is None:
        return jsonify({'status': 'Invalid Request'}), 400

    print('[*] Analysing', video_file.name)

    upload_path = Path('uploads')
    upload_path.mkdir(exist_ok=True, parents=True)

    filepath = upload_path / f'{time.time()}-{video_file.filename}'
    video_file.save(filepath)

    result = DETECTOR(str(filepath), boolean=False).item()
    print(result)

    return jsonify({'result': result})
