from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2
import numpy as np
from modules.preprocessing import crop_and_transform
from modules.upload_image import upload_to_imgur
from modules.enhancement import adjust_contrast_brightness, gamma_correction, simplest_cb, top_hat_transform
from modules.shadow_removal import shadow_removal_ICASSP2018
from modules.ocr.ocr import OCRAPI_for_Web
import glob


app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024
answer_images = [cv2.imread(file) for file in sorted(glob.glob(
    'modules/ocr/data/processedImage/answer_letter/*.jpg'))]


@app.route('/')
def entry_point():
    return redirect(url_for('preprocessing'), code=302)


@app.route('/preprocessing')
def preprocessing():
    return render_template('preprocessing.html')


@app.route('/enhancement')
def enhancement():
    return render_template('enhancement.html', image_url=request.args.get('image_url', ''))


@app.route('/ocr')
def ocr():
    return render_template('ocr.html', image_url=request.args.get('image_url', ''))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/api/preprocessing', methods=['POST'])
def preprocessing_api():
    if 'image' not in request.files:
        return Response(status=400)
    image_buffer = np.frombuffer(request.files['image'].read(), np.uint8)
    if len(image_buffer) == 0:
        return Response(status=400)
    img = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    output, result = crop_and_transform(img, request.form.get('size', 'A4'))

    output_link, result_link = upload_to_imgur(output), upload_to_imgur(result)
    if output_link is None or result_link is None:
        return Response(status=500)
    else:
        return jsonify({'output': output_link, 'result': result_link})


@app.route('/api/enhancement', methods=['POST'])
def enhancement_api():
    if 'image' not in request.files:
        return Response(status=400)
    image_buffer = np.frombuffer(request.files['image'].read(), np.uint8)
    if len(image_buffer) == 0:
        return Response(status=400)
    img = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if request.form.get('method') == 'tophat':
        result = top_hat_transform(
            img, int(request.form.get('kernel_size', 21)))
    elif request.form.get('method') == 'contrast':
        result = adjust_contrast_brightness(img, float(request.form.get('multiplier', 1.0)),
                                            float(request.form.get('shift', 0.0)))
    elif request.form.get('method') == 'grayscale':
        result = img
        if result.shape[2] == 3:
            result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif request.form.get('method') == 'gamma':
        result = gamma_correction(img, float(request.form.get('gamma', 1.0)),
                                  float(request.form.get('gain', 1.0)))
    elif request.form.get('method') == 'colorbalance':
        result = simplest_cb(img, float(request.form.get('percent', 1.0)))
    elif request.form.get('method') == 'shadowremove':
        result = shadow_removal_ICASSP2018(
            img, 1)
    else:
        result = img

    result_link = upload_to_imgur(result)
    if result_link is None:
        return Response(status=500)
    else:
        return jsonify({'result': result_link})


@app.route('/api/ocr', methods=['POST'])
def ocr_api():
    if 'image' not in request.files:
        return Response(status=400)
    image_buffer = np.frombuffer(request.files['image'].read(), np.uint8)
    if len(image_buffer) == 0:
        return Response(status=400)
    img = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if img.shape[2] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mode = [26, 51] if 'upperOnly' in request.form else [0, 51]
    result, letter_images, letter_info, sentence = OCRAPI_for_Web(
        [img], answer_images, int(request.form.get('space', 5)), int(request.form.get('newline', 15)), mode)
    result_link = upload_to_imgur(result)
    if result_link is None:
        return Response(status=500)
    else:
        return jsonify({'result': result_link, 'sentence': sentence})


if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)
    # app.run(debug=True)
