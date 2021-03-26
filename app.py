import os
import time
import shutil
import skimage

from flask import Flask, render_template, redirect, request, flash

from skimage import io
from gtts import gTTS

import configs

app = Flask(__name__)

app.config.from_object(configs)


def clear_src_buffer():
    # clear the aduio and image buffer
    shutil.rmtree('./static/audio')
    os.mkdir('./static/audio')
    shutil.rmtree('./static/upload')
    os.mkdir('./static/upload')


def save_img_by_url(img_url):
    image = io.imread(img_url)
    app.config['UPLOAD_ID'] = int(time.time() * 100)
    upload_id = str(app.config['UPLOAD_ID'])
    img_name = upload_id + '.jpg'
    file_save_loc = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
    print(file_save_loc)
    skimage.io.imsave(file_save_loc, image)
    return os.path.exists(file_save_loc)


# introduction of this project (Student ID, name, supervisor)
@app.route('/')
def intro():
    app.config['APP_SITE'] = request.base_url
    app.logger.info("--> the strat page")
    url_select = app.config['APP_SITE'] + 'select/'
    return render_template('index.html', url_select=url_select)


@app.route('/select/')
def select_file():
    clear_src_buffer()
    print("3:", request.url)
    app.logger.info("--> select file")
    url_upload = app.config['APP_SITE'] + 'upload/'
    return render_template('select.html', url_upload=url_upload)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/upload/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file_url = request.form['url_file']
        file_extension = file_url.rsplit('.', 1)[-1]
        print("File extension:", file_extension)
        if file_extension in app.config['ALLOWED_EXTENSIONS']:
            print("File URL Valid:", file_url)
            if not save_img_by_url(file_url):
                return "Cannot get image file by the URL, please check the URL!"
        else:
            print("File URL Invalid!")
            flash('No file part')
            return "File URL Invalid! Check the URL extension!"
    else:
        url_select = app.config['APP_SITE'] + 'select/'
        return render_template('index.html', url_select=url_select)
    app.logger.info("--> upload file")
    return redirect(app.config['APP_SITE'] + 'evaluate/')


@app.route('/evaluate/')
def evaluate():
    print(1, app.config['UPLOAD_ID'])
    app.logger.info("--> evaluation!")

    upload_id = str(app.config['UPLOAD_ID'])
    image_path = './static/upload/' + upload_id + '.jpg'

    app.config['AUDIO_ID'] = int(time.time() * 100)
    audio_id = str(app.config['AUDIO_ID'])
    audio_path = './static/audio/' + audio_id + '.mp3'

    # result, attention_plot = evaluate(image_path)
    result = ' '.join(configs.evaluate_2(image_path))
    result = result.rsplit('<end>', 1)[0]
    audio_src = gTTS(text=result, lang='en')
    audio_src.save(audio_path)
    print("[2] ------- audio/img id: ", audio_id, upload_id)
    print("[2.1] ------- audio/img path: ", audio_path, image_path)
    ###
    # implement the TS-API, generate the audio
    ###
    url_select = app.config['APP_SITE'] + 'select/'
    url_result = app.config['APP_SITE'] + 'result/'
    url_redirect = app.config['APP_SITE'] + 'evaluate/redirect/'

    while not os.path.exists(audio_path):
        print("Wait for file...")

    if os.path.exists(audio_path):
        print("[2.2] ------- File get: ", audio_path)
        return render_template('result.html', url_select=url_select, url_result=url_result,
                               url_redirect=url_redirect, result=result, upload_id=upload_id, audio_id=audio_id)
    else:
        return "No Audio File Error! Try again please."


@app.route('/evaluate/redirect/')
def rewrite():
    audio_path = './static/audio/' + str(app.config['AUDIO_ID']) + '.mp3'
    print("[3.1] ------- audio_path: ", audio_path)
    if os.path.exists(audio_path):  # should remove the (last) audio file before generating the new audio
        os.remove(audio_path)
    app.config['AUDIO_ID'] = 0
    print("[3] ------- conf. audio id: ", app.config['AUDIO_ID'])
    print("[4] ------- to where:", app.config['APP_SITE'] + 'evaluate/')
    return redirect(app.config['APP_SITE'] + 'evaluate/')  # go back to the evaluate result page again


if __name__ == '__main__':
    app.run()
