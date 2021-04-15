from colorama import init, Fore
init(autoreset=True)
github_repo = 'https://github.com/Midoriya-Gh0st/CSISFYP'
print(f"{Fore.BLUE}>>> [Git]. Visit the github repository to get more instructions: {github_repo} <<<")
print(f"{Fore.RED}[sys]. Please wait (1 min) for application initialisation...")
print(f"{Fore.RED}[sys]. service site url should show later...")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import time
import shutil

from flask import Flask, render_template, redirect, request, flash
from gevent.pywsgi import WSGIServer
from skimage import io
from gtts import gTTS

print("[0.-]. program lib loading...")

import app_configs
app = Flask(__name__)
app.config.from_object(app_configs)
print("[0.-]. setting application configs...")


def clear_src_buffer():
    # clear the audio and image buffer
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
    # print(file_save_loc)
    io.imsave(file_save_loc, image)
    return os.path.exists(file_save_loc)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def intro():
    app.config['APP_SITE'] = request.base_url
    # app.logger.info("--> the start page")
    url_select = app.config['APP_SITE'] + 'select/'
    return render_template('index.html', url_select=url_select)


@app.route('/select/')
def select_file():
    clear_src_buffer()
    print(f"[0.0]. current website: {request.url}")
    # app.logger.info("--> select file")
    url_upload = app.config['APP_SITE'] + 'upload/'
    return render_template('select.html', url_upload=url_upload)


@app.route('/upload/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file_url = request.form['url_file']
        file_extension = file_url.rsplit('.', 1)[-1]
        print(f"[1.1]. image type: {file_extension}.")
        if file_extension in app.config['ALLOWED_EXTENSIONS']:
            print(f"[1.2]. valid file from url {file_url}.")
            if not save_img_by_url(file_url):
                return "Cannot get image file by the URL, please check the URL!"
        else:
            print(f"[1.X]. INVALID FILE URL!")
            flash('No file part')
            return "File URL Invalid! Check the URL extension!"
    else:
        url_select = app.config['APP_SITE'] + 'select/'
        return render_template('index.html', url_select=url_select)
    # app.logger.info("--> upload file")
    return redirect(app.config['APP_SITE'] + 'evaluate/')


@app.route('/evaluate/')
def evaluate():
    port_number = [80, 135, 445, 3306, 8080, 1080]



    upload_id = str(app.config['UPLOAD_ID'])
    image_path = './static/upload/' + upload_id + '.jpg'

    app.config['AUDIO_ID'] = int(time.time() * 100)
    audio_id = str(app.config['AUDIO_ID'])
    audio_path = './static/audio/' + audio_id + '.mp3'

    # result, attention_plot = evaluate(image_path)
    result = ' '.join(app_configs.evaluate_2(image_path, encoder=app_configs.encoder_2, decoder=app_configs.decoder_2,
                                             tok=app_configs.tokenizer))
    result = result.rsplit('<end>', 1)[0]
    audio_src = gTTS(text=result, lang='en')
    audio_src.save(audio_path)
    print(f"[2.1]. generate audio/img id: {audio_id, upload_id}")
    print(f"[2.2]. save audio/img in path: {audio_path, image_path}")
    ###
    # implement the TS-API, generate the audio
    ###
    url_select = app.config['APP_SITE'] + 'select/'
    url_result = app.config['APP_SITE'] + 'result/'
    url_redirect = app.config['APP_SITE'] + 'evaluate/redirect/'

    while not os.path.exists(audio_path):
        print("[2.X]. bad network ... waiting for file...")

    if os.path.exists(audio_path):
        print(f"[2.3]. success get audio at: {audio_path}")
        return render_template('result.html', url_select=url_select, url_result=url_result,
                               url_redirect=url_redirect, result=result, upload_id=upload_id, audio_id=audio_id)
    else:
        return "No Audio File Error! Try again please."


@app.route('/evaluate/redirect/')
def rewrite():
    audio_path = './static/audio/' + str(app.config['AUDIO_ID']) + '.mp3'
    print(f"[3.1]. check audio path: {audio_path}")
    if os.path.exists(audio_path):  # should remove the (last) audio file before generating the new audio
        os.remove(audio_path)
    app.config['AUDIO_ID'] = 0
    print(f"[3.2]. rewriting audio id in configure: {app.config['AUDIO_ID']}")
    print(f"[3.3]. action target to: {app.config['APP_SITE'] + 'evaluate/'}")
    return redirect(app.config['APP_SITE'] + 'evaluate/')  # go back to the evaluate result page again


if __name__ == '__main__':
    print(f"{Fore.RED}[sys]. PLEASE ENSURE THE PORT 5000 *IS NOT* OCCUPIED ON YOUR MACHINE")
    server = WSGIServer(listener=('0.0.0.0', 5000), application=app, log=None)
    print(f"{Fore.GREEN}[sys]. Please visit http://127.0.0.1:{server.server_port} to start up the service.")
    server.serve_forever()
    # app.run()
