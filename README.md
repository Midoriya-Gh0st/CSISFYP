# FYP-Image-Captioning

## 1. Introduction
This project is dedicated to providing people with visual impairments to understand the content of pictures in social media by voice.

Through multi-modal machine learning, using CNN as the encoder, RNN (GRU) as the decoder, and gTTS as the speech converter, a picture-text-speech content conversion chain is formed.

 At the same time, through transfer learning from the Google Interception-v3 model, a large number of image features are obtained, which improves the encoder's ability to a certain extent.

This project also uses the Attention mechanism, which allows the agent to find the main object in the picture.



## 2. Client
The client GUI of this project is built through the Python Flask framework, allowing MVC development.

The client is the exe program packaged by Pyinstaller. In order to facilitate the understanding of the project process, the background console display is enabled for this project, and the test output in the program can be verified in the background.

By loading the training model saved during the development process, the client can use it directly without training the model for a long time.



## 3. User's Guidance
Step-01: Program initialization
1) Unzip the folder
2) Click iMagic.exe to start the program
3) Wait for the program to initialize (load related libraries and models) until a link is displayed (usually 127.0.0.1:5000)
4) Note: It must be ensured that the localhost port 5000 is not occupied.
5) Paste the address in the browser to jump to the main page of the program.

Step-02: Service using
1) Click `Start Now` on the main page (http://127.0.0.1:5000/) to enter the image upload page (http://127.0.0.1:5000/select/).

<div><img src="https://raw.githubusercontent.com/Midoriya-Gh0st/CSISFYP/master/static/images/ui/index_page.PNG"  width="400" height="200" alt="index page"> </div>

2) Paste the URL of the picture in .jpg format into the text box. Note that the picture format currently only supports .jpg
3) You can use your image link, as well as the example links listed in the section 4 (Example Images' URL).

<div><img src="https://raw.githubusercontent.com/Midoriya-Gh0st/CSISFYP/master/static/images/ui/file_upload.PNG"  width="400" height="200" alt="index page"> </div>

4) Click `Submit`, the program will jump to the result page (http://127.0.0.1:5000/evaluate/). You may need to wait a while, because the image resource needs to be downloaded from the URL to the local and parsed into tensor format.
5) Pictures and predicted captions should now be displayed.

<div><img src="https://raw.githubusercontent.com/Midoriya-Gh0st/CSISFYP/master/static/images/ui/result_page.PNG"  width="400" height="200" alt="index page"> </div>

6) The voice will be played automatically, <u>**Note: If you wear headphones, please be sure to control the volume**</u>
7) Click `Try Again`, you can re-predict the current picture.
8) Click `Select New`, you can return to the homepage and upload the picture again.
9) You can view the background output which records the operation process of the program.




## 4. Example Images' URL

Example 01: `bike_man`
https://raw.githubusercontent.com/Midoriya-Gh0st/CSISFYP/master/static/examples/example01_bike_man.jpg

Example 02:  `giraffe_eating`
https://raw.githubusercontent.com/Midoriya-Gh0st/CSISFYP/master/static/examples/example02_giraffe_eating.jpg

Example 03: `umbrella_lady`
https://raw.githubusercontent.com/Midoriya-Gh0st/CSISFYP/master/static/examples/example03_umbrella_lady.jpg
