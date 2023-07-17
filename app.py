from flask import Flask, request, render_template, url_for
import os
from findingSinglePerson import *

app = Flask(__name__)

# route to render home page
@app.route('/')
def home():
    return render_template('home.html')

# route to handle photo and video inputs
@app.route('/display', methods=['POST'])
def display():
    # get photo and video inputs from user
    photo = request.files['photo']
    video = request.files['video']

    photo_path = os.path.join('inputImg', 'input.jpg')
    photo.save(photo_path)
    video_path = os.path.join('inputVid', 'input.mp4')
    video.save(video_path)

    run()

    return render_template('display.html')

if __name__ == '__main__':
    app.run(debug=True)
