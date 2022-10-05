from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

cap = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)


import cv2
import numpy as np

from keras.models import load_model
from keras_preprocessing.image import img_to_array
from playsound import playsound
from threading import Thread


def start_alarm(sound):
    """Play the alarm sound"""
    playsound('data/alarm.mp3')

def gen_frames():  # generate frame by frame from camera
        
    classes = ['Closed', 'Open']
    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
    right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")
    cap = cv2.VideoCapture(0)
    model = load_model("drowiness_new7.h5")
    count = 0
    alarm_on = False
    alarm_sound = "data/alarm.mp3"
    status1 = ''
    status2 = ''

    while True:
        _, frame = cap.read()
        height = frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            left_eye = left_eye_cascade.detectMultiScale(roi_gray)
            right_eye = right_eye_cascade.detectMultiScale(roi_gray)
            for (x1, y1, w1, h1) in left_eye:
                cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
                eye1 = roi_color[y1:y1+h1, x1:x1+w1]
                eye1 = cv2.resize(eye1, (145, 145))
                eye1 = eye1.astype('float') / 255.0
                eye1 = img_to_array(eye1)
                eye1 = np.expand_dims(eye1, axis=0)
                pred1 = model.predict(eye1)
                status1=np.argmax(pred1)
                break

            for (x2, y2, w2, h2) in right_eye:
                cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
                eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
                eye2 = cv2.resize(eye2, (145, 145))
                eye2 = eye2.astype('float') / 255.0
                eye2 = img_to_array(eye2)
                eye2 = np.expand_dims(eye2, axis=0)
                pred2 = model.predict(eye2)
                status2=np.argmax(pred2)
                break

            # If the eyes are closed, start counting
            if status1 == 2 and status2 == 2:
                count += 1
                cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                # if eyes are closed for 10 consecutive frames, start the alarm
                if count >= 10:
                    cv2.putText(frame, "Drowsiness Alert!!!", (100, height-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    if not alarm_on:
                        alarm_on = True
                        # play the alarm sound in a new thread
                        t = Thread(target=start_alarm, args=(alarm_sound,))
                        t.daemon = True
                        t.start()
            else:
                cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                count = 0
                alarm_on = False

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show resul

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)