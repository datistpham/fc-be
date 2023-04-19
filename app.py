from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
import time
from deepface import DeepFace
from Silent_Face_Anti_Spoofing.src.anti_spoof_predict import AntiSpoofPredict
from Silent_Face_Anti_Spoofing.src.generate_patches import CropImage
from Silent_Face_Anti_Spoofing.src.utility import parse_model_name


app=Flask(__name__)
camera = cv2.VideoCapture(0)


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames():
    model_dir = 'Silent_Face_Anti_Spoofing/resources/anti_spoof_models'
    vid = cv2.VideoCapture(0)

    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()
    if os.path.isfile(os.path.join("db_face", "representations_arcface.pkl")):
        os.remove(os.path.join("db_face", "representations_arcface.pkl"))  
    while True:
        success, image = vid.read()  # read the camera frame
        if not success:
            break
        else:
            time_string = ""

            start_time = time.time()
            image_bbox = model_test.get_bbox(image)
            detect_face_time = time.time() - start_time

            time_string += "detect face time: {} ".format(detect_face_time)

            start_time = time.time()
            prediction = np.zeros((1, 3))
            # sum the prediction from single model's result
            for model_name in os.listdir(model_dir):
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                param = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False
                img = image_cropper.crop(**param)
                start = time.time()
                prediction += model_test.predict(img, os.path.join(model_dir, model_name))

            # draw result of prediction
            label = np.argmax(prediction)
            value = prediction[0][label]/2
            anti_spoofing_time = time.time() - start_time
            time_string += "anti spoofing time: {} ".format(anti_spoofing_time)

            if label == 1:
                start_time = time.time()
                face_image = image[image_bbox[1]:image_bbox[1]+image_bbox[3], image_bbox[0]:image_bbox[0]+image_bbox[2]]
                if face_image is None:
                    continue
                df = DeepFace.find(face_image,
                        db_path = "db_face", 
                        model_name = 'ArcFace',
                        distance_metric= 'cosine',
                        enforce_detection= False,
                        detector_backend= 'skip'
                    )
                if len(df['identity']) != 0 and df['ArcFace_cosine'][0] < 0.4:
                    person_name = os.path.basename(df['identity'][0]).split('.')[0]
                    result_text = "RealFace: " + person_name
                    color = (255, 0, 0)
                else:
                    person_name = 'no indentity'
                    result_text = "RealFace: " + person_name
                    color = (0, 0, 255)
                indentify_face_time = time.time() - start_time
                time_string += "indentify face time: {}".format(indentify_face_time)
            else:
                result_text = "FakeFace"
                color = (0, 0, 255)
            cv2.rectangle(
                image,
                (image_bbox[0], image_bbox[1]),
                (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                color, 2)
            cv2.putText(
                image,
                result_text,
                (image_bbox[0], image_bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

            print(time_string)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)