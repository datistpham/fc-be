import cv2
import numpy as np
import os
import time
from deepface import DeepFace
from Silent_Face_Anti_Spoofing.src.anti_spoof_predict import AntiSpoofPredict
from Silent_Face_Anti_Spoofing.src.generate_patches import CropImage
from Silent_Face_Anti_Spoofing.src.utility import parse_model_name
from flask import Flask, jsonify, request
import mysql.connector
from flask_cors import CORS
import os
import base64
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './db_face/'
UPLOAD_FOLDER2 = './time_keeping/'
mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="time_keeping"
    )

    # Tạo một đối tượng cursor để truy xuất cơ sở dữ liệu
cursor = mydb.cursor()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
@app.after_request
def apply_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="time_keeping"
)
mycursor = mydb.cursor()

def process(img_path):

    model_dir = 'Silent_Face_Anti_Spoofing/resources/anti_spoof_models'
   
    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()
    
    image = cv2.imread(img_path)
    image_bbox = model_test.get_bbox(image)
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

    label = np.argmax(prediction)
    if label == 1:
        face_image = image[image_bbox[1]:image_bbox[1]+image_bbox[3], image_bbox[0]:image_bbox[0]+image_bbox[2]]
        if face_image is None:
            return 'no face'
        df = DeepFace.find(face_image,
                db_path = "db_face", 
                model_name = 'ArcFace',
                distance_metric= 'cosine',
                enforce_detection= False,
                detector_backend= 'skip',
                silent= True
            )
        if len(df['identity']) != 0 and df['ArcFace_cosine'][0] < 0.4:
            person_name = os.path.basename(df['identity'][0]).split('.')[0]
        else:
            person_name = 'no indentity'
        return person_name
    else:
        return "fake face"

        
@app.route("/signup", methods=["POST"])
async def signup():
    data= request.json
    name= data["name"]
    email= data["email"]
    phone= data["phone"]
    password= data["password"]
    try:
        mycursor.execute(f"SELECT email FROM employees WHERE email = '{email}'")
        results = mycursor.fetchall()
        if(len(results) > 0 ):
            return jsonify({"signup": False, "exist": True})
        mycursor.execute("INSERT INTO employees(name, email, phone, password) VALUES(%s, %s, %s, %s)", (name, email, phone, password))
        mydb.commit()
        mycursor.execute(f"SELECT id FROM employees WHERE email= '{email}'")
        results2= mycursor.fetchall()
        return jsonify({"signup": True, "uid": results2[0]})
    except Exception as e: 
        print(str(e))
        mydb.rollback()
        return {'signup': False}
@app.route("/staff/confirm-user", methods=["POST"])
async def confirmUser(): 
    file = request.form['file']
    name= request.form["name"]
    img = base64.b64decode(file.split(',')[1])
    filename = secure_filename(name+ '.png')
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'wb') as f:
        f.write(img)
    # print(process("hieu_anh.jpg"))
    return jsonify({"confirm": True})
@app.route("/login", methods=["POST"])
async def login():
    data= request.json
    email= data["email"]
    password= data["password"]
    try:
        mycursor.execute(f"SELECT * FROM employees WHERE email= '{email}' AND password= '{password}'")
        results = mycursor.fetchall()
        if(len(results)> 0):
            return jsonify({"login": True, "staff": results[0]})
        else:
            return jsonify({"login": False})
    except Exception as e: 
        print(e)
        return jsonify({"error": True})
@app.route("/api/timekeeping", methods=["POST"])
async def timekeeping():
    file = request.form['file']
    name= request.form["name"]
    img = base64.b64decode(file.split(',')[1])
    filename = secure_filename(name+ '.png')
    with open(filename, 'wb') as f:
        f.write(img)
    result= process(filename)
    os.remove(filename)
    return jsonify({"result": result})
@app.route("/staff/list", methods=["GET"])
async def getListStaff():
    try:
        staff_json= []
        mycursor.execute("SELECT id, name, email, phone FROM employees")
        staffs= mycursor.fetchall()
        for staff in staffs:
            staff_dict= {
                'id': staff[0],
                'name': staff[1],
                'email': staff[2],
                'phone': staff[3]
            }
            staff_json.append(staff_dict)
        return jsonify(staff_json)
    except Exception as e:
        return jsonify({"error": True})
@app.route("/api/detail/staff", methods=["GET"])
async def getDetailStaff():
    uid= request.args.get("uid")

    mycursor.execute(f"SELECT id, name, email, phone FROM employees WHERE id= {uid}")
    staffs= mycursor.fetchall()
    if(len(staffs) > 0):
        staffs= staffs[0]
        staff_dict= {
            'id': staffs[0],
            'name': staffs[1],
            'email': staffs[2],
            'phone': staffs[3]
        }
        return jsonify(staff_dict)
    return jsonify({"login": False})
if __name__ == '__main__':
    app.run(debug=True)