from flask import Flask, render_template, request, redirect, jsonify, url_for
import uuid
import os
from werkzeug.utils import secure_filename
import keras
import tensorflow as tf
import numpy as np
import cv2
import json
import boto3
from boto3.dynamodb.conditions import Key, Attr

import sentry_sdk
import requests
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn=os.environ['sentry_dsn'],
    integrations=[FlaskIntegration()]
)
app = Flask(__name__, static_folder='uploads')

app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "TIFF"]

boto3.setup_default_session(region_name='us-west-2')
aws_access_key_id_= os.environ['aws_access_key_id_']
aws_secret_access_key_=os.environ['aws_secret_access_key_']
s3 = boto3.resource('s3',aws_access_key_id=aws_access_key_id_,aws_secret_access_key=aws_secret_access_key_)
dyn = boto3.resource('dynamodb',aws_access_key_id=aws_access_key_id_,aws_secret_access_key=aws_secret_access_key_)
bucket_name = 'ramcovid'
image_table = "ramcovid"
table_name = dyn.Table(image_table)

def allowed_image(filename):

    # We only want files with a . in the filename
    if not "." in filename:
        return False

    # Split the extension from the filename
    ext = filename.rsplit(".", 1)[1]

    # Check if the extension is in ALLOWED_IMAGE_EXTENSIONS
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

           
def prediction(file_name):

    img = cv2.resize(cv2.cvtColor(cv2.imread(file_name),cv2.COLOR_BGR2RGB),(256,256))
    img = img/255
    global sess1
    sess1 = tf.Session()
    keras.backend.set_session(sess1)
    global model

    model = keras.models.Sequential()
    # model.add(keras.layers.Lambda(lambda x: x/255))
    #Feature extraction layers
    model.add(keras.layers.Convolution2D(16,3,input_shape=(256,256,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Convolution2D(16,3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPooling2D(strides=2))
    model.add(keras.layers.Convolution2D(32,3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Convolution2D(32,3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPooling2D(strides=2))
    model.add(keras.layers.Convolution2D(64,3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Convolution2D(64,3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPooling2D(strides=2))
    model.add(keras.layers.Convolution2D(128,3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Convolution2D(128,3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPooling2D(strides=2))


    #Classification layer
    model.add(keras.layers.Convolution2D(128,4))

    ##average pooling
    model.add(keras.layers.Flatten())

    ##Dropout(0.3)
    model.add(keras.layers.Dropout(0.3))

    #output
    model.add(keras.layers.Dense(4,activation='softmax'))
    model.load_weights('model.h5')

    # model = keras.models.load_model('dense.h5')

    global graph1
    graph1 = tf.get_default_graph()
    with graph1.as_default():
        keras.backend.set_session(sess1)
        y_p = model.predict(np.reshape(img,(1,256,256,3)))
        y_p = np.around(y_p * 100,decimals = 2).T
        return y_p[0][0],y_p[1][0],y_p[2][0],y_p[3][0]
    

@app.route('/', methods=["GET", "POST"])
def home():
    print('home')
    global fn
    if 'predict button' in request.form:
        if request.method == 'POST':
            if request.files:
                image = request.files['image']

                if image.filename == "":
                    print("No filename")
                    return redirect(request.url)

                if allowed_image(image.filename):
                    # post this to slack
                    webhook_url="https://hooks.slack.com/services/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"
                    event_slack = {'text': " New Image uploaded"}
                    response = requests.post(webhook_url, 
                                        data=json.dumps(event_slack),
                                        headers={'Content-Type': 'application/json'}
                                    )
                    filename = secure_filename(image.filename)
                    filename = os.path.join('uploads', str(uuid.uuid4()) + '.'+filename.split('.')[-1])
                    image.save(filename)
                    print('image saved', filename.split('/')[1])
                    fn = filename.split('/')[1]

                    s3.meta.client.upload_file(filename, bucket_name, filename.split('/')[1], ExtraArgs={'ACL':'public-read'})

                    normal,bacterial,viral,covid = prediction(filename)
                    print(normal,bacterial,viral,covid)

                    response = table_name.put_item(
                            Item={
                                    'image_id': filename.split('/')[1],
                                    'Normal': str(normal),
                                    'Bacterial': str(bacterial),
                                    'Viral': str(viral),
                                    'Covid-19':str(covid),
                                    'image_name': image.filename
                                    }
                                                                )
                    return render_template('image.html', image_name=filename.split('/')[1] , normal = normal, bacterial=bacterial, viral=viral, covid=covid)
    elif 'share button' in request.form:
        return redirect(url_for('share', image_id=fn))
    return render_template('home.html')

           
@app.route('/debug-sentry')
def trigger_error():
    division_by_zero = 1 / 0
    

@app.route('/share/<image_id>')
def share(image_id):
    # get the image id from the uset
    # return a html that has the results
    
    # get the results from DB
    print("in share")
    response = table_name.query(
                             KeyConditionExpression=Key('image_id').eq(image_id)
                    )
    if response['Count'] > 0:
        # there is a match
        result = response['Items'][0]
        image_name=result['image_name']
        Normal = result['Normal']
        Bacterial = result['Bacterial']
        Viral = result['Viral']
        Covid = result['Covid-19']
        return render_template('share.html', image_name=image_name, normal=Normal,
                                           bacterial=Bacterial, viral=Viral, covid=Covid)
    else:
        # no match
        return jsonify({})

if __name__ == '__main__':
    app.run(debug=True)