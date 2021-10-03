'''
Author: Peter Yu
Reference: https://zhuanlan.zhihu.com/p/206947798
'''

from flask import Flask,render_template,request,jsonify
import time, pickle, json, cv2, requests, base64
from requests.models import Response
from io import BytesIO
import numpy as np

app = Flask( __name__)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def getResult(img):

    data = json.dumps({
        "instances": [img],
    })
    headers = {"content-type": "application/json"}
    json_response = requests.post(
        'http://localhost:8501/v1/models/my_model:predict',
        data=data, headers=headers)
    predictions = json.loads(json_response.text)
    content = []
    for i in predictions['predictions'][0]:
        if i==3 or tokenizer.word_index['<unk>']==i: continue
        if i==4: break
        content.append(i)
    text = tokenizer.sequences_to_texts([content])[0]
    print(text)
    return text

@app.route('/{}'.format("model"), methods=['POST'])
def predict():
    if request.method == 'POST' and request.files.get('image_file'):
        file = request.files.get('image_file')
        img_size = 320
        img = file.read()
        im_b64 = base64.b64encode(img)
        im_bytes = base64.b64decode(im_b64)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        img = cv2.resize(img,(img_size, img_size)).tolist()
        value = getResult(img)
        return jsonify({'text':  value})
    else:
        return jsonify({"error_code": "1001"})

if __name__ == '__main__':
    # production environment
    production = True
    if production:
        from waitress import serve
        serve(app, host="0.0.0.0", port=8080)
    else:
        app.run('0.0.0.0', port=8123, debug=True)
