from io import BytesIO
import requests, json, datetime, time

'''
WARNING: DO NOT USE MCGILL NETWORK TO SEND TEST REQUESTS!

Test pictures source:

test.jpg: https://www.apple.com/newsroom/2018/02/apple-watch-series-3-now-tracks-skiing-and-snowboarding-activity/

test2.jpg: https://fotochuk.com/blog/fine-art/scenic/urban/streets-of-old-montreal-after-rain/
'''
def recognize_captcha(file):

    s = time.time()
    with open(file,'rb') as f:
        content = f.read()

    files = {'image_file': (file, BytesIO(content), 'application')}
    r = requests.post(url='https://httpbin.org/post', files=files, timeout = 12)
    e = time.time()
    print("API Response: {}".format(r.text))
    predict_text = json.loads(r.text)
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("【{}】Time Taken: {}ms Result: {}".format(now_time, int((e-s)*1000), predict_text))

if __name__=='__main__':
    recognize_captcha('test2.jpg')