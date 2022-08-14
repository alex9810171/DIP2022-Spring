import requests
from base64 import b64encode
import random
import string
import cv2
CLIENT_ID = '47a12b43b020d51'
def upload_to_imgur(img):
    # get a random name string for the image
    name = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    # encode the image to base64
    ret, img_encoded = cv2.imencode('.jpg', img)
    img_encoded = b64encode(img_encoded)
    # upload the image to imgur
    response = requests.post('https://api.imgur.com/3/image',
                                headers={'Authorization': 'Client-ID ' + CLIENT_ID},
                                data={
                                    'image': img_encoded,
                                    'type': 'base64',
                                    'name': f'img_encoded.jpg',
                                    'title': name}).json()
    if response['status'] == 200:
        return response['data']['link']
    else:
        return None

if __name__ == '__main__':
    img = cv2.imread('../../data/result.png')
    print(upload_to_imgur(img))