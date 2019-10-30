import numpy as np
from aip import AipBodyAnalysis
import cv2
import base64

""" 你的 APPID AK SK """
APP_ID = '17363443'
API_KEY = 'cxAUPSbo7xCQlGIGsYDhSZ6C'
SECRET_KEY = 'XzaYnnD8qrf9cYwEWF4SdXvENiqNlUxk'

client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)

""" 读取图片 """


def get_file_content(filePath):
	with open(filePath, 'rb') as fp:
		return fp.read()


image = get_file_content('habout.jpg')

""" 调用人像分割 """
client.bodySeg(image)

""" 如果有可选参数 """
options = {"type": "foreground"}

""" 带参数调用人像分割 """
result = client.bodySeg(image, options)
print(type(result))

# decode base64 encoded string to bytes
test = base64.b64decode(result.get("foreground"))
# cv2.imwrite("smm.jpg", result)
print(test)


# write bytes to file
with open("tgb.png", "wb") as f:
	f.write(test)
# nparr = np.fromstring(test, np.uint8)
# img_new = cv2.imdecode(nparr, 1)
# img_new = np.where(img_new == 1, 255, img_new)
# # cv2.imwrite("img_new.png", img_new)
