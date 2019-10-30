# encoding:utf-8
import base64
import urllib
# import urllib2
import requests

'''
手势识别
'''

request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/gesture"

# 二进制方式打开图片文件
f = open('2.png', 'rb')
img = base64.b64encode(f.read())

params = {"image": img}
# params = urllib.urlencode(params)

access_token = '24.244686cbb3855693ab2690a11c46f12c.2592000.1571553134.282335-17289545'
request_url = request_url + "?access_token=" + access_token
response = requests.post(url=request_url, data=params, headers={'Content-Type': "application/x-www-form-urlencoded"})
content = response.content
if content:
	print(content)
