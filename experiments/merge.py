import cv2

# from ma
import numpy as np
from PIL import Image

# fg = cv2.imread("fg.png", cv2.IMREAD_UNCHANGED)
fg = cv2.imread("sky.jpg")

fg = cv2.resize(fg, (962, 1280))
cv2.imwrite("test10.png", fg)
# im = Image.open("fg.png")
# rgb_im = im.convert('RGB')
# rgb_im.save("fg.jpg")
# cv2.imshow("image", fg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # cv2.imshow("image", img1)
bg = cv2.imread("sky.jpg")
# bg = cv2.imread("background.png", cv2.IMREAD_UNCHANGED)
# bg = cv2.cvtColor(bg, cv2.COLOR_RGB2RGBA)
# # im = Image.open("pyramid.jpg")
# # im.save("pyramid.png")
#
# # dim = (620, 825)
# #
# # img1_resized = cv2.resize(img1, dim)
# #
# # img2_resized = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)
# # img = (img1_resized + img2_resized) / 2
# #
# fg = cv2.resize(fg, (481, 640))
# bg = cv2.resize(bg, (400, 400))
# # fg = fg[0:698, :]
# bg = bg[0:fg.shape[0], 0:fg.shape[1]]
# bg = cv2.resize(bg, (962, 1280))
# cv2.imshow("image", bg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.addWeighted(bg, 1, fg, 1, 0)
# img = cv2.add(bg, fg)
#
# #
# # img = img1/2+img2/2
# # # img = (img1+img2)/2.0

mask = cv2.imread("labelmap.jpg", 0)
# im = Image.open("labelmap.png")
# rgb_im = im.convert('RGB')
# rgb_im.save("labelmap.jpg")
# mask = cv2.resize(mask, (200, 300))
# res = cv2.bitwise_and(fg, fg, mask=mask)
# cv2.imshow("image", res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
mask = mask * 255
cv2.imwrite("mask.png", mask)

# fg_mask = 255 * np.ones(fg.shape, fg.dtype)
# fg_mask[:] = (255, 255, 255, 255)
width, height, channels = bg.shape
center = (height // 2, width // 2)
# center = (0, 0)
output = cv2.seamlessClone(fg, bg, mask, center, cv2.NORMAL_CLONE)

cv2.imwrite("tour1.png", output)
