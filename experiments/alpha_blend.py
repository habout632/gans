from PIL import Image


def changeImageSize(maxWidth, maxHeight, image):
	widthRatio = maxWidth / image.size[0]
	heightRatio = maxHeight / image.size[1]
	newWidth = int(widthRatio * image.size[0])
	newHeight = int(heightRatio * image.size[1])
	newImage = image.resize((newWidth, newHeight))
	return newImage


bg = Image.open("pyramid.png").convert("RGBA")
bg = changeImageSize(600, 800, bg)
bg = bg.convert("RGBA")

fg = Image.open("tgb.png")
fg = changeImageSize(600, 800, fg)

Image.alpha_composite(bg, fg).save("test.png")
# bg.paste(fg, (0, 0), fg)
# bg.show()
