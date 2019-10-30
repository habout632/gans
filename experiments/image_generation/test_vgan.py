import os
D_PATH = 'models/vgan/a.pth'
print(os.getcwd()+D_PATH)
if os.path.isfile(D_PATH):
	print("existed")
