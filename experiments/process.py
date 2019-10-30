import os

# with open("TestImages/frames") as f:
# from api import PRN
# from texture import texture_editing

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # GPU number, -1 for CPU
# prn = PRN(is_dlib=True)
from experiments.main import _evaluate

path = "frames/"
files = os.listdir(path)
files.sort()
for index, file in enumerate(files):
	print(file)
	_evaluate(path+file, "images/21styles/starry_night.jpg", "results/frame_{}.png".format(index), model="models/21styles.model")