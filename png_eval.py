from PIL import Image
import sys
import os
import re

origin_path = './eval_png/'
dirs = os.listdir(origin_path)
output_path = './eval/'

for dir in dirs:
    print("The target directory is " + dir)
    input_path = origin_path + dir + '/'
    files = os.listdir(input_path)
    for file in files:
        input_im = Image.open(input_path + file)
        rgb_im = input_im.convert('RGB')
        rgb_im.save(output_path + dir + '/' + file.replace("png", "jpg"), quality=30)
        print("transcation finished for " + file)
