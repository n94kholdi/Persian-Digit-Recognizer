import csv
import os
import random

import arabic_reshaper

import cv2
import numpy as np
from bidi.algorithm import get_display

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

def create_dataset(rand_num, image_name, path_image, image):

    # use a good font!
    fontFile = "Sahel.ttf"

    # load the font and font_size
    font_size = random.randint(20, 50)
    font = ImageFont.truetype(fontFile, font_size)

    numbers = ["۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"]
    try:

        reshaped_text = arabic_reshaper.reshape(numbers[rand_num])    # correct its shape
        bidi_text = get_display(reshaped_text)  # correct its direction

        # start drawing on image
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        position_1 = random.randint(25, image.size[0] - 25)
        position_2 = random.randint(25, image.size[1] - 25)

        draw.text((position_1, position_2), bidi_text, (0, 0, 0), font=font)
        draw = ImageDraw.Draw(image)

        image = np.asarray(image)
        shift = int(font_size/3)
        # cv2.rectangle(image, (position_1, position_2 + shift), (position_1 + int(font_size/2), position_2 + font_size), (255, 255, 255), 1)

        # save it
        # image.save(path_image + image_name)
        # cv2.imwrite(path_image + image_name, image)

        return position_1, position_2, font_size, image

    except:
        return -1, -1, -1, image



# Get the list of all files and directories
path = "../../DataSet/Cityscape_1/Data/"
dir_list = os.listdir(path)

print("Files and directories in '", path, "' :")
numbers = ["۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"]

with open('Produced_dataset/train.csv', 'w') as csvfile:

    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    fields = ['file', 'height', 'label', 'left', 'top', 'width']
    csvwriter.writerow(fields)

    print(len(dir_list))
    path = "../../DataSet/Cityscape_1/Data/"
    for ind, image_name in enumerate(dir_list[:9000]):
        print(ind)
        image = Image.open(path + image_name)
        for j in range(0, 4):
            rand_num = random.randint(0, 9)
            path_image = "Produced_dataset/train_one_digit/"
            pos_1, pos_2, font_size, image = create_dataset(rand_num, image_name, path_image, image)
            if pos_1 > 0:
                csvwriter.writerow([str(ind), path_image + image_name, font_size-3, numbers[rand_num], pos_1 - int(font_size/4), pos_2+int(font_size/4), font_size-3])


with open('Produced_dataset/valid.csv', 'w') as csvfile:

    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    fields = ['file', 'height', 'label', 'left', 'top', 'width']
    csvwriter.writerow(fields)

    print(len(dir_list))
    path = "../../DataSet/Cityscape_1/Data/"
    for ind, image_name in enumerate(dir_list[9000:12000]):

        print(ind)
        image = Image.open(path + image_name)
        for j in range(0, 4):
            rand_num = random.randint(0, 9)
            path_image = "Produced_dataset/valid_one_digit/"
            pos_1, pos_2, font_size, image = create_dataset(rand_num, image_name, path_image, image)
            if pos_1 > 0:
                csvwriter.writerow([str(ind), path_image + image_name, font_size-3, numbers[rand_num], pos_1 - int(font_size/4), pos_2+int(font_size/4), font_size-3])


with open('Produced_dataset/test.csv', 'w') as csvfile:

    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    fields = ['file', 'label', 'left', 'top', 'width', 'height']
    csvwriter.writerow(fields)

    print(len(dir_list))
    path_ref = "../../DataSet/Cityscape_1/Data/"
    path_dest = "Produced_dataset/test_one_digit/"

    for ind, image_name in enumerate(dir_list[12000:15000]):

        print(ind)
        image = cv2.imread(path_ref + image_name)
        image = cv2.resize(image, (200, 200))
        for j in range(0, 4):
            rand_num = random.randint(0, 9)
            pos_1, pos_2, font_size, image = create_dataset(rand_num, image_name, path_dest, image)
            shift = int(font_size/3)
            if pos_1 > 0:
                csvwriter.writerow([str(ind), path_dest + image_name, numbers[rand_num], pos_1, pos_2 + shift, pos_1 + int(font_size / 2), pos_2 + font_size])

        image = Image.fromarray(image)
        image.save(path_dest + image_name)
        # cv2.imwrite(path_dest + image_name, image)


