# from PIL import Image, ImageFont, ImageDraw
#
# my_image = Image.open("0.jpg")
# # title_font = ImageFont.truetype('playfair/playfair-font.ttf', 200)
# title_text = "۱۹".encode('utf8')
#
# image_editable = ImageDraw.Draw(my_image)
# image_editable.text((15,15), title_text, (237, 230, 211))#, font=title_font)
#
# my_image.save("result.jpg")

# Tested on Python 3.6.1

# install: pip install --upgrade arabic-reshaper
import csv
import os
import random

import arabic_reshaper

# install: pip install python-bidi
from bidi.algorithm import get_display

# install: pip install Pillow
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

def create_dataset(rand_num, image_name, path_image, image):

    # use a good font!
    fontFile = "Sahel.ttf"

    # load the font and image
    font_size = random.randint(20, 50)
    font = ImageFont.truetype(fontFile, font_size)

    # first you must prepare your text (you dont need this step for english text)
    numbers = ["۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"]
    try:

        reshaped_text = arabic_reshaper.reshape(numbers[rand_num])    # correct its shape
        bidi_text = get_display(reshaped_text)  # correct its direction

        # start drawing on image
        draw = ImageDraw.Draw(image)
        position_1 = random.randint(25, image.size[0] - 25)
        position_2 = random.randint(25, image.size[1] - 25)

        draw.text((position_1, position_2), bidi_text, (0, 0, 0), font=font)
        draw = ImageDraw.Draw(image)

        # save it
        image.save(path_image + image_name)

        return position_1, position_2, font_size, image

    except:
        return -1, -1, -1, -1
    # reshaped_text = arabic_reshaper.reshape(numbers[rand_num])  # correct its shape



# Get the list of all files and directories
# in the root directory
path = "../../DataSet/Cityscape_1/Data/"
dir_list = os.listdir(path)

print("Files and directories in '", path, "' :")
numbers = ["۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"]

# print the list
# print(dir_list)

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
    fields = ['file', 'height', 'label', 'left', 'top', 'width']
    csvwriter.writerow(fields)

    print(len(dir_list))
    path = "../../DataSet/Cityscape_1/Data/"
    for ind, image_name in enumerate(dir_list[12000:15000]):

        print(ind)
        image = Image.open(path + image_name)
        for j in range(0, 4):
            rand_num = random.randint(0, 9)
            path_image = "Produced_dataset/test_one_digit/"
            pos_1, pos_2, font_size, image = create_dataset(rand_num, image_name, path_image, image)
            if pos_1 > 0:
                csvwriter.writerow([str(ind), path_image + image_name, font_size-3, numbers[rand_num], pos_1 - int(font_size/4), pos_2+int(font_size/4), font_size-3])

