import random

import arabic_reshaper
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os

from bidi.algorithm import get_display
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

out_path = '../Data/Graded_images/'


def sliding_window(img, classifier, detector, window=(32, 32)):

    bounding_box = {}
    numbers = ["۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"]

    step_size = 20
    for row in range(0, img.shape[0] - window[0], step_size):#32 works
        for col in range(0, img.shape[1] - window[1], step_size):

            cropped_img = img[row:row + window[0], col:col + window[1]]
            cropped_img = cv2.resize(cropped_img, (32, 32))

            cropped_img = np.expand_dims(cropped_img, axis=0)

            detection = detector.predict(cropped_img)
            #print ("detect -> " ,detection[0][1])
            prediction = classifier.predict(cropped_img)

            if max(prediction[0]) == 1. and detection[0][1] == 1.:

                label = np.argmax(prediction[0])
                others = [prediction[0][i] for i in range(len(prediction[0])) if i != label]
                if sum(others) == 0. and label != 10:
                    print(numbers[label], row, col)
                    bounding_box[label] = (row, col)

    return bounding_box


def create_dir(*args):
    for directory in args:
        if not os.path.exists(directory):
            os.makedirs(directory)


def isRectangleOverlap(R1, R2):

    if (R1[0] >= R2[2]) or (R1[2] <= R2[0]) or (R1[3] <= R2[1]) or (R1[1] >= R2[3]):
        return False
    else:
        return True

if __name__ == '__main__':


    # input_path = "../Data/Input_Images/"
    input_path = "../Produced_dataset/test_one_digit/"
    csv_path = "../Produced_dataset/test_processed.csv"
    test_df = pd.read_csv(csv_path)

    # for image_name in os.listdir(input_path):
    Images = {}
    classifier = tf.keras.models.load_model("../Saved_models/Classifier/myCNN.h5")
    detector = tf.keras.models.load_model("../Saved_models/Detector/myDetectorCNN.h5")
    numbers = ["۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"]

    for count, (index, rows) in enumerate(test_df.iterrows()):

        if rows['file'] not in Images.keys():

              Images[rows['file']] = 1

              create_dir(out_path)
              img = cv2.imread('../' + rows['file'])
              copied_image = img.copy()
              (h, w) = copied_image.shape[:2]
              img = cv2.resize(img, (200, 200))

              mser = cv2.MSER_create()
              gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
              gray = cv2.GaussianBlur(gray, (15, 15), 0)
              vis = img.copy()

              regions, _ = mser.detectRegions(gray)
              hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
              cv2.polylines(vis, hulls, 1, (0, 255, 0))

              mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
              mask = cv2.dilate(mask, np.ones((150, 150), np.uint8))

              for contour in hulls:
                  cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
                  text_only = cv2.bitwise_and(img, img, mask=mask)

              bounding_box = sliding_window(text_only.copy(), classifier, detector)


        for label, (height, width) in bounding_box.items():

              # a = image[int(rows['top']):int(rows['bottom']), int(rows['left']):int(rows['right'])]
              cv2.rectangle(img, (width, height), (width + 32, height + 32), (255, 0, 0), 1)
              # cv2.rectangle(img, (int(rows['top']), int(rows['left'])), (int(rows['bottom']), int(rows['right'])), (255, 255, 255), 1)
              # cv2.rectangle(img, (int(rows['bottom']), int(rows['right'])), (int(rows['bottom']) + 10, int(rows['right']) + 10), (255, 255, 255), 1)

              Detect_rect = [width, height, width + 32, height + 32]
              Orig_rect = [int(rows['top']) - 32, int(rows['left']) + 16, int(rows['top']), int(rows['left']) + 32 + 16]

              image_name = rows['file'].strip().split("/")[2]

              if rows['label'] == numbers[label]:
                      # if isRectangleOverlap(Orig_rect, Detect_rect):
                      #       print('Number is detected correctly...', image_name, ':', rows['label'])
                      #       print('region_orig:', Orig_rect)
                      #       print('region_detect', Detect_rect)

                      reshaped_text = arabic_reshaper.reshape(numbers[label])  # correct its shape
                      bidi_text = get_display(reshaped_text)  # correct its direction

                      # start drawing on image![](../Data/Graded_images/7117 (copy).jpg)
                      img = Image.fromarray(img)
                      # img = Image.open(out_path + image_name)
                      draw = ImageDraw.Draw(img)
                      position_1 = width-15
                      position_2 = height+15

                      font_size = 30
                      fontFile = "../../Sahel.ttf"
                      font = ImageFont.truetype(fontFile, font_size)
                      draw.text((position_1, position_2), bidi_text, font=font, fill=(255, 255, 0))
                      draw = ImageDraw.Draw(img)

                      img = np.asarray(img)

        cv2.imwrite(out_path + image_name, img)
