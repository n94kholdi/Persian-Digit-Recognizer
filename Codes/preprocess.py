
from unpacker import DigitStructWrapper
import pandas as pd
import numpy as np
import cv2
import os

TRAIN_PATH = "../Produced_dataset/"
TEST_PATH  = "../Produced_dataset/"
VALID_PATH = "../Produced_dataset/"
IMAGE_SIZE = (32, 32)

PROCESSED_TRAIN_PATH = "../Data/processed/train/"
PROCESSED_VALID_PATH = "../Data/processed/valid/"
PROCESSED_TEST_PATH  = "../Data/processed/test/"


def crop_image_and_save(dataframe, path, new_size):

    metadata = []

    for count, (index, rows) in enumerate(dataframe.iterrows()):

        try:
                print(count)
                image = cv2.imread('../' + rows['file'])
                if rows['label'] == 10:
                    rows['label'] = 0.0
                directory = path + str(rows['label']) + "/"

                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_name = directory + str(count) + ".png"

                if rows['left'] < 0:
                    rows['left'] = 0
                if rows['top'] < 0:
                    rows['top'] = 0

                cropped_image = image[int(rows['top']):int(rows['top']) + int(rows['height'])
                , int(rows['left']):int(rows['left']) + int(rows['width'])]

                cropped_image = cv2.resize(cropped_image, new_size)

                cv2.imwrite(file_name, cropped_image)

                metadata.append({'file': file_name, 'label': np.array(rows['label'])})
        except:
            print("Error at row --> " + str(rows))


    pd.DataFrame(metadata).to_csv(path+'metadata.csv')


if __name__ == "__main__":

    train_df = pd.read_csv(TRAIN_PATH + "train.csv")
    valid_df = pd.read_csv(VALID_PATH + "valid.csv")
    test_df  = pd.read_csv(TEST_PATH + "test.csv")

    #crop the images to 64 X 64

    crop_image_and_save(train_df, PROCESSED_TRAIN_PATH, IMAGE_SIZE)
    crop_image_and_save(valid_df, PROCESSED_VALID_PATH, IMAGE_SIZE)
    crop_image_and_save(test_df, PROCESSED_TEST_PATH, IMAGE_SIZE)























