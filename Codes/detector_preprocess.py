
from unpacker import DigitStructWrapper
import pandas as pd
import numpy as np
import cv2
import os

TRAIN_PATH = "../Produced_dataset/"
VALID_PATH = "../Produced_dataset/"
IMAGE_SIZE = (32, 32)

PROCESSED_TRAIN_PATH = "../Data/detector/train/"
PROCESSED_VALID_PATH = "../Data/detector/valid/"

def create_dir(*args):
    for directory in args:
        if not os.path.exists(directory):
            os.makedirs(directory)


def perform_aggregation(dataframe):

    dataframe['bottom'] = dataframe['top']  + dataframe['height']
    dataframe['right']  = dataframe['left'] + dataframe['width']

    dataframe.drop(['height', 'width'], axis=1, inplace=True)

    dataframe = dataframe.groupby(['file', 'label'], as_index=False).agg(
        {'top': 'min', 'left': 'min', 'bottom': 'max', 'right': 'max'}
    )

    return dataframe


def expand_image(dataframe):

    dataframe['width_expand']  = (0.3 * (dataframe['right'] - dataframe['left']))  / 2.
    dataframe['height_expand'] = (0.3 * (dataframe['bottom']  - dataframe['top'])) / 2.

    dataframe['left']  -= dataframe['width_expand'].astype('int')
    dataframe['right'] += dataframe['width_expand'].astype('int')

    dataframe['top']    -= dataframe['height_expand'].astype('int')
    dataframe['bottom'] += dataframe['height_expand'].astype('int')

    dataframe.drop(['width_expand','height_expand'],axis=1,inplace=True)

    return dataframe


def get_image_size(dataframe):

    file_names = dataframe['file'].tolist()
    image_size = []

    for name in file_names:
        try:
            image = cv2.imread('../' + name)
            image_size.append(image.shape[:2])
        except:
             image_size.append((0, 0))

    image_x_size = [x for (x, y) in image_size]
    image_y_size = [y for (x, y) in image_size]

    dataframe['image_height'] = image_x_size
    dataframe['image_width']  = image_y_size

    dataframe = dataframe[dataframe.image_height > 0]

    return dataframe


def correct_boundaries(dataframe):

    dataframe['top'].loc[dataframe['top'] < 0] = 0
    dataframe['top'].loc[dataframe['left'] < 0] = 0

    dataframe['bottom'].loc[dataframe['bottom'] > dataframe['image_height']] = dataframe['image_height']
    dataframe['right'].loc[dataframe['right']  > dataframe['image_width']]  = dataframe['image_width']

    return dataframe



def crop_seqimage_and_save(dataframe,path,new_size):

    metadata = []

    for count, (index, rows) in enumerate(dataframe.iterrows()):
        try:
                image = cv2.imread('../' + rows['file'])
                directory_yes = path + "/1.0/"
                create_dir(directory_yes)
                file_name = directory_yes + str(count) + ".png"
                if rows['left'] < 0:
                    rows['left'] = 0
                if rows['top'] < 0:
                    rows['top'] = 0
                cropped_image = image[int(rows['top']):int(rows['bottom'])
                , int(rows['left']):int(rows['right'])]

                cropped_image = cv2.resize(cropped_image, new_size)

                cv2.imwrite(file_name, cropped_image)

                #get non-digit patch
                directory_no = path + "/0.0/"
                create_dir(directory_no)
                file_name = directory_no + str(count) + ".png"

                if int(rows['top']) > 32:
                    if int(rows['left']) > 32:

                        cropped_image = image[0:32,0:32]
                        cropped_image = cv2.resize(cropped_image, new_size)

                        cv2.imwrite(file_name, cropped_image)

                    if int(rows['left']) < 32:

                        cropped_image = image[0:32,32:int(rows['image_width'])]
                        cropped_image = cv2.resize(cropped_image, new_size)

                        cv2.imwrite(file_name, cropped_image)

                if  int(rows['image_height']) - int(rows['bottom']) > 32:

                    if int(rows['image_width']) - int(rows['right']) > 32:

                        cropped_image = image[int(rows['bottom']):int(rows['image_height']),
                                        int(rows['right']):int(rows['image_width'])]
                        cropped_image = cv2.resize(cropped_image, new_size)

                        cv2.imwrite(file_name, cropped_image)

                    if int(rows['image_width']) - int(rows['right']) < 32:
                        cropped_image = image[int(rows['bottom']):int(rows['image_height']),
                                        0:32]
                        cropped_image = cv2.resize(cropped_image, new_size)

                        cv2.imwrite(file_name, cropped_image)

        except:
            print("Error at row --> " + str(rows))



if __name__ == "__main__":

        create_dir(PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, PROCESSED_VALID_PATH)

        print('read csvfile')

        train_df = pd.read_csv(TRAIN_PATH + "train.csv")
        valid_df = pd.read_csv(TEST_PATH + "valid.csv")

        print('aggregation')
        train_df = perform_aggregation(train_df)
        valid_df = perform_aggregation(valid_df)

        #increase the box by 30%

        print('expad')
        train_df = expand_image(train_df)
        valid_df = expand_image(valid_df)


        #append image size

        print('size')
        train_df = get_image_size(train_df)
        valid_df = get_image_size(valid_df)

        #correct the expanded bounding box

        print('boundaries')
        train_df = correct_boundaries(train_df)
        valid_df  = correct_boundaries(valid_df)

        #crop the images to 32 X 32

        crop_seqimage_and_save(train_df, PROCESSED_TRAIN_PATH, IMAGE_SIZE)
        crop_seqimage_and_save(valid_df, PROCESSED_VALID_PATH, IMAGE_SIZE)
        






















