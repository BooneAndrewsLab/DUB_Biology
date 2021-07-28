# Copyright (c) 2017, Oren Kraus All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Segmentation script by Nil Sahin and Matej Usaj
# Edited by Myra Paz Masinas on January 2018

from segment_predict_lib import *
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import fcntl
import h5py
import copy
import re

rc_phenix = re.compile('.+r([0-9]{2})c([0-9]{2})f.+-ch[0-9]sk([0-9])fk.+')
rc_tiff = re.compile('([0-9]{3})([0-9]{3})[0-9]{3}')
scale_factor = 1
img_type = 'phenix'
model = 'chong'
opt_save = 'cell'

localizationTerms = ['Actin', 'Bud Site', 'Bud Neck', 'Cell Periphery', 'Cytoplasm',
                     'Cytoplasmic Foci', 'Eisosomes', 'Endoplasmic Reticulum', 'Endosome',
                     'Golgi', 'Lipid Particles', 'Mitochondria', 'Mitotic Spindle', 'None',
                     'Nuclear Periphery', 'Nuclear Periphery Foci', 'Nucleolus', 'Nucleus',
                     'Peroxisomes', 'Punctate Nuclear', 'Vacuole', 'Vacuole Periphery']
numClasses = len(localizationTerms)
SCREEN_NUMCHANNELS = 3

model_ckpt = '/home/myra/PycharmProjects/trainmodel/logs_harsha_trial5/model.ckpt-8000'
testpath = '/home/myra/harsha/train_final/trial5/harsha_test_dataset.hdf5'
outfile_c = '/home/myra/harsha/train_final/trial5/cells_loc.csv'
test_ds = h5py.File(testpath, 'r')
images = test_ds['info1'].value
cell_cnt = images.shape[0]
print('Total cell count: %s' %cell_cnt)

def get_identifiers(ms):
    idf = []
    mapsheet_df = pd.read_csv(ms, index_col=False)
    mapcols = mapsheet_df.columns
    plate_index = mapcols.get_loc('Plate')
    for i in range(plate_index):
        idf.append(mapcols[i])
    return idf, mapsheet_df


def write_csv(dataframe, outfile, **kwargs):
    with open(outfile, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        dataframe.to_csv(outfile, **kwargs)
        fcntl.flock(f, fcntl.LOCK_UN)


def process_crops_loc(processedBatch, predicted_y, inputs, is_training, keep_prob, sess):
    crop_list = np.zeros((len(processedBatch), 5, numClasses))
    for crop in range(5):
        images = processedBatch[:, crop, :, :, :]
        if model == 'chong':
            tmp = copy.copy(sess.run([predicted_y], feed_dict={inputs: images, is_training: False}))
        else:
            tmp = copy.copy(sess.run([predicted_y], feed_dict={inputs: images, is_training: False, keep_prob: 1.0}))
        crop_list[:, crop, :] = tmp[0]

    mean_crops = np.mean(crop_list, 1)
    del crop_list
    return mean_crops


class ScreenClass:
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """

    def __init__(self, data):
        self.sql_data = data
        gfp_images = np.unique(self.sql_data['Image'])
        gfp_images.sort()
        self.wells = np.unique([seq for seq in gfp_images])
        self.cropSize = 60
        self.imSize = 64
        self.numClasses = numClasses
        self.numChan = SCREEN_NUMCHANNELS

    def processWell(self, in_image, row, col, map_df=None, identifiers=None):
        if len(in_image) == 1:
            image = in_image[0]
            curImage = Image.open(image)
            if scale_factor != 1:
                curImage = enlarge_image(curImage, scale_factor)
            numFrames = curImage.n_frames
            G, R = load(curImage)
            G_arrays = convert(G)
            R_arrays = convert(R)
        else:
            numFrames = SCREEN_NUMCHANNELS
            image = in_image[0]
            image_ch2 = in_image[1]
            G = Image.open(image)
            R = Image.open(image_ch2)
            if scale_factor != 1:
                G = enlarge_image(G, scale_factor)
                R = enlarge_image(R, scale_factor)
            G_array = np.array(G)
            R_array = np.array(R)
            if SCREEN_NUMCHANNELS == 3:
                image_ch3 = in_image[2]
                GR = Image.open(image_ch3)
                if scale_factor != 1:
                    GR = enlarge_image(GR, scale_factor)
                GR_array = np.array(GR)

        MAX_CELLS = 1200
        croppedCells = np.zeros((MAX_CELLS, self.imSize ** 2 * SCREEN_NUMCHANNELS))
        coordUsed = np.zeros((MAX_CELLS, 2))
        imageUsed = np.chararray((MAX_CELLS, 2), itemsize=256)
        if identifiers is not None:
            identifierUsed = np.chararray((MAX_CELLS, len(identifiers)), itemsize=50)
        else:
            identifierUsed = np.empty((0, 0))
        ind = 0

        for frame in range(1, (numFrames/2) + 1):
            if len(in_image) == 1:
                G_array = G_arrays[frame - 1]
                R_array = R_arrays[frame - 1]
            # else:
            #     G_array = G_arrays
            #     R_array = R_arrays
            #     if SCREEN_NUMCHANNELS == 3:
            #         GR_array = GR_arrays
            curCoordinates = self.sql_data[(self.sql_data['Image'] == image) & (self.sql_data['Frame'] == frame)][
                ['Center_x', 'Center_y']]
            coord = 0
            while coord < len(curCoordinates):
                cur_x, cur_y = curCoordinates.values[coord]
                cur_x = int(cur_x)
                cur_y = int(cur_y)
                cur_y_left = int(np.floor(cur_y - self.imSize/2))
                cur_y_right = int(np.floor(cur_y + self.imSize/2))
                cur_x_up = int(np.floor(cur_x - self.imSize/2))
                cur_x_down = int(np.floor(cur_x + self.imSize/2))
                if cur_y - self.imSize/2 > 0 and cur_y + self.imSize/2 < G_array.shape[0] and cur_x - self.imSize/2 >\
                        0 and cur_x + self.imSize/2 < G_array.shape[1]:
                    croppedCells[ind, : self.imSize**2] = (G_array[cur_y_left:cur_y_right, cur_x_up:cur_x_down]).ravel()
                    if SCREEN_NUMCHANNELS == 2:
                        croppedCells[ind, self.imSize ** 2:] = (
                            R_array[cur_y_left:cur_y_right, cur_x_up:cur_x_down]).ravel()
                    else:
                        croppedCells[ind, self.imSize ** 2: self.imSize ** 2 * 2] = (
                            R_array[cur_y_left:cur_y_right, cur_x_up:cur_x_down]).ravel()
                        croppedCells[ind, self.imSize ** 2 * 2:] = (
                            GR_array[cur_y_left:cur_y_right, cur_x_up:cur_x_down]).ravel()
                    coordUsed[ind, :] = [cur_x, cur_y]
                    imageUsed[ind, :] = [image, frame]

                    ind += 1

                coord += 1
                if ind > (MAX_CELLS - 1):
                    break
            if ind > (MAX_CELLS - 1):
                break
        curCroppedCells = croppedCells[:ind]
        coordUsed = coordUsed[:ind]
        imageUsed = imageUsed[:ind]
        if identifiers is not None:
            identifierUsed = identifierUsed[:ind]

        stretchLow = 0.1  # stretch channels lower percentile
        stretchHigh = 99.9  # stretch channels upper percentile
        processedBatch = preProcessTestImages(curCroppedCells,
                                              self.imSize, self.cropSize, self.numChan,
                                              rescale=False, stretch=True,
                                              means=None, stds=None,
                                              stretchLow=stretchLow, stretchHigh=stretchHigh)
        del croppedCells
        del curCroppedCells

        return processedBatch, coordUsed, imageUsed, identifierUsed, ind



# noinspection PyTypeChecker
def getPredictions(xy_data):
    global tpUsed
    print('Getting predictions...')
    # Load networks #
    loc = tf.Graph()
    with loc.as_default():
        loc_saver = tf.train.import_meta_graph(model_ckpt + '.meta')
    locSession = tf.Session(graph=loc)
    loc_saver.restore(locSession, model_ckpt)

    pred_loc = loc.get_tensor_by_name('softmax:0')
    input_loc = loc.get_tensor_by_name('input:0')
    is_training_loc = loc.get_tensor_by_name('is_training:0')

    # Images to process #
    curScreenClass = ScreenClass(data=xy_data)
    images = curScreenClass.sql_data['Image'].unique() # images = xy_data['Image'].unique()
    sql_columns = list(curScreenClass.sql_data.columns)
    sql_numrows = curScreenClass.sql_data.shape[0]
    columns = sql_columns + ['Top Prediction'] + localizationTerms

    allPred_ind = 0
    allPred = pd.DataFrame(np.zeros((sql_numrows, len(columns))), columns=columns)

    wellList = []
    # Loop through each images #
    for image in images:
        print('currently in image: %s' %image)
        if img_type == 'phenix':
            if 'ch2' in image or 'ch3' in image:
                print '\tSkipping...'
                continue
            else:
                rc = rc_phenix.match(image)
                if rc:
                    row = int(rc.group(1))
                    col = int(rc.group(2))
                    tp = int(rc.group(3))
                else:
                    print 'ROW-COLUMN NOT FOUND', image
                    continue
            if (row, col, tp) not in wellList:
                wellList.append((row, col, tp))
            image_ch2 = image.replace('ch1', 'ch2')
            in_image = [image, image_ch2]
            if SCREEN_NUMCHANNELS == 3:
                image_ch3 = image.replace('ch1', 'ch3')
                in_image.append(image_ch3)
            processedBatch, coordUsed, imageUsed, identifierUsed, ind = curScreenClass.processWell(
                in_image, row, col, map_df=map_df, identifiers=identifiers)
        else:
            rc = rc_tiff.match(image)
            tp = None
            if rc:
                row = int(rc.group(1))
                col = int(rc.group(2))
            else:
                print 'ROW-COLUMN NOT FOUND', image
                continue
            if (row, col) not in wellList:
                wellList.append((row, col))
            processedBatch, coordUsed, imageUsed, identifierUsed, ind = curScreenClass.processWell(
                [image], row, col, map_df=map_df, identifiers=identifiers)


        predictedBatch_Loc = process_crops_loc(processedBatch=processedBatch, predicted_y=pred_loc,
                                               inputs=input_loc, is_training=is_training_loc,
                                               keep_prob=None, sess=locSession)
        (numrow, numcol) = predictedBatch_Loc.shape

        predCell = np.chararray((numrow, 1), itemsize=50)
        for i in range(numrow):
            predCell[i] = localizationTerms[np.argmax(predictedBatch_Loc[i, :])]
        allPred.iloc[allPred_ind:allPred_ind + len(predictedBatch_Loc), :] = np.hstack(
            (imageUsed, coordUsed, predCell, predictedBatch_Loc))

        allPred_ind += len(predictedBatch_Loc)

    if img_type == 'phenix' or img_type == 'opera1':
        allPred = allPred.drop(['Frame'], axis=1)

    # Save predictions file to CSV #
    allPred = allPred.iloc[:allPred_ind, :]
    write_csv(allPred, outfile_c, index=False)

    # Close network #
    locSession.close()

    return allPred

imagepath = []
frames = []
x_coord = []
y_coord = []

for i in range(cell_cnt):
    imageinfo = images[i]
    imagepath.append(imageinfo[0])
    frames.append(1)
    x_coord.append(imageinfo[5])
    y_coord.append(imageinfo[6])

df = pd.DataFrame(columns=['Image', 'Frame', 'Center_x', 'Center_y'])
df['Image'] = imagepath
df['Frame'] = frames
df['Center_x'] = x_coord
df['Center_y'] = y_coord

identifiers = None
map_df = None
prediction_df = getPredictions(df)
