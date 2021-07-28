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

from skimage.measure import regionprops
from segment_and_predict_lib import *
from skimage.io import imsave
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import fcntl
import glob
import copy
import sys
import os
import re


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
        if model == 'chong' or model == 'harsha':
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

    def __init__(self, screen, data):
        self.basePath = screen
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
            curImage = Image.open(os.path.join(self.basePath, image))
            numFrames = curImage.n_frames
            G, R = load(curImage)
            G_arrays = convert(G)
            R_arrays = convert(R)
        else:
            numFrames = SCREEN_NUMCHANNELS
            image = in_image[0]
            image_ch2 = in_image[1]
            G = Image.open(os.path.join(self.basePath, image))
            R = Image.open(os.path.join(self.basePath, image_ch2))
            G_array = np.array(G)
            R_array = np.array(R)
            if SCREEN_NUMCHANNELS == 3:
                image_ch3 = in_image[2]
                GR = Image.open(os.path.join(self.basePath, image_ch3))
                GR_array = np.array(GR)

        MAX_CELLS = 1200
        croppedCells = np.zeros((MAX_CELLS, self.imSize ** 2 * SCREEN_NUMCHANNELS))
        imageUsed = np.chararray((MAX_CELLS, 2), itemsize=50)
        coordUsed = np.zeros((MAX_CELLS, 2))
        abundanceUsed = np.zeros((MAX_CELLS, 1))

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
            curData = self.sql_data[(self.sql_data['Image'] == image) & (self.sql_data['Frame'] == frame)]
            curCoordinates = curData[['Center_x', 'Center_y']]
            curAbundances = curData[['Protein Abundance']]
            coord = 0

            while coord < len(curCoordinates):
                cur_x, cur_y = curCoordinates.values[coord]
                cur_abundance = curAbundances.values[coord]
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
                    abundanceUsed[ind, :] = [cur_abundance]
                    if identifiers is not None:
                        try:
                            identifierUsed[ind, :] = map_df[(map_df['Plate'] == int(plate)) & (map_df['Row'] == row) &
                                                        (map_df['Column'] == col)][identifiers]
                        except ValueError:
                            identifierUsed[ind, :] = map_df[(map_df['Plate'] == plate) & (map_df['Row'] == row) &
                                                        (map_df['Column'] == col)][identifiers]
                    ind += 1

                coord += 1
                if ind > (MAX_CELLS - 1):
                    break
            if ind > (MAX_CELLS - 1):
                break
        curCroppedCells = croppedCells[:ind]
        coordUsed = coordUsed[:ind]
        imageUsed = imageUsed[:ind]
        abundanceUsed = abundanceUsed[:ind]
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

        return processedBatch, coordUsed, imageUsed, abundanceUsed, identifierUsed, ind


def getWellDF(localizationTerms, rows, perWell, perWell_ind, wellstr):
    abundance_avg = round(rows['Protein Abundance'].sum()/rows.shape[0], 2)
    allAve = np.zeros(len(localizationTerms))
    for i, label in enumerate(localizationTerms):
        predAve = rows[label].sum()/rows.shape[0]
        allAve[i] = predAve
    # predMax = localizationTerms[np.argmax(allAve)]

    avgtop = []
    avgscore = sorted(allAve, reverse=True)[:3]
    for avg in avgscore:
        avgtop.append(localizationTerms[allAve.tolist().index(avg)])
    avgtop = ' - '.join(avgtop)

    to_drop = localizationTerms + ['Center_x', 'Center_y', 'Protein Abundance', 'Top Prediction']
    if img_type == 'opera4':
        to_drop += ['Frame']
    rows = rows.drop(to_drop, axis=1)
    cols_left = rows.columns
    to_stack = []
    for col in cols_left:
        if col == 'Top Prediction':
            continue
        elif col == 'Image':
            to_stack.append(str(wellstr))
        else:
            to_stack.append(rows[col].iloc[0])

    to_stack.extend([rows.shape[0], abundance_avg, avgtop, allAve])
    perWell.iloc[perWell_ind, :] = np.hstack(tuple(to_stack))
    perWell_ind += 1
    return perWell, perWell_ind


def getWell(allPred, sql_columns, sql_numrows, identifiers, localizationTerms, wellList):
    if identifiers is not None:
        columns_w = [sql_columns[0]] + identifiers + ['NumCells'] + ['Protein Abundance'] + ['Top Prediction'] + localizationTerms
    else:
        columns_w = [sql_columns[0]] + ['NumCells'] + ['Protein Abundance'] + ['Top Prediction'] + localizationTerms
    if timepoint == 'True':
        columns_w.insert(0, 'Timepoint')

    perWell = pd.DataFrame(np.zeros((sql_numrows, len(columns_w))), columns=columns_w)
    perWell_ind = 0
    for i in range(len(wellList)):
        if img_type == 'phenix':
            (row, col, tp) = wellList[i]
            wellstr = 'r' + str(row).zfill(2) + 'c' + str(col).zfill(2)
            if timepoint == 'True':
                rows = allPred[(allPred['Image'].str.contains(wellstr, na=False)) & (allPred['Timepoint'] == str(tp))]
            else:
                rows = allPred[allPred['Image'].str.contains(wellstr, na=False)]
            rows = rows.convert_objects(convert_numeric=True)
            if rows.shape[0] == 0:
                continue
            perWell, perWell_ind = getWellDF(localizationTerms, rows, perWell, perWell_ind, wellstr)
        else:
            (row, col) = wellList[i]
            wellstr = str(row).zfill(3) + str(col).zfill(3)
            wellname = 'r' + str(row).zfill(2) + 'c' + str(col).zfill(2)
            rows = allPred[allPred['Image'].str.startswith(wellstr, na=False)]
            rows = rows.convert_objects(convert_numeric=True)
            if rows.shape[0] == 0:
                continue
            perWell, perWell_ind = getWellDF(localizationTerms, rows, perWell, perWell_ind, wellname)

    perWell = perWell[:perWell_ind]
    perWell = perWell.rename(columns={'Image': 'Well'})
    return perWell


# noinspection PyTypeChecker
def getPredictions(xy_data, outdir, outfile, mainrow, identifiers=None, map_df=None):
    global tpUsed
    print 'Getting predictions...'
    # Load networks #
    loc = tf.Graph()
    with loc.as_default():
        loc_saver = tf.train.import_meta_graph(model_ckpt + '.meta')
    locSession = tf.Session(graph=loc)
    loc_saver.restore(locSession, model_ckpt)

    if model == 'chong' or model == 'harsha':
        pred_loc = loc.get_tensor_by_name('softmax:0')
        input_loc = loc.get_tensor_by_name('input:0')
        is_training_loc = loc.get_tensor_by_name('is_training:0')
    else:
        logits = loc.get_tensor_by_name('final_layer/batch_norm/final_layer_batch_norm/batchnorm/add_1:0')
        pred_loc = tf.nn.softmax(logits)
        keep_prob = loc.get_tensor_by_name('Placeholder:0')
        input_loc = loc.get_tensor_by_name('input:0')
        is_training_loc = loc.get_tensor_by_name('is_training:0')


    # Images to process #
    curScreenClass = ScreenClass(screen=input_images, data=xy_data)
    images = curScreenClass.sql_data['Image'].unique() # images = xy_data['Image'].unique()
    sql_columns = list(curScreenClass.sql_data.columns)
    sql_numrows = curScreenClass.sql_data.shape[0]

    if identifiers is not None:
        columns = sql_columns + identifiers + ['Top Prediction'] + localizationTerms
    else:
        columns = sql_columns + ['Top Prediction'] + localizationTerms

    outfile_c = os.path.join(outdir, outfile.replace('.csv', '_perCell_%s.csv' % mainrow))
    outfile_w = os.path.join(outdir, outfile.replace('.csv', '_perWell.csv'))
    if timepoint == 'True':
        columns.insert(0, 'Timepoint')
    allPred_ind = 0
    allPred = pd.DataFrame(np.zeros((sql_numrows, len(columns))), columns=columns)

    wellList = []
    # Loop through each images #
    for image in images:
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
            processedBatch, coordUsed, imageUsed, abundanceUsed, identifierUsed, ind = curScreenClass.processWell(
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
            processedBatch, coordUsed, imageUsed, abundanceUsed, identifierUsed, ind = curScreenClass.processWell(
                [image], row, col, map_df=map_df, identifiers=identifiers)

        if tp and timepoint == 'True':
            tp_array = np.repeat(tp, ind)
            tpUsed = np.ndarray(shape=(ind, 1), buffer=tp_array, dtype=int)

        if model == 'chong' or model == 'harsha':
            predictedBatch_Loc = process_crops_loc(processedBatch=processedBatch, predicted_y=pred_loc,
                                                   inputs=input_loc, is_training=is_training_loc,
                                                   keep_prob=None, sess=locSession)
        else:
            predictedBatch_Loc = process_crops_loc(processedBatch=processedBatch, predicted_y=pred_loc,
                                                   inputs=input_loc, is_training=is_training_loc,
                                                   keep_prob=keep_prob, sess=locSession)
        (numrow, numcol) = predictedBatch_Loc.shape

        del processedBatch

        predCell = np.chararray((numrow, 1), itemsize=50)
        for i in range(numrow):
            predCell[i] = localizationTerms[np.argmax(predictedBatch_Loc[i, :])]
        if tp and timepoint == 'True':
            if identifierUsed.shape != (0, 0):
                allPred.iloc[allPred_ind:allPred_ind + len(predictedBatch_Loc), :] = np.hstack(
                    (tpUsed, imageUsed, coordUsed, abundanceUsed, identifierUsed, predCell, predictedBatch_Loc))
                del identifierUsed
            else:
                allPred.iloc[allPred_ind:allPred_ind + len(predictedBatch_Loc), :] = np.hstack(
                    (tpUsed, imageUsed, coordUsed, abundanceUsed, predCell, predictedBatch_Loc))
            del tpUsed
        else:
            if identifierUsed.shape != (0, 0):
                allPred.iloc[allPred_ind:allPred_ind + len(predictedBatch_Loc), :] = np.hstack(
                    (imageUsed, coordUsed, abundanceUsed, identifierUsed, predCell, predictedBatch_Loc))
                del identifierUsed
            else:
                allPred.iloc[allPred_ind:allPred_ind + len(predictedBatch_Loc), :] = np.hstack(
                    (imageUsed, coordUsed, abundanceUsed, predCell, predictedBatch_Loc))
        allPred_ind += len(predictedBatch_Loc)

        del imageUsed
        del coordUsed
        del abundanceUsed
        del predCell
        del predictedBatch_Loc

    del xy_data

    if img_type == 'phenix' or img_type == 'opera1':
        allPred = allPred.drop(['Frame'], axis=1)

    # Save predictions file to CSV #
    allPred = allPred.iloc[:allPred_ind, :]
    if opt_save == "cell":
        write_csv(allPred, outfile_c, index=False)
    elif opt_save == "well":
        perWell = getWell(allPred, sql_columns, sql_numrows, identifiers, localizationTerms, wellList)
        write_csv(perWell, outfile_w, mode='a', index=False, header=not(os.path.isfile(outfile_w)))
    else:
        write_csv(allPred, outfile_c, index=False)
        perWell = getWell(allPred, sql_columns, sql_numrows, identifiers, localizationTerms, wellList)
        write_csv(perWell, outfile_w, mode='a', index=False, header=not(os.path.isfile(outfile_w)))

    del perWell
    del allPred
    perWell_csv = pd.read_csv(outfile_w)
    sorted_perWell = perWell_csv.sort_values(by=['Well'])
    write_csv(sorted_perWell, outfile_w, index=False)

    # Close network #
    locSession.close()


def getCoordinates(row_images, farred, SCREEN_NUMCHANNELS):
    image_names = []
    frame_number = []
    x_coord = []
    y_coord = []
    intensity = []

    for img in row_images:
        # Split channels to pixel arrays and images
        if ('.tiff' in img) or ('.flex' in img) or ('.tif' in img):  # Check if the image has a .flex or .tiff extension
            numframes = Image.open(os.path.join(input_images, img)).n_frames
            if numframes == 1:  # phenix image (1field x 1channel)
                if 'ch1' in img:
                    img_ch2 = img.replace('ch1', 'ch2')
                    image_green = Image.open(os.path.join(input_images, img))
                    image_red = Image.open(os.path.join(input_images, img_ch2))
                    green = np.asarray(image_green)
                    red = np.asarray(image_red)
                    if (farred == 'True') and (SCREEN_NUMCHANNELS == 3):
                        img_ch3 = img.replace('ch1', 'ch3')
                        image_farred = Image.open(os.path.join(input_images, img_ch3))
                        farred_array = np.asarray(image_farred)
                else:
                    continue
            else:
                green, red = split_channels(os.path.join(input_images, img))
                image_green, image_red = split_images(os.path.join(input_images, img))

            if opt_blur=='True':
                # Preprocess image by blurring with kernel size = 13
                # preprocessed_image = rescale_channel(green, flex=(img_type == 'opera1' or img_type == 'phenix'))
                if (farred == 'True') and (SCREEN_NUMCHANNELS == 3):
                    preprocessed_image = rescale_channel(farred_array, flex=(img_type=='opera1' or img_type=='phenix'))
                else:
                    preprocessed_image = rescale_channel(red, flex=(img_type == 'opera1' or img_type == 'phenix'))


            if img_type == 'opera1' or img_type == 'phenix':
                print('On image %s' % img)
                labeled_path = os.path.join(outdir, "%s_labeled.png" % img.split('.')[0])

                # Size of full image
                width, height = image_red.size

                # Run mixture model to find foreground, background and middleground pixel clusters
                if opt_blur == 'True':
                    image = preprocessed_image
                else:
                    if (farred == 'True') and (SCREEN_NUMCHANNELS==3):
                        print("using farred channel to segment %s" % farred)
                        image = farred_array
                    else:
                        image = red

                image = image.astype('uint16')
                labeled = run_segmentation(image)
                # Save labeled image
                imsave(labeled_path, labeled.astype(np.int16))
                measurements = regionprops(labeled, intensity_image=green)

                # Save image array
                s = 32  # cell dimension is 64x64
                for i in range(len(measurements)):
                    centroid = measurements[i].centroid
                    center_x = int(centroid[1])
                    center_y = int(centroid[0])
                    loc_left = center_x - s
                    loc_upper = center_y - s
                    loc_right = center_x + s
                    loc_lower = center_y + s

                    if not_on_border(width, height, loc_left, loc_upper, loc_right, loc_lower):
                        mean_intensity = round(measurements[i].mean_intensity, 2)
                        image_names.append(img)
                        frame_number.append(1)
                        x_coord.append(center_x)
                        y_coord.append(center_y)
                        intensity.append(mean_intensity)

            else:
                num_frames = len(green)
                # Run segmentation on each frame
                for frame in range(num_frames):
                    print('On image %s frame %d' % (img, (frame + 1)))
                    labeled_path = os.path.join(outdir, "%s_%d_labeled.png" % (img.split('.')[0], frame+1))

                    # Size of full image
                    width, height = image_red[frame].size

                    # Run mixture model to find foreground, background and middleground pixel clusters
                    image = red[frame]
                    if opt_blur == 'True':
                        image = preprocessed_image[frame]

                    image = image.astype('uint16')
                    labeled = run_segmentation(image)
                    # Save labeled image
                    imsave(labeled_path, labeled.astype(np.int16))
                    measurements = regionprops(labeled, intensity_image=green)

                    # Save image array
                    s = 32  # cell dimension is 64x64
                    for i in range(len(measurements)):
                        centroid = measurements[i].centroid
                        center_x = centroid[1]
                        center_y = centroid[0]
                        loc_left = center_x - s
                        loc_upper = center_y - s
                        loc_right = center_x + s
                        loc_lower = center_y + s

                        if not_on_border(width, height, loc_left, loc_upper, loc_right, loc_lower):
                            mean_intensity = round(measurements[i].mean_intensity, 2)
                            image_names.append(img)
                            frame_number.append(frame + 1)
                            x_coord.append(center_x)
                            y_coord.append(center_y)
                            intensity.append(mean_intensity)
        else:
            print('Skipping file %s' % img)

    df = pd.DataFrame(columns=['Image', 'Frame', 'Center_x', 'Center_y', 'Protein Abundance'])
    df['Image'] = image_names
    df['Frame'] = frame_number
    df['Center_x'] = x_coord
    df['Center_y'] = y_coord
    df['Protein Abundance'] = intensity
    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Segment and evaluate predictions of images using DeepLoc model')
    parser.add_argument("-m", "--model", action="store", dest="model", default='harsha',
                        help="Choose model to use: 'chong' or 'wt2017' or 'harsha'. Default is harsha.")
    parser.add_argument("-i", "--images", action="store", dest="input_images",
                        default='/home/morphology/shared/morphology/deeploc_cluster/example/images/',
                        help="Path to input images")
    parser.add_argument("-r", "--row", action="store", dest="row", help="Current row to process (i.e. r01)")
    parser.add_argument("-o", "--output", action="store", dest="output_dir",
                        default="/home/morphology/shared/morphology/deeploc_cluster/example/",
                        help="Path where to store output file")
    parser.add_argument("-n", "--outfile", action="store", dest="output_file", default="Images_predictions.csv",
                        help="Specify output filename prefix. Default is Image_predictions")
    parser.add_argument("-f", "--mapsheet_file", action="store", dest="mapsheet", default='None',
                        help="Path to mapping sheet")
    parser.add_argument("-p", "--plate", action="store", dest="plate", default="1",
                        help="Indicate plate number when processing multi-plate screens and mapsheet is specified")
    parser.add_argument("-t", "--type", action="store", dest="imagetype", default="phenix",
                        help="Indicate image type. Options: opera4 (4field x 2channel), opera1 (1field x 2 channel),"
                             "phenix (1field x 1channel). Default is phenix")
    parser.add_argument("-x", "--timepoint", action="store", dest="timepoint", default="False",
                        help="Use this flag if the input folder contains images with multiple timepoints. Default is False."
                             "Can ONLY be True when image type is phenix.")
    parser.add_argument("-y", "--farred", action="store_true", dest="farred",
                        help="Use farred channel (3rd channel) for segmentation instead of red. Can ONLY be used when image type is phenix.")
    parser.add_argument("-s", "--save", action="store", dest="save", default="both",
                        help="Choose option to save predictions: 'well', 'cell' or 'both'. Default is 'both'")
    parser.add_argument("-b", "--blur", action="store", dest='blur', default="False",
                        help="Use this flag if you want to use the blur function for segmenting the input images. Default is False.")

    args = parser.parse_args()

    # Define global variables
    global model, input_images, outdir, plate, img_type, timepoint, opt_save, opt_blur, localizationTerms, numClasses
    global rc_phenix, rc_tiff, model_ckpt, localizations, SCREEN_NUMCHANNELS

    model = args.model.lower()
    input_images = args.input_images
    mainrow = args.row
    outdir = args.output_dir
    outfile = args.output_file
    mapsheet = args.mapsheet
    plate = args.plate
    img_type = args.imagetype
    timepoint = args.timepoint
    farred = args.farred
    opt_save = args.save
    opt_blur = args.blur

    rc_phenix = re.compile('r([0-9]{2})c([0-9]{2})f.+-ch[0-9]sk([0-9])fk.+')
    rc_tiff = re.compile('([0-9]{3})([0-9]{3})[0-9]{3}')

    if model=='chong':
        model_ckpt = '/path/to/DUB_Biology/pretrained_models/chong_model.ckpt-5000'
        localizations = '/path/to/DUB_Biology/localizations_chong.txt'
        SCREEN_NUMCHANNELS = 2
    elif model=='wt2017':
        model_ckpt = '/path/to/DUB_Biology/pretrained_models/wt2017_model.ckpt-9500'
        localizations = '/path/to/DUB_Biology/localizations_wt2017.txt'
        SCREEN_NUMCHANNELS = 3
    elif model=='harsha':
        model_ckpt = '/path/to/DUB_Biology/pretrained_models/harsha_model.ckpt-8000'
        localizations = '/path/to/DUB_Biology/localizations_harsha.txt'
        SCREEN_NUMCHANNELS = 3

    # Start processing row images
    row_images = []
    for image in glob.glob(os.path.join(input_images, mainrow) + '*'):
        row_images.append(os.path.basename(image))
    row_images = sorted(row_images)

    localizationTerms = []
    with open(localizations, 'r') as labels:
        lines = (line.rstrip() for line in labels)
        lines = (line for line in lines if line)
        for line in lines:
            localizationTerms.append(line)
    numClasses = len(localizationTerms)

    if mapsheet != 'None':
        identifiers, map_df = get_identifiers(mapsheet)
    else:
        identifiers = None
        map_df = None

    dataframe = getCoordinates(row_images, farred, SCREEN_NUMCHANNELS)
    getPredictions(dataframe, outdir, outfile, mainrow, identifiers, map_df)

    print('Done processing!')


if __name__ == '__main__':
    main()
