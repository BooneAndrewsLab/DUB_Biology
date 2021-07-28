from segmentation import mixture_model as cmm
from sympy.functions.special.gamma_functions import digamma
from sympy import nsolve, symbols, functions
from PIL import Image, ImageEnhance
import scipy.ndimage as nd
from copy import deepcopy
import mahotas as mh
import numpy as np
import cv2


###########################################################			
### Load_GR functions					###
### Copyright (c) 2017, Oren Kraus All rights reserved. ###
###########################################################

class ImageSequence:
    def __init__(self, im):
        self.im = im
    def __getitem__(self, ix):
        try:
            if ix:
                self.im.seek(ix)
            return self.im
        except EOFError:
            raise IndexError # end of sequence
            
def load(im):
    #im.seek(0)
    Red_in=[]
    Green_in=[]
    count=0
    for frame in ImageSequence(im):
        if count%2==0:
            Green_in.append(frame.copy())
        else:
            Red_in.append(frame.copy())
        count+=1
    return Green_in,Red_in
           
def convert(images):
    return [np.array(image) for image in images]
    
    
###########################################################			
### preprocess_images functions 			###
### Copyright (c) 2017, Oren Kraus All rights reserved. ###
###########################################################


def preProcessImages(batchData,imSize,cropSize,channels,rescale=True,stretch=False,
                     means=None,stds=None,stretchLow=None,stretchHigh=None,jitter=True,randTransform=True):
    
    if rescale:
        batchData = rescaleBatch(batchData,means,stds,imSize,channels)
    if stretch:
        batchData = stretchBatch(batchData, stretchLow, stretchHigh, imSize, channels)

    tensorBatchData = flatBatch2Tensor(batchData, imSize, channels)
    if jitter:
        tensorBatchData = jitterBatch(tensorBatchData,cropSize,imSize)
    if randTransform:
        tensorBatchData = randTransformBatch(tensorBatchData)
    return tensorBatchData

def preProcessTestImages(batchData,imSize,cropSize,channels,rescale=True,stretch=False,
                     means=None,stds=None,stretchLow=None,stretchHigh=None):

    if rescale:
        batchData = rescaleBatch(batchData,means,stds,imSize,channels)
    if stretch:
        batchData = stretchBatch(batchData, stretchLow, stretchHigh, imSize, channels)
    tensorBatchData = flatBatch2Tensor(batchData,imSize,channels)
    tensorBatchData = extractCrops(tensorBatchData,cropSize,imSize)

    return tensorBatchData

def flatBatch2Tensor(batchData,imSize,channels):
    splitByChannel = [batchData[:,(chan*imSize**2):((chan+1)*imSize**2)].reshape((-1,imSize,imSize,1)) \
                      for chan in range(channels)]
    tensorBatchData = np.concatenate(splitByChannel,3)
    
    return tensorBatchData


def rescaleBatch(batchData,means,stds, imSize, channels):
    for chan in range(channels):
        batchData[:,(chan*imSize**2):((chan+1)*imSize**2)] = \
            (batchData[:,(chan*imSize**2):((chan+1)*imSize**2)] - means[chan]) / stds[chan]
    return batchData


def stretchBatch(batchData, lowerPercentile, upperPercentile, imSize, channels):
    for chan in range(channels):
        for i in range(len(batchData)):
            batchData[i, (chan * imSize ** 2):((chan + 1) * imSize ** 2)] = \
                stretchVector(batchData[i, (chan * imSize ** 2):((chan + 1) * imSize ** 2)],
                              lowerPercentile, upperPercentile)
    return batchData

def stretchVector(vec, lowerPercentile, upperPercentile):
    minVal = np.percentile(vec, lowerPercentile)
    maxVal = np.percentile(vec, upperPercentile)
    vec[vec > maxVal] = maxVal
    vec = vec - minVal
    if (maxVal-minVal)>1.:
        vec = vec / (maxVal - minVal)

    return vec

def jitterBatch(batchData,cropSize,imSize):
    batchSize,x,y,channels = batchData.shape
    croppedBatch = np.zeros((batchSize,cropSize,cropSize,channels),dtype=batchData.dtype)
    jitterPix = imSize-cropSize
    for i in range(batchSize):
        offset = np.random.randint(0,jitterPix,2)
        croppedBatch[i,:,:,:] = batchData[i,offset[0]:x-(jitterPix-offset[0]),
                                            offset[1]:y-(jitterPix-offset[1]),:]  
    return croppedBatch

def extractCrops(batchData,cropSize,imSize):
    batchSize,x,y,channels = batchData.shape
    crops = 5
    croppedBatch = np.zeros((batchSize,crops,cropSize,cropSize,channels),dtype=batchData.dtype)
    jitterPix = imSize-cropSize


    for i in range(batchSize):

     #center crop
        offset = [jitterPix/2,jitterPix/2]
        croppedBatch[i,0,:,:,:] = batchData[i,offset[0]:x-(jitterPix-offset[0]),
                                            offset[1]:y-(jitterPix-offset[1]),:]
    #left top crop
        offset = [0,0]
    #for i in range(batchSize):
        croppedBatch[i,1,:,:,:] = batchData[i,offset[0]:x-(jitterPix-offset[0]),
                                            offset[1]:y-(jitterPix-offset[1]),:]
    #left bottom crop
        offset = [0,jitterPix]
    #for i in range(batchSize):
        croppedBatch[i,2,:,:,:] = batchData[i,offset[0]:x-(jitterPix-offset[0]),
                                            offset[1]:y-(jitterPix-offset[1]),:]
    #right top crop
        offset = [jitterPix,0]
    #for i in range(batchSize):
        croppedBatch[i,3,:,:,:] = batchData[i,offset[0]:x-(jitterPix-offset[0]),
                                            offset[1]:y-(jitterPix-offset[1]),:]
    #right bottom crop
        offset = [jitterPix,jitterPix]
    #for i in range(batchSize):
        croppedBatch[i,4,:,:,:] = batchData[i,offset[0]:x-(jitterPix-offset[0]),
                                            offset[1]:y-(jitterPix-offset[1]),:]
    return croppedBatch


def randTransformBatch(croppedBatchData):
    for i in range(len(croppedBatchData)):
        if np.random.choice([True, False]):
            croppedBatchData[i,:,:,:] = np.flipud(croppedBatchData[i,:,:,:])
        if np.random.choice([True, False]):
            croppedBatchData[i,:,:,:] = np.fliplr(croppedBatchData[i,:,:,:])
        croppedBatchData[i,:,:,:] = np.rot90(croppedBatchData[i,:,:,:], k=np.random.randint(0,3))   
    return croppedBatchData



# Enlarge image for Phenix images #
def enlarge_image(image, scale_factor):
    im_arr = np.array(image.copy()).astype(float)
    im_scale = 1 / im_arr.max()
    im_new = ((im_arr * im_scale) * 255).round().astype(np.uint8)
    image = Image.fromarray(im_new)
    w, h = image.size
    w = int(w*scale_factor)
    h = int(h*scale_factor)
    img_resized = image.resize((w, h), Image.LANCZOS)
    return img_resized


###################################################################		
###    	Segmentation functions 					###
### 	Edited by Nil Sahin on December 2017			###
###	Adapted from the link by Anastasiia Razdaibiedina	###
###	http://mahotas.readthedocs.io/en/latest/labeled.html 	###
###################################################################

def split_channels(path):
    # Split Red and Green channels from .flex and .tiff files
    # Return channel for .flex files
    # Return lists of channels in separate frames for .tiff files
    # .flex files contain one frame per channel
    # .tiff files contain four frams per channel

    im = Image.open(path)
    red = im.copy()
    green = im.copy()

    if 'flex' in path:
        for im_number in range(im.n_frames):
            im.seek(im_number)
            if im_number%2:
                red = np.asarray(im.copy())
            else:
                green = np.asarray(im.copy())

    else:
        red = []
        green = []
        for im_number in range(im.n_frames):
            im.seek(im_number)
            if im_number%2:
                red.append(np.asarray(im.copy()))
            else:
                green.append(np.asarray(im.copy()))

    return green, red


def split_images(path):
    # Split Red and Green images from .flex and .tiff files
    # Return image for .flex files
    # Return lists of images in separate frames for .tiff files
    # .flex files contain one frame per channel
    # .tiff files contain four frams per channel

    im = Image.open(path)
    red = im.copy()
    green = im.copy()

    if 'flex' in path:
        for im_number in range(im.n_frames):
            im.seek(im_number)
            if im_number % 2:
                red = im.copy().convert('L')
            else:
                green = im.copy().convert('L')
    else:
        red = []
        green = []
        for im_number in range(im.n_frames):
            im.seek(im_number)
            if im_number%2:
                red.append(im.copy().convert('L'))
            else:
                green.append(im.copy().convert('L'))

    return green, red


def blur_channel(ch):

    return cv2.GaussianBlur(ch, (13, 13), 0)


def rescale_channel(channel, flex=False):
    if flex:
        return blur_channel(channel.copy())

    else:
        rescaled = []
        for i in range(len(channel)):
            rescaled.append(blur_channel(channel[i].copy()))

        return rescaled


def normalize_channel(ch):
    minn = np.min(ch)
    maxx = np.max(ch)

    return (ch - minn) / (maxx - minn)


def even_background_channel(ch):
    mean = np.mean(ch)
    for m in range(ch.shape[0]):
        for n in range(ch.shape[1]):
            if ch[m][n] < mean:
                ch[m][n] = 0
    return ch


def contrast_image(im):
    new_image = []
    for i in im:
        new_image.append(np.asarray(ImageEnhance.Contrast(i).enhance(2)))

    return new_image


def run_segmentation(image):
    seg, _ = cmm(image)
    #seg, prob = mixture_model(image)

    # Watershed segmentation
    watershedSeg = Watershed_MRF(image, seg)

    # Threshold by size
    labeled = watershedSeg.copy()
    sizes = mh.labeled.labeled_size(labeled)
    labeled = mh.labeled.remove_regions(labeled, np.where((sizes < 300) | (sizes > 2600)))
    labeled, c = mh.labeled.relabel(labeled)

    return labeled


def mixture_model(im_original, verbose=False):

    # ------------------------------------------------------------------------------------ #
    #                                                                                      #
    # This algorithm is from: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6298972  #
    # A Nonsymmetric Mixture Model for Unsupervised Image Segmentation                     #
    # Thanh Minh Nguyen and Q. M. Jonathan Wu                                              #
    # IEEE April 2013                                                                      #
    #                                                                                      #
    # This algorithm is implemented by Oren Kraus on July 2013                             #
    #                                                                                      #
    # ------------------------------------------------------------------------------------ #
    
    # INITIALIZE PARAMETERS

    # Expected nucleus and cell size
    meanNucSize = 210
    meanCellSize = 1900

    # Initialize means
    U = [[0, 500.0], [17500, 22500], [50000]]
    # Initialize variance
    var = 10000.0 ** 2
    S = [[var, var], [var, var], [var]]
    # Initial degree of freedom for t-distribution
    V = [[1, 1], [100, 100], [100]]
    
    # Number of components
    K = 3
    # t-distributions per component
    Kj = [2,2,1]
    # Neighborhood for averaging
    Nh = 7
    # Maximum iterations
    max_iterations = 100
    # Image shape
    M, N = im_original.shape
    # Mixing proportions for t-distributions
    Eta=[]
    for k in range(K):
        Eta.append((1.0 / Kj[k]) * np.ones(Kj[k]))
    
    # Learning rate
    learning_rate = 10 ** -6
    # Unknown parameters
    B = 12
    Bold = 0
    Vsym = symbols('Vsym')


    # ESTIMATE INITIAL MIXING PROPORTIONS

    # Number of nuclei
    nucNum = nd.measurements.label(im_original>110)[1]
    # Expected nucleus and cell area
    NucArea = nucNum * meanNucSize / (M * N)
    CytArea = nucNum * meanCellSize / (M * N)
    BackArea = 1 - CytArea
    if BackArea < 0.1:
        BackArea = 0.1
        CytArea = 1 - BackArea - NucArea
    if NucArea < 0.1:
        NucArea = 0.1
        CytArea = 0.2
        BackArea = 1 - NucArea - CytArea
    area = [BackArea, CytArea, NucArea]
    if verbose:
        print('Percent area - Background %.1f - Cytoplasm %.1f - Nucleus %.1f' % (area[0], area[1], area[2]))

    im_ad = (np.double(im_original) * 2 ** 16 / im_original.max()).round()


    # NON-SYMMETRIC MIXTURE MODEL

    # Mixing proportions per pixel
    MP = np.ones((M, N, K))
    # Posterior probability
    Z = np.zeros((M, N, K))
    AveLocZ = np.zeros((M, N, K))
    # Initialize parameters
    Temp0 = np.zeros((M, N, K))
    Temp2 = np.zeros((M, N, K))
    LogLike = np.zeros((max_iterations))
    Y = []
    u = []
    Temp1 = []
    StudPDFVal = []
    for k in range(K):
        MP[:, :, k] = area[k]
        Y.append(np.zeros((M, N, Kj[k])))
        u.append(np.zeros((M, N, Kj[k])))
        Temp1.append(np.zeros((M, N, Kj[k])))
        StudPDFVal.append(np.zeros((M, N, Kj[k])))

    # Run mixture model
    for iters in range(max_iterations):

        # E-step
        for k in range(K):
            temp = np.zeros((M, N))
            for m in range(Kj[k]):
                StudPDFVal[k][:,:,m] = StudPDF(im_ad, U[k][m], S[k][m], V[k][m])
                temp += Eta[k][m] * StudPDFVal[k][:,:,m]
                Y[k][:,:,m] = Eta[k][m] * StudPDFVal[k][:,:,m]
                u[k][:,:,m] = (V[k][m] + 1) / (V[k][m] + (im_ad - U[k][m]) ** 2 / S[k][m])
            Z[:,:,k] = MP[:,:,k] * temp
            sumYk = Y[k].sum(axis=2)

            for m2 in range(Kj[k]):
                Y[k][:,:,m2] = Y[k][:,:,m2] / sumYk

        sumZ = Z.sum(axis=2)
        for k in range(K):
            Z[:,:,k] = Z[:,:,k]/sumZ
        
        # M-step
        for k in range(K):
            for m in range(Kj[k]):
                U[k][m] = (Z[:,:,k] * Y[k][:,:,m] * u[k][:,:,m] * im_ad).sum() /\
                          (Z[:,:,k] * Y[k][:,:,m] * u[k][:,:,m]).sum()
                try:
                    V[k][m] = np.fabs(float(nsolve(-digamma(Vsym / 2) + functions.log(Vsym / 2) + 1 +\
                                                   ((Z[:,:,k] * Y[k][:,:,m] * (np.log(u[k][:,:,m]) - u[k][:,:,m]))
                                                    .sum() / (Z[:,:,k] * Y[k][:,:,m]).sum()) +\
                                                   digamma((V[k][m] + 1) / 2) - np.log((V[k][m] + 1) / 2),
                                                   Vsym, V[k][m]).real))
                except:
                    V[k][m] = V[k][m]

                Eta[k][m] = (Z[:,:,k] * Y[k][:,:,m]).sum() / (Z[:,:,k] * Y[k].sum(axis=2)).sum()
            nd.uniform_filter(Z[:,:,k], size=Nh, output=MP[:,:,k], mode='constant')
            MP[:,:,k] = np.exp(B * MP[:,:,k])
            nd.uniform_filter(Z[:,:,k], size=Nh, output=AveLocZ[:,:,k], mode='constant')
        
        sumMP = MP.sum(axis=2)
        for k in range(K):
            MP[:,:,k] = MP[:,:,k] / sumMP
            for m in range(Kj[k]):
                if S[k][m] > 500:
                    S[k][m] = (Z[:,:,k] * Y[k][:,:,m] * u[k][:,:,m] * (im_ad-U[k][m]) ** 2).sum() /\
                              (Z[:,:,k] * Y[k][:,:,m]).sum()
                else:
                    S[k][m] = 500
        
        # Update components
        Utemp = deepcopy(U)
        Stemp = deepcopy(S)
        Vtemp = deepcopy(V)

        if max(U[0]) > min(U[1]):
            indSwitch0 = np.argmax(U[0])
            indSwitch1 = np.argmin(U[1])
            U[0][indSwitch0] = Utemp[1][indSwitch1]
            S[0][indSwitch0] = Stemp[1][indSwitch1]
            V[0][indSwitch0] = Vtemp[1][indSwitch1]
            U[1][indSwitch1] = Utemp[0][indSwitch0]
            S[1][indSwitch1] = Stemp[0][indSwitch0]
            V[1][indSwitch1] = Vtemp[0][indSwitch0]
            Utemp = deepcopy(U)
            Stemp = deepcopy(S)
            Vtemp = deepcopy(V)
        
        if max(U[2]) < max(U[1]):
            indSwitch = np.argmax(U[1])
            U[2] = [Utemp[1][indSwitch]]
            S[2] = [Stemp[1][indSwitch]]
            V[2] = [Vtemp[1][indSwitch]]
            U[1][indSwitch] = Utemp[2][0]
            S[1][indSwitch] = Stemp[2][0]
            V[1][indSwitch] = Vtemp[2][0]
        
        while np.fabs(B - Bold) > 0.05:
            Bold = B
            expAveLocZ = np.exp(B * AveLocZ)
            SumExpLocZ = (AveLocZ * expAveLocZ).sum(axis=2) / expAveLocZ.sum(axis=2)
            for k in range(K):
                Temp0[:,:,k] = AveLocZ[:,:,k] - SumExpLocZ
            Bnew = Bold - learning_rate * (-((Z*Temp0).sum(axis=2)).sum())
            B = Bnew

        for k in range(K):
            for m in range(Kj[k]):
                Temp1[k][:,:,m] = Eta[k][m] * StudPDFVal[k][:,:,m]
            Temp2[:,:,k] = MP[:,:,k] * Temp1[k].sum(axis=2)
        LogLike[iters] = np.log(Temp2.sum(axis=2)).sum()
        
        if iters > 0:
            if verbose:
                print('Iterations = ', iters,
                      ' LogLikelihood = ', LogLike[iters],
                      ' DiffLikelihood = ', np.fabs(LogLike[iters-1]-LogLike[iters]))

            # Threshold for early stopping
            Ptot = LogLike[iters]
            if np.fabs(LogLike[iters - 1] - LogLike[iters]) < 3000:
                break
        
        im_output = np.argmax(Z, axis=2)

    return np.uint8(im_output), Ptot
    
    
def StudPDF(X,U,Covar,Dof):

    # ------------------------------------------------------------------------------------ #
    #                                                                                      #
    # This algorithm is implemented by Oren Kraus on July 2013                             #
    #                                                                                      #
    # ------------------------------------------------------------------------------------ #

    return (np.math.gamma(Dof/2.0+1/2.0)*(Covar)**(-1/2.0))/\
            (np.sqrt(Dof*np.pi)*np.math.gamma(Dof/2.0))/(1+(X-U)**2/(Dof*Covar))**((Dof+1)/2.0)


def Watershed_MRF(Iin, I_MM):

    # ------------------------------------------------------------------------------------ #
    #                                                                                      #
    # This algorithm is implemented by Oren Kraus on July 2013                             #
    #                                                                                      #
    # ------------------------------------------------------------------------------------ #

    Fgm = (I_MM > 0)
    SdsLab = mh.label(I_MM == 2)[0]
    SdsSizes = mh.labeled.labeled_size(SdsLab)
    too_small_Sds = np.where(SdsSizes < 30)
    SdsLab = mh.labeled.remove_regions(SdsLab, too_small_Sds)
    Sds = SdsLab > 0

    se2 = nd.generate_binary_structure(2, 2).astype(np.int)
    dilatedNuc = nd.binary_dilation(Sds, se2)
    Fgm = (dilatedNuc.astype(np.int) + Fgm.astype(np.int)) > 0

    FgmLab = mh.label(Fgm)[0]
    FgmSizes = mh.labeled.labeled_size(FgmLab)
    too_small_Fgm = np.where(FgmSizes < 30)
    FgmLab = mh.labeled.remove_regions(FgmLab, too_small_Fgm)
    Fgm = FgmLab > 0

    se3 = nd.generate_binary_structure(2, 1).astype(np.int)
    Fgm = nd.binary_erosion(Fgm, structure=se3)

    Fgm_Lab, Fgm_num = nd.measurements.label(Fgm)

    Nuc_Loc_1d = np.where(np.ravel(Sds == 1))[0]
    for Lab in range(Fgm_num):
        Fgm_Loc_1d = np.where(np.ravel(Fgm_Lab == Lab))[0]
        if not bool((np.intersect1d(Fgm_Loc_1d, Nuc_Loc_1d)).any()):
            Fgm[Fgm_Lab == Lab] = 0

    Im_ad = (np.double(Iin) * 2 ** 16 / Iin.max()).round()
    Im_ad = nd.filters.gaussian_filter(Im_ad, .5, mode='constant')

    Im_ad_comp = np.ones(Im_ad.shape)
    Im_ad_comp = Im_ad_comp * Im_ad.max()
    Im_ad_comp = Im_ad_comp - Im_ad
    mask = ((Sds == 1).astype(np.int) + (Fgm == 0).astype(np.int))
    mask = nd.label(mask)[0]
    LabWater = mh.cwatershed(np.uint16(Im_ad_comp), mask)
    back_loc_1d = np.where(np.ravel(Fgm == 0))[0]
    for Lab in range(2, LabWater.max()):
        cell_Loc_1d = np.where(np.ravel(LabWater == Lab))[0]
        if bool((np.intersect1d(cell_Loc_1d, back_loc_1d)).any()):
            LabWater[LabWater == Lab] = 1

    return LabWater


def not_on_border(width, height, loc_left, loc_upper, loc_right, loc_lower):

    if loc_left >= 0 and loc_right <= width and loc_upper >= 0 and loc_lower <= height:
        return True

    return False

