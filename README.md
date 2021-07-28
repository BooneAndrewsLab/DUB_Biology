# DUB_Biology

This repository contains scripts and example datasets for the paper: ________


### Prerequisites
Python 2.7: https://www.python.org/downloads

### Installation
Clone the repository
```
git clone https://github.com/BooneAndrewsLab/DUB_Biology.git
cd DUB_Biology
```

Install required packages
```
pip install -r requirements.txt
```


### Input Requirements 
* A folder containing microscopy images taken from Phenix High-Content Screening System.
Example images can be found in example/images
* For each run, the user needs to specify which row to process

### Running the Pipeline: Segmentation + DeepLoc + Protein Abundance

**segment_and_predict.py**

This script is a multi-step pipeline that consists of running single cell Segmentation, DeepLoc prediction and Protein Abundance
measurement. The output file contains the x and y coordinates of the cells on each of the images as well as the protein abundance and 
predicted protein localization based on either: (1) a pretrained model from Chong et al.'s dataset, (2) a fine-tuned model on wt2017 
dataset or (3) a pretrained model from DUBs's phenix dataset.

Script parameters:
```
-m	choose which model to use: chong or wt2017 or harsha. The default is harsha.
-i	path to the input images
-r	row to process (i.e. r01)
-o	path to output  directory
-n	output filename
-f	path to mapping sheet (optional)
-p	indicate plate number when processing multi-plate screen and when mapping sheet is specified. Default is 1.
-t	indicate the image type. The options are:
	opera4 – 4field x 2 channel
	opera1 – 1field x 1 channel
	phenix – 1field x 1 channel (default)
-s	choose which to save:
	cell – compute prediction score for each cell
	well – get average prediction score for all cells which belong to a well
	both – save both perCell and perWell output files (default)
-b	use this flag if you want to use the blur function for segmenting the cell. Default is ‘False’.
-x	use this flag if the input folder contains images with multiple timepoints. Default is ‘False’. Use this flag
	only when the image type is phenix.
-y	use this flag if you want to use FarRed channel (3rd channel) for segmentation instead of the Red channel. 
	Can ONLY be used when image type is phenix.
```

Example usage when using the default values:
```
python segment_and_predict.py -i example/images -r r01 -o example/ -n Image_predictions.csv -f example/mapping_sheet.csv
```