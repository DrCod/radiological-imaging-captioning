__author__ = 'tylin'
__version__ = '2.0'
# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  segToMask  - Convert polygon segmentation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>segToMask, COCO>showAnns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from skimage.draw import polygon
import urllib
import copy
import itertools
import os
import string
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import pandas as pd

class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = {}
        self.imgToAnns = {}
        self.imgs = {}
        self.img_name_to_id = {}

        if not annotation_file == None:
            print ('loading medical annotations into memory...')
            tic = time.time()
            print ('Done (t=%0.2fs)'%(time.time()- tic))
            self.dataset = dataset
            self.process_dataset()
            self.createIndex()

    def createIndex(self):
        # create index
        print ('creating index...')
        anns = {}
        imgToAnns = {}
        catToImgs = {}
        cats = {}
        imgs = {}
        img_name_to_id = {}

        if 'annotations' in self.dataset:
            imgToAnns = {ann['image_id']: [] for ann in self.dataset['annotations']}
            anns =      {ann['id']:       [] for ann in self.dataset['annotations']}
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']] += [ann]
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            imgs      = {im['id']: {} for im in self.dataset['images']}
            for img in self.dataset['images']:
                imgs[img['id']] = img
                img_name_to_id[img['file_name']] = img['id']


        print ('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.imgs = imgs
        self.img_name_to_id = img_name_to_id

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print ('%s: %s'%(key, value))

    def getAnnIds(self, imgIds=[],iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]

        if len(imgIds) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                # this can be changed by defaultdict
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids


    def getImgIds(self, imgIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if type(imgIds) == list else [imgIds]

        if len(imgIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]



    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == list:
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]


    def process_dataset(self):
        for ann in self.dataset['annotations']:
            q = ann['caption'].lower()
            if q[-1]!='.':
                q = q + '.'
            ann['caption'] = q

    def filter_by_cap_len(self, max_cap_len):
        print("Filtering the captions by length...")
        keep_ann = {}
        keep_img = {}
        for ann in tqdm(self.dataset['annotations']):
            if len(word_tokenize(ann['caption']))<=max_cap_len:
                keep_ann[ann['id']] = keep_ann.get(ann['id'], 0) + 1
                keep_img[ann['image_id']] = keep_img.get(ann['image_id'], 0) + 1

        self.dataset['annotations'] = \
            [ann for ann in self.dataset['annotations'] \
            if keep_ann.get(ann['id'],0)>0]
        self.dataset['images'] = \
            [img for img in self.dataset['images'] \
            if keep_img.get(img['id'],0)>0]

        self.createIndex()

    def filter_by_words(self, vocab):
        print("Filtering the captions by words...")
        keep_ann = {}
        keep_img = {}
        for ann in tqdm(self.dataset['annotations']):
            keep_ann[ann['id']] = 1
            words_in_ann = word_tokenize(ann['caption'])
            for word in words_in_ann:
                if word not in vocab:
                    keep_ann[ann['id']] = 0
                    break
            keep_img[ann['image_id']] = keep_img.get(ann['image_id'], 0) + 1

        self.dataset['annotations'] = \
            [ann for ann in self.dataset['annotations'] \
            if keep_ann.get(ann['id'],0)>0]
        self.dataset['images'] = \
            [img for img in self.dataset['images'] \
            if keep_img.get(img['id'],0)>0]

        self.createIndex()

    def all_captions(self):
        return [ann['caption'] for ann_id, ann in self.anns.items()]
