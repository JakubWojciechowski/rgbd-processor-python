import scipy.io as spio
import numpy as np
import imageio
import math
from os import listdir
from os.path import isfile, join, abspath, exists, curdir, expanduser
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from camera import cropCamera,getCameraParam, processCamMat
from depthImgProcessor import processDepthImage
from utils import checkDirAndCreate
from glob import glob
import json
import cv2
import types

ROOTPATH = expanduser('~/data/')


ans = {
  1:  'wall',
  2:  'floor',
  3:  'cabinet',
  4:  'bed',
  5:  'chair',
  6:  'sofa',
  7:  'table',
  8:  'door',
  9:  'window',
  10:  'bookshelf',
  11:  'picture',
  12:  'counter',
  13:  'blinds',
  14:  'desk',
  15:  'shelves',
  16:  'curtain',
  17:  'dresser',
  18:  'pillow',
  19:  'mirror',
  20:  'floor_mat',
  21:  'clothes',
  22:  'ceiling',
  23:  'books',
  24:  'fridge',
  25:  'tv',
  26:  'paper',
  27:  'towel',
  28:  'shower_curtain',
  29:  'box',
  30:  'whiteboard',
  31:  'person',
  32:  'night_stand',
  33:  'toilet',
  34:  'sink',
  35:  'lamp',
  36:  'bathtub',
  37:  'bag'
}

def search_in_dictionary(dictionary, search_value):
    for key, value in dictionary.items():
        if value == search_value:
            return key


def getHHAImg(depthImage, missingMask,cameraMatrix):
    pc, N, yDir, h, R = processDepthImage(depthImage * 100, missingMask, cameraMatrix)

    tmp = np.multiply(N, yDir)
    # with np.errstate(invalid='ignore'):
    acosValue = np.minimum(1,np.maximum(-1,np.sum(tmp, axis=2)))
    angle = np.array([math.degrees(math.acos(x)) for x in acosValue.flatten()])
    angle = np.reshape(angle, h.shape)

    pc[:,:,2] = np.maximum(pc[:,:,2], 100)
    I = np.zeros(pc.shape)
    I[:,:,0] = 31000/pc[:,:,2]
    I[:,:,1] = h
    I[:,:,2] = (angle + 128-90)
    HHA = I.astype(np.uint8)
    return HHA

def getHeightMap(depthImage, missingMask,cameraMatrix):
    height, width = depthImage.shape

    pc, N, yDir, h, R = processDepthImage(depthImage, missingMask, cameraMatrix)

    X = pc[:,:,0]
    Y = h
    Z = pc[:,:,2]
    X = X - np.min(X) + 1
    Z = Z - np.min(Z) + 1
    roundX = X.astype(int)
    roundZ = Z.astype(int)
    maxX = np.max(roundX)
    maxZ = np.max(roundZ)
    mx,mz = np.meshgrid(np.array(range(maxX+1)), np.array(range(maxZ+1)))
    heightMap = np.ones([maxZ+1, maxX+1]) * np.inf
    for i in range(height):
        for j in range(width):
            tx = roundX[i,j]
            tz = roundZ[i,j]
            heightMap[tz,tx] = min(h[i,j], heightMap[tz,tx])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(mx, mz, heightMap, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    ax.set_zlabel('height above ground')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def generate_hha(chooseSplit = "training"):
    rootpath = ROOTPATH
    outputpath = 'imgs/'

    olderr = np.seterr(all='ignore')
    try:
        fp = open(rootpath+'SUNRGBD/'+'sunrgbd_'+chooseSplit+'_images.txt', 'r')
        filenameSet = fp.readlines()
    finally:
        fp.close()
    checkDirAndCreate(outputpath + chooseSplit)
    for idx, file in enumerate(filenameSet):
        split_items = file.split('/')
        camAddr = rootpath + '/'.join(p for p in split_items[:-2]) + '/intrinsics.txt'
        with open(camAddr, 'r') as camf:
            cameraMatrix = processCamMat(camf.readlines())

        depthAddr_root  = rootpath + '/'.join(p for p in split_items[:-2]) + '/depth_bfx/' #+ split_items[-1].split('.')[0]+'_abs.png'
        rawDepthAddr_root = rootpath + '/'.join(p for p in split_items[:-2]) + '/depth/' #+ split_items[-1].split('.')[0]+'_abs.png'

        depthAddr = [depthAddr_root + f for f in listdir(depthAddr_root) if isfile(join(depthAddr_root,f ))][0]
        rawDepthAddr = [rawDepthAddr_root  +  f for f in listdir(rawDepthAddr_root) if isfile(join(rawDepthAddr_root,f ))][0]

        depthImage = imageio.imread(depthAddr).astype(float)/10000
        rawDepth = imageio.imread(rawDepthAddr).astype(float)/100000
        missingMask = (rawDepth == 0)

        HHA = getHHAImg(depthImage, missingMask, cameraMatrix)

        imageio.imwrite(outputpath + chooseSplit + '/hha/' + str(idx+1) + '.png',HHA)
        imageio.imwrite(outputpath + chooseSplit + '/height/' + str(idx+1) + '.png', HHA[:,:,1])

def get_paths(image_path):
    depth_path = image_path.replace('image', 'depth').replace('.jpg', '.png')
    intrinsics_path = abspath(join(image_path, '../../intrinsics.txt'))
    annotation_path = abspath(join(image_path, '../../annotation2Dfinal/index.json'))
    assert(exists(depth_path))
    assert(exists(intrinsics_path))
    assert(exists(annotation_path))
    return depth_path, intrinsics_path, annotation_path

def generate_map(annotation_path, h, w):
    mask = np.zeros((h, w), dtype = np.uint8)
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
    except:
        with open(annotation_path, 'r') as f:
            string_data = f.read()
        if '\\' in string_data:
            string_data = string_data.replace('\\', '')
            data = json.loads(string_data)
            print('FIXED')
        else:
            print('ERROR decoding {}'.format(annotation_path))

    for poly in data['frames'][0]['polygon']:
        object_id = poly['object']
        if object_id >= len(data['objects']):
            continue
        obj_name = data['objects'][object_id]['name']
        obj_id = search_in_dictionary(ans, obj_name)
        if not obj_id: #class not in 37 classes given in ans
            continue
        if not isinstance(poly['x'], list):
            continue
        cnts = np.expand_dims(np.stack([poly['x'], poly['y']], axis=1), axis = 1).astype(np.int)
        if len(cnts) > 0:
            mask = cv2.drawContours(mask, [cnts], 0 , obj_id, -1)
    return mask

def test_annotations(annotation_path):
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
    except:
        with open(annotation_path, 'r') as f:
            string_data = f.read()
        if '\\' in string_data:
            string_data = string_data.replace('\\', '')
            data = json.loads(string_data)
            print('FIXED')
        else:
            print('ERROR decoding {}'.format(annotation_path))


def generate_targets(chooseSplit = "training"):
    rootpath = ROOTPATH
    outputpath = 'imgs/'
    olderr = np.seterr(all='ignore')
    try:
        fp = open(rootpath+'SUNRGBD/'+'sunrgbd_'+chooseSplit+'_images.txt', 'r')
        filenameSet = fp.readlines()
    finally:
        fp.close()
    checkDirAndCreate(outputpath + chooseSplit, checkNameList=['targets'])
    for idx, file in enumerate(filenameSet):
        split_items = file.split('/')
        depthAddr_root  = rootpath + '/'.join(p for p in split_items[:-2]) + '/depth_bfx/' #+ split_items[-1].split('.')[0]+'_abs.png'
        depthAddr = [depthAddr_root + f for f in listdir(depthAddr_root) if isfile(join(depthAddr_root,f ))][0]

        depthImage = imageio.imread(depthAddr).astype(float)/10000
        h, w = depthImage.shape

        annotation_path = rootpath + '/'.join(p for p in split_items[:-2]) + '/annotation2Dfinal/index.json'
        #test_annotations(annotation_path)
        mask = generate_map(annotation_path, h, w)

        imageio.imwrite(outputpath + chooseSplit + '/targets/' + str(idx+1) + '.png',mask)

def generate_lst(chooseSplit = "training"):
    rootpath = ROOTPATH
    outputpath = 'imgs/'
    try:
        fp = open(rootpath+'SUNRGBD/'+'sunrgbd_'+chooseSplit+'_images.txt', 'r')
        filenameSet = fp.readlines()
    finally:
        fp.close()
    with open('sunrgbd_{}.lst'.format(chooseSplit), 'w') as f:
        for idx, file in enumerate(filenameSet):
            imgname = abspath(join(rootpath, file)).rstrip()
            segname = abspath(join(curdir, outputpath, chooseSplit, 'targets','{}.png'.format(idx+1)))
            split_items = file.split('/')
            depthAddr_root  = rootpath + '/'.join(p for p in split_items[:-2]) + '/depth_bfx/'
            depthname = [depthAddr_root + f for f in listdir(depthAddr_root) if isfile(join(depthAddr_root,f ))][0]
            HHAname = abspath(join(curdir, outputpath, chooseSplit, 'hha','{}.png'.format(idx+1)))
            assert(exists(imgname) and exists(segname) and exists(depthname) and exists(HHAname))
            f.write('{} {} {} {}\n'.format(imgname, segname, depthname, HHAname))




if __name__ == "__main__":
    for chooseSplit in ["training", 'testing']:
        generate_hha(chooseSplit)
        generate_targets(chooseSplit)
        generate_lst(chooseSplit)
