import numpy as np
import cv2
import matplotlib.pyplot as plt
import tqdm

import input_data

import imgaug as ia
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug.augmenters as iaa
import itertools

def visualizeAugData(imgs, labels, outPdfFile):
    gridH = 10
    gridW = 10
    fig, axs = plt.subplots(gridH, gridW)

    fig.set_size_inches(20, 20 * (gridH / gridW))

    numImgs = imgs.shape[0]
    Ids = list(range(numImgs))
    np.random.shuffle(Ids)
    print("Number corners:", Ids)
    for iA, iC in itertools.product(range(gridH), range(gridW)):
        if iC + gridW * iA >= numImgs:
            break
        imgId = Ids[iC + gridW * iA]

        axs[iA, iC].imshow(np.squeeze(imgs[imgId, :, :]), cmap='gray')
        axs[iA, iC].set_title(str(np.argmax(labels[imgId, :])))
        axs[iA, iC].axis('off')

    fig.savefig(outPdfFile, dpi=500, transparent=True, bbox_inches='tight', pad_inches=0)


def augmentImgs(imgs, labels, augCfg):
    seq = iaa.Sequential([
        # iaa.ElasticTransformation(alpha=500, sigma=50),
        # iaa.Multiply(augCfg['mul']),
        iaa.Affine(rotate=augCfg['r'], shear=augCfg['shear'], order=[0])
        # iaa.AddToHueAndSaturation((-10, 10))  # color jitter, only affects the image
    ])

    imgsAuged = []
    labelsNew = []
    for i in tqdm.tqdm(range(imgs.shape[0])):
        for iA in range(augCfg['numAugs']):
            imgAug = seq(image=imgs[i,:,:])
            imgsAuged.append(imgAug)
            labelsNew.append(labels[i, :])
            # cv2.imshow('imgAug', imgAug)
            # cv2.waitKey(-1)
            # print(labels[i, :])
    return np.array(imgsAuged), np.array(labelsNew)

if __name__ == "__main__":
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

    readWeights = False

    imgsTrain = mnist.train._images
    imgsTrain = imgsTrain.reshape(-1, 28, 28)
    labelsTrain = mnist.train.labels

    imgsTest = mnist.test._images
    imgsTest = imgsTest.reshape(-1, 28, 28)
    labelsTest = mnist.test.labels

    # cv2.imshow('imgtrain', imgsTrain[0,:,:])
    # cv2.waitKey(-1)

    augCfg = {
        # 'mul': (0.8, 1.2),
        'r': (-180, 180),
        'shear': (30, 30),
        'scales':(0.7, 1.2),
        'numAugs': 10,

    }

    imgsTrainAug, labelsTrainAug = augmentImgs(imgsTrain, labelsTrain, augCfg)
    imgsTestAug, labelsTestAug = augmentImgs(imgsTest, labelsTest, augCfg)

    np.save('imgsTrainAug.npy', imgsTrain)
    np.save('labelsTrainAug.npy', labelsTrainAug)

    np.save('imgsTestAug.npy', imgsTestAug)
    np.save('labelsTestAug.npy', labelsTestAug)

    # imgsTestAug, labelsTestAug = augmentImgs(imgsTest[:200,:,:], labelsTest[:200, :], augCfg)

    visualizeAugData(imgsTrain, labelsTrain, 'AugmentedTrain.pdf')
    visualizeAugData(imgsTestAug, labelsTestAug, 'AugmentedTest.pdf')