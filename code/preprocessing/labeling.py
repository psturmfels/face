import numpy as np

from preprocessing import preprocess

def writeBlankLabels(outFile='../data/labels.txt', imageDir='../data/cropped', imageExtension='.png'):
    images, names = preprocess.getAllImagesInDirectory(dir=imageDir, imageExtension=imageExtension)
    labels = np.zeros((len(images)), dtype=int)

    f = open(outFile, 'w')
    for index in range(len(names)):
        f.write(names[index] + ',' + str(labels[index]) + '\n')

    f.close()

def getImagesAndLabels(labelsFile='../data/labels.txt', imageDir='../data/cropped', imageExtension='.png'):
    images, names = preprocess.getAllImagesInDirectory(dir=imageDir, imageExtension=imageExtension)
    labels = []

    f = open(labelsFile, 'r')
    counter = 0
    for line in f:
        label = int(line.split(',')[-1])
        labels.append(label)

        imageName = line.split(',')[0]
        if (names[counter] != imageName):
            raise ValueError('Names' + names[counter] + ',' + imageName +
            ' in the label file and in the image directory do not match.')
        counter += 1

    return (images, names, labels)

if __name__ == '__main__':
    writeBlankLabels()
    getImagesAndLabels()
