import imageio
import os
import shutil

def makeGIF(plotroot, filename='Qtable', interval=0.3, end_pause=12):
    names = {}
    pics_for_each_gif = {}
    # Set the directory you want to start from
    rootDir = './'+plotroot+'/'
    files={}
    for dirName, subdirList, fileList in os.walk(rootDir):
        for f in fileList:
            if f[0]=='Q':
                iteration = f[1:-4]
                files[int(iteration)] = f
    keys = list(files.keys())
    keys.sort()


    dims = {}
    for k in keys:
        path = rootDir+files[k]
        im = imageio.imread(path)
        (h, w, _) = im.shape
        # print("height = "+str(h)+", width = "+str(w))


    images = []
    for k in keys:
        images.append(imageio.imread(rootDir+files[k]))
    for _ in range(end_pause):
        images.append(imageio.imread(rootDir+files[keys[-1]]))
    kargs = { 'duration': interval }
    imageio.mimsave(filename+'.gif', images, **kargs)


    # delete all of the png files
    shutil.rmtree(rootDir)
    os.mkdir(plotroot)
