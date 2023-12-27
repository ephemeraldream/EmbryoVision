from PIL import Image
import os, sys


#def resize():
#    for item in os.listdir(path):
#        if os.path.isfile(item):
#            im = Image.open(item)
#            f, e = os.path.splitext(item)
#            imResize = im.resize((384,275), Image.ANTIALIAS)
#            imResize.save(f + ' resized.jpg', 'JPEG', quality=90)


def resize():
    path = 'C:\Work\Data\embryoCV\SelectedImages\\'
    dirs = os.listdir( path )
    count = 0
    for files in dirs:
         im = Image.open(path+files)
         f, e = os.path.splitext(path+files)
         imResize = im.resize((384,275), Image.Resampling.LANCZOS)
         imResize.save('C:\Work\Data\embryoCV\Images1Resized\\' + str(count) + 'resized.jpg', 'JPEG', quality=90)
         count += 1

resize()