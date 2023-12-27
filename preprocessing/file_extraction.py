import os
import shutil

directory = os.fsencode('C:\Work\Data\embryoCV\RawData1')
count = 17
for subdir, dirs, files in os.walk(directory):
    for file in files:
        x = os.path.join(subdir, file)
        if count % 17 == 0:
            shutil.copy(str(x)[2:-1], "C:\Work\Data\embryoCV\SelectedImages")
        count += 1



