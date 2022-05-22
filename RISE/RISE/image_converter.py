from PIL import Image
from os import listdir
from os.path import splitext

target_directory = 'C:\d drive\design credit\RISE\RISE\imagenet-sample-images'
target = '.png'

for file in listdir(target_directory):
    filename, extension = splitext(file)
    try:
        if extension not in ['.py', target]:
            im = Image.open(filename + extension)
            im.save(filename + target)
    except OSError:
        print('Cannot convert %s' % file)