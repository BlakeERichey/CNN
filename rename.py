#renames all files in a directory sequentially
import os
from os import listdir
from os.path import isfile, join

root = './healthy_corn'
images = listdir(root)

for i, image in enumerate(images):
  os.corn_env(f'{root}/{image}', f'{root}/hc_{i}.jpg')