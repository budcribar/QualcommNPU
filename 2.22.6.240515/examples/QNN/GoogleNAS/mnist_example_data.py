# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.

import os
import gzip
import wget
import glob
import numpy as np

path = 'images'
os.makedirs(path, exist_ok = True)
url = 'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz'
wget.download(url)
f = gzip.open('t10k-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 100

# Get rid of the header info
f.read(16)

# Read some images
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)
for i in range(num_images):
    data[i].tofile(os.path.join(path,str(i)+'.raw'))


files = glob.glob(os.path.join(path,'*.raw'))
inputlist = open('input_list.txt','w')
for filename in files:
    inputlist.write(filename+'\n')
inputlist.close()
