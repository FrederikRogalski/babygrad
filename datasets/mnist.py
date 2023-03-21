import os
import gzip
import numpy as np
from urllib.request import urlretrieve


class MNIST:
    def __init__(self):
        self.base_url = 'http://yann.lecun.com/exdb/mnist/'
        self.files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
        self.train_images_url = self.base_url + self.files[0]
        self.train_labels_url = self.base_url + self.files[1]
        self.test_images_url = self.base_url + self.files[2]
        self.test_labels_url = self.base_url + self.files[3]
        self.cache_dir = os.path.join(os.path.expanduser("~/.cache/babygrad/"), self.__class__.__name__.lower() + '/')
        if not self.cached():
            self.download()
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.load()
    def cached(self):
        # check if all files exist ín cache_dir
        for f in self.files:
            # check if file exists
            if not os.path.exists(os.path.join(self.cache_dir, f)):
                return False
        return True
            
    def download(self):
        # create cache_dir if not exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # download files
        for f in self.files:
            urlretrieve(self.base_url + f, os.path.join(self.cache_dir, f))
        print('Downloaded and cached MNIST dataset in {}'.format(self.cache_dir))
    
    def load(self):
        # load files
        train_images = self.load_images(os.path.join(self.cache_dir, self.files[0]))
        train_labels = self.load_labels(os.path.join(self.cache_dir, self.files[1]))
        test_images = self.load_images(os.path.join(self.cache_dir, self.files[2]))
        test_labels = self.load_labels(os.path.join(self.cache_dir, self.files[3]))
        return train_images, train_labels, test_images, test_labels
    
    def load_images(self, path):
        with gzip.open(path, 'rb') as f:
            magic = int.from_bytes(f.read(4), byteorder='big')
            assert magic == 2051, 'Magic number mismatch, expected 2051, got {}'.format(magic)
            num_images = int.from_bytes(f.read(4), byteorder='big')
            num_rows = int.from_bytes(f.read(4), byteorder='big')
            num_cols = int.from_bytes(f.read(4), byteorder='big')
            buf = f.read(num_rows * num_cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(num_images, num_rows, num_cols)
            return data
    
    def load_labels(self, path):
        with gzip.open(path, 'rb') as f:
            magic = int.from_bytes(f.read(4), byteorder='big')
            assert magic == 2049, 'Magic number mismatch, expected 2049, got {}'.format(magic)
            num_items = int.from_bytes(f.read(4), byteorder='big')
            buf = f.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            return labels
    
if __name__ == '__main__':
    mnist = MNIST()
    print(mnist.train_images.shape)
    print(mnist.train_labels.shape)
    print(mnist.test_images.shape)
    print(mnist.test_labels.shape)