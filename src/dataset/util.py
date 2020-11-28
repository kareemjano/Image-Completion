from zipfile import ZipFile
import urllib
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from skimage import io, color


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def _download_zip(url, zip_filename, target_dir):
    zip_path = os.path.join(target_dir, zip_filename)
    if len(os.listdir(target_dir)) == 0:
        print('\ndownloading zip file...')


        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, zip_path, reporthook=t.update_to)
    else:
        print('Dir is not empty')

    return zip_path

def get_dataset(dataset_url):
    print('\ncreating directories...')
    data_path = os.path.join(os.getcwd(), 'data')
    if 'data' not in os.listdir():
        print('creating', data_path)
        os.mkdir(data_path)
    else:
        print('data dir exists')

    url = dataset_url
    zip_filename = url.split('/')[-1]
    filename = zip_filename.split('.')[0]
    target_dir = os.path.join(data_path, filename)
    if filename not in os.listdir(data_path):
        print('creating', target_dir)
        os.mkdir(target_dir)
    else:
        print(target_dir, ' exists')

    zip_path = _download_zip(url, zip_filename, target_dir)

    if os.path.exists(zip_path):
        print('\nunzipping file...')
        thefile = ZipFile(zip_path)
        thefile.extractall(target_dir)
        thefile.close()

        print('\nremoving zip file...')
        os.remove(zip_path)
        print('\ndone')
    else:
        print('zip file doesnt exist')

    return target_dir

def visualize_samples(imgs_path, gray=False, n_cols=5, n_rows=1):
    """Visualize samples."""

    plt.figure(figsize = (3*n_cols,3*n_rows))
    for n,i in enumerate(np.random.randint(len(imgs_path), size = n_cols*n_rows)):
        plt.subplot(n_rows,n_cols,n+1)
        plt.axis('off')
        img = io.imread(imgs_path[i])
        if gray:
            img = color.rgb2gray(img)
            plt.imshow(img, cmap=plt.cm.gray)
        else:
            plt.imshow(img)
    plt.show()

def visualize_torch(images, gray=False, n_cols=5, n_rows=1):
    """Visualize samples."""

    fig = plt.figure(figsize = (3*n_cols,3*n_rows))
    for i in range(n_cols*n_rows):
        plt.subplot(n_rows,n_cols,i+1)
        plt.axis('off')
        img = images[i].permute(1, 2, 0).squeeze()
        if gray:
            plt.imshow(img, cmap=plt.cm.gray)
        else:
            plt.imshow(img)
    return fig

