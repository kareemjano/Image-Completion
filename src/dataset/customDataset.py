from torch.utils.data import Dataset
import os
from skimage import io, color, transform
import glob


class custumDataset(Dataset):
    # define the constructor of this dataset object
    def __init__(self, dataset_folder, in_size=256, out_size=40, out_center=[128, 128], ext='**/*.ppm', transform=None):

        self.out_size = out_size
        self.in_size = in_size
        self.out_center = out_center

        self.transform = transform
        self.imgs_path = glob.glob(os.path.join(dataset_folder, ext), recursive=True)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        image = transform.resize(color.rgb2gray(io.imread(self.imgs_path[idx])), (self.in_size, self.in_size))
        in_img = image.copy()
        top = int(self.out_center[0]-(self.out_size/2))
        left = int(self.out_center[1]-(self.out_size/2))
        in_img[top:top+self.out_size,left:left+self.out_size] = 0
        target_img = image[top:top+self.out_size,left:left+self.out_size]
        if self.transform:
            image = self.transform(image)
            in_img = self.transform(in_img)
            target_img = self.transform(target_img)

        return image, in_img, target_img
