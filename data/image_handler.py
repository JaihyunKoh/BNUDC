import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
NPY_EXTENSIONS = ['.npy']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def is_npy_file(filename):
    return any(filename.endswith(extension) for extension in NPY_EXTENSIONS)

def make_dataset_image(dir):
    imagesX = []
    imagesY = []
    for root, dir, fnames in sorted(os.walk(dir)):
        if os.path.basename(root) == 'sharp' or os.path.basename(root) == 'GT' or os.path.basename(root) == 'HQ':
            for fname in fnames:
                if is_image_file(fname) or is_npy_file(fname):
                    path = os.path.join(root, fname)
                    imagesX.append(path)
        elif os.path.basename(root) == 'blur' or os.path.basename(root) == 'input' or os.path.basename(root) == 'LQ':
            for fname in fnames:
                if is_image_file(fname) or is_npy_file(fname):
                    path = os.path.join(root, fname)
                    imagesY.append(path)
    return imagesX, imagesY