import math
import gzip
import sys
import os

from PIL import Image
import numpy as np


def arr_from_img(im, mean=0, std=1):
    '''
    Args:
        im: Image
        shift: Mean to subtract
        std: Standard Deviation to subtract
    Returns:
        Image in np.float32 format, in width height channel format. With values in range 0,1
        Shift means subtract by certain value. Could be used for mean subtraction.
    '''
    width, height = im.size
    arr = im.getdata()
    c = int(np.product(arr.size) / (width * height))

    return (np.asarray(arr, dtype=np.float32).reshape((height, width, c)).transpose(2, 1, 0) / 255. - mean) / std


def get_image_from_array(X, index, mean=0, std=1):
    '''
    Args:
        X: Dataset of shape N x C x W x H
        index: Index of image we want to fetch
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    '''
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = (((X[index] + mean) * 255.) * std).reshape(ch, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret

# loads mnist from web on demand
def load_dataset(training=True):
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        return data / np.float32(255)

    if training:
        return load_mnist_images('train-images-idx3-ubyte.gz')
    return load_mnist_images('t10k-images-idx3-ubyte.gz')


def generate_moving_mnist(training, shape=(64, 64), num_frames=30, num_sequences=2, original_size=28, nums_per_image=3,
                          traj_per_image=2):
    '''
    Args:
        training: Boolean, used to decide if downloading/generating train set or test set
        shape: Shape we want for our moving images (new_width and new_height)
        num_frames: Number of frames in a particular movement/animation/gif
        num_sequences: Number of movement/animations/gif to generate
        original_size: Real size of the images (eg: MNIST is 28x28)
        nums_per_image: Digits per movement/animation/gif.
    Returns:
        Dataset of np.uint8 type with dimensions num_frames * num_sequences x 1 x new_width x new_height
    '''
    mnist = load_dataset(training)
    width, height = shape

    # Get how many pixels can we move around a single image (to fit its width)
    lims = (x_lim, y_lim) = width - original_size, height - original_size
    print(lims)

    # Create a dataset of shape of num_frames * num_sequences x 1 x new_width x new_height
    # Eg : 3000000 x 1 x 64 x 64
    dataset = np.empty((num_frames * num_sequences, 1, width, height), dtype=np.uint8)

    for img_idx in range(num_sequences):
        # Randomly generate direction, speed and velocity for both images
        direcs = np.pi * (np.random.rand(traj_per_image) * 2 - 1)  # Scalars, one per digit
        direcs = np.insert(direcs, 0, direcs[0])
        speeds = np.random.randint(5, size=traj_per_image) + 2  # Scalars, one per digit
        speeds = np.insert(speeds, 0, speeds[0])
        # veloc is 2xnums_per_image (x and y component for velocity for each digit)
        veloc = np.asarray([(speed * math.cos(direc), speed * math.sin(direc)) for direc, speed in zip(direcs, speeds)])

        # Get a list containing three PIL images randomly sampled from the database
        mnist_images = [Image.fromarray(get_image_from_array(mnist, r, mean=0)).resize((original_size, original_size),
                                                                                       Image.ANTIALIAS) \
                        for r in np.random.randint(0, mnist.shape[0], nums_per_image)]
        # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
        positions = np.asarray([(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(nums_per_image)])
        print(positions)

        # Generate new frames for the entire num_framesgth
        for frame_idx in range(num_frames):

            canvases = [Image.new('L', (width, height)) for _ in range(nums_per_image)]
            canvas = np.zeros((1, width, height), dtype=np.float32)

            # In canv (i.e Image object) place the image at the respective positions
            # Super impose both images on the canvas (i.e empty np array)
            for i, canv in enumerate(canvases):
                canv.paste(mnist_images[i], tuple(positions[i].astype(int)))
                canvas += arr_from_img(canv, mean=0)

            # Get the next position by adding velocity
            next_pos = positions + veloc

            # Iterate over velocity and see if we hit the wall
            # If we do then change the  (change direction)
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j] + 2:
                        # One of list(veloc[i][:j]) or list(veloc[i][j + 1:]) always gives an empty list [].
                        # Whereas [-1 * veloc[i][j]] reverses that component.
                        # list(list + list) is just concatenating lists.
                        veloc[i] = list(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:]))

            # Make the permanent change to position by adding updated velocity
            positions = positions + veloc

            # Add the canvas to the dataset array
            dataset[img_idx * num_frames + frame_idx] = (canvas * 255).clip(0, 255).astype(np.uint8)

    return dataset


def main(training, dest, filetype='jpg', frame_size=64, num_frames=30, num_sequences=2, original_size=28,
         nums_per_image=3):
    dat = generate_moving_mnist(training, shape=(frame_size, frame_size), num_frames=num_frames, num_sequences=num_sequences, \
                                original_size=original_size, nums_per_image=nums_per_image)
    n = num_sequences * num_frames
    if filetype == 'npz':
        np.savez(dest, dat)
    elif filetype == 'jpg':
        for i in range(dat.shape[0]):
            Image.fromarray(get_image_from_array(dat, i, mean=0)).save(os.path.join(dest, '{}.jpg'.format(i)))