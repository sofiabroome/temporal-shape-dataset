import math
import gzip
import sys
import os

from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np

# Code adapted from the following gist by Praateek Mahajan:
# https://gist.github.com/praateekmahajan/b42ef0d295f528c986e2b3a0b31ec1fe


def arr_from_img(im, mean=0, std=1):
    """
    Args:
        im: Image
        mean: Mean to subtract
        std: Standard Deviation to subtract
    Returns:
        Image in np.float32 format, in width height channel format. With values in range 0,1
        Shift means subtract by certain value. Could be used for mean subtraction.
    """
    width, height = im.size
    arr = im.getdata()
    c = int(np.product(arr.size) / (width * height))

    return (np.asarray(arr, dtype=np.float32).reshape(
        (height, width, c)).transpose(2, 1, 0) / 255. - mean) / std


def get_image_from_array(data_array, index, allow_nuances, mean=0, std=1):
    """
    Args:
        data_array: Dataset of shape N x C x W x H
        index: Index of image we want to fetch
        allow_nuances: Boolean, whether to rescale (=removes nuances) or not
        mean: Mean to add
        std: Standard Deviation to add
    Returns:
        Image with dimensions H x W x C or H x W if it's a single channel image
    """
    ch, w, h = data_array.shape[1], data_array.shape[2], data_array.shape[3]

    if allow_nuances:
        scaling_factor = 1
    else:
        scaling_factor = 255.

    ret = (((data_array[index] + mean) * scaling_factor) * std).reshape(
        ch, w, h).transpose(2, 1, 0).clip(0, 255).astype(np.uint8)
    if ch == 1:
        ret = ret.reshape(h, w)
    return ret


# loads mnist from web on demand
def load_mnist(training=True):
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
        return data / np.float32(255)

    if training:
        return load_mnist_images('../../data/train-images-idx3-ubyte.gz')
    return load_mnist_images('../../data/t10k-images-idx3-ubyte.gz')


def generate_temporal_shape_dataset(training, shape=(64, 64), num_frames=30, num_sequences=2,
                                    original_size=28, nums_per_image=3, traj_per_image=2):
    """
    Args:
        training: Boolean, used to decide if downloading/generating train set or test set
        shape: Shape we want for our moving images (new_width and new_height)
        num_frames: Number of frames in a particular movement/animation/gif
        num_sequences: Number of movement/animations/gif to generate
        original_size: Real size of the images (eg: MNIST is 28x28)
        nums_per_image: Digits per movement/animation/gif.
        traj_per_image: The number of different trajectories per movement/animation/gif.
    Returns:
        Dataset of np.uint8 type with dimensions
        num_frames * num_sequences x 1 x new_width x new_height
    """
    mnist = load_mnist(training)
    width, height = shape

    labels = []

    # Get how many pixels can we move around a single image (to fit its width)
    lims = (x_lim, y_lim) = width - original_size, height - original_size

    # Create a dataset of shape of num_frames * num_sequences x 1 x new_width x new_height
    # Eg : 3000000 x 1 x 64 x 64
    dataset = np.empty((num_frames * num_sequences, 1, width, height), dtype=np.uint8)

    print('Generating sequences...')
    for img_idx in tqdm(range(num_sequences)):
        # Randomly generate direction, speed and velocity for both images
        direcs = np.pi * (np.random.rand(traj_per_image) * 2 - 1)  # Scalars, one per digit
        direcs = np.insert(direcs, 0, direcs[0])
        speeds = np.random.randint(5, size=traj_per_image) + 2  # Scalars, one per digit
        speeds = np.insert(speeds, 0, speeds[0])
        # veloc is 2xnums_per_image (x and y component for velocity for each digit)
        veloc = np.asarray([(speed * math.cos(direc), speed * math.sin(direc))
                            for direc, speed in zip(direcs, speeds)])

        # Get a list containing three PIL images randomly sampled from the database
        mnist_images = [Image.fromarray(
            get_image_from_array(mnist, r, mean=0)).resize(
            (original_size, original_size), Image.ANTIALIAS)
            for r in np.random.randint(0, mnist.shape[0], nums_per_image)]
        # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
        positions = np.asarray([(np.random.rand() * x_lim, np.random.rand() * y_lim)
                                for _ in range(nums_per_image)])
        label_positions = [(x_ul, y_ul, x_ul + original_size, y_ul + original_size)
                           for (x_ul, y_ul) in positions]
        label_positions = np.asarray(label_positions).flatten().astype(np.uint8)
        labels.append(label_positions)

        # Generate the frames
        for frame_idx in range(num_frames):

            canvases = [Image.new('L', (width, height)) for _ in range(nums_per_image)]
            canvas = np.zeros((1, width, height), dtype=np.float32)

            for i, canv in enumerate(canvases):
                # In canv (an Image object), place the image at the respective positions
                canv.paste(mnist_images[i], tuple(positions[i].astype(int)))
                # Superimpose both images on the canvas (i.e., empty np-array)
                canvas += arr_from_img(canv, mean=0)

            # Get the next position by adding velocity
            next_pos = positions + veloc

            # Iterate over velocity and see if we hit the wall
            # If we do then change the  (change direction)
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j] + 2:
                        # One of list(veloc[i][:j]) or list(veloc[i][j + 1:])
                        # always gives an empty list [].
                        # Whereas [-1 * veloc[i][j]] reverses that component.
                        # list(list + list) is just concatenating lists.
                        veloc[i] = list(
                            list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:])
                        )

            # Make the permanent change to position by adding updated velocity
            positions = positions + veloc

            # Add the canvas to the dataset array
            dataset[img_idx * num_frames + frame_idx] = (canvas * 255).clip(0, 255).astype(np.uint8)

    col_headers = ['ul_x_0', 'ul_y_0', 'lr_x_0', 'lr_y_0',
                   'ul_x_1', 'ul_y_1', 'lr_x_1', 'lr_y_1',
                   'ul_x_2', 'ul_y_2', 'lr_x_2', 'lr_y_2']
    labels_df = pd.DataFrame(labels, columns=col_headers)

    return dataset, labels_df


def main(training, dest, filetype='jpg', frame_size=64, num_frames=30, num_sequences=2,
         original_size=28, nums_per_image=3, save_gifs=True):
    dat, labels_df = generate_temporal_shape_dataset(training, shape=(frame_size, frame_size), num_frames=num_frames,
                                          num_sequences=num_sequences, original_size=original_size,
                                          nums_per_image=nums_per_image)
    labels_df.to_csv(os.path.join(dest, 'labels.csv'))
    if filetype == 'npz':
        np.savez(dest, dat)
    elif filetype == 'jpg':
        print('Saving jpgs...')
        for i in tqdm(range(dat.shape[0])):
            Image.fromarray(get_image_from_array(dat, i, mean=0)).save(os.path.join(dest, '{}.jpg'.format(i)))
    if save_gifs:
        print('Saving gifs...')
        for i in tqdm(range(num_sequences)):
            if i > 100:
                break
            start_index = i * num_frames
            images_for_gif = [Image.fromarray(get_image_from_array(dat, j, mean=0)).convert('P') for j in
                              range(start_index, start_index + num_frames)]
            images_for_gif[0].save(os.path.join(dest, f'seq_{i}_start_{start_index}.gif'),
                                   save_all=True, append_images=images_for_gif[1:],
                                   include_color_table=False, optimize=False, duration=120)


if __name__ == '__main__':
    num_frames = 20
    num_sequences = 100000
    train_test = 'train'

    train = True if train_test == 'train' else False
    dest = f'../../data/regression_{train_test}_{num_sequences}seqs_{num_frames}_per_seq/'
    if not os.path.isdir(dest):
        os.mkdir(dest)
    main(training=train, dest=dest, num_frames=num_frames, num_sequences=num_sequences, save_gifs=True)

