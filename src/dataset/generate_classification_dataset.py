import math
import os

from PIL import Image
from tqdm import tqdm
from enum import Enum
import pandas as pd
import numpy as np

from generate_regression_dataset import arr_from_img, get_image_from_array, load_mnist

# Code adapted from the following gist by Praateek Mahajan:
# https://gist.github.com/praateekmahajan/b42ef0d295f528c986e2b3a0b31ec1fe


class TemporalShape(Enum):
    CIRCLE = 0
    SPIRAL = 1
    LINE = 2
    RECTANGLE = 3
    ARC = 4
    ZIGZAG = 5
    S = 6


def generate_circle(time_steps, r):

    angles = np.linspace(0, 2*np.pi, time_steps)
    velocities = [np.array([r*np.cos(phi), r*np.sin(phi)])
                  for phi in angles]
    # velocities = np.asarray(velocities)

    return velocities


def generate_spiral(time_steps, max_radius):
    t_array = np.linspace(0, 10, time_steps)
    velocities = [np.array([t*np.cos(t), t*np.sin(t)])
                  for t in t_array]
    # velocities = np.asarray(velocities)

    return velocities


def generate_line(time_steps):
    # Randomly generate direction, speed and velocity for both images
    direc = np.pi * (np.random.rand() * 2 - 1)  # Scalars, one per digit
    # speed = np.random.randint(5) + 2  # Scalars, one per digit
    speed = 1  # Scalars, one per digit
    # veloc is 2xnums_per_image (x and y component for velocity for each digit)
    veloc = np.asarray((speed * math.cos(direc), speed * math.sin(direc)))
    # velocities = np.tile(veloc, (time_steps, 1))
    velocities = [veloc for i in range(time_steps)]
    return velocities


def get_limits(lims, symbol_width, max_radius, shape=None):
    extra_margin = max_radius + 2
    low = 2 * max_radius + extra_margin
    high = lims[1] - symbol_width
    if shape == 'SPIRAL':
        low = 2 * max_radius # + extra_margin
        high = lims[1] - symbol_width - max_radius
    return low, high


def generate_temporal_shape_dataset(training, shape=(64, 64), num_frames=30, num_sequences=2,
                                    original_size=14, nums_per_image=1, traj_per_image=2,
                                    max_radius=2):
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
    low, high = get_limits(lims, original_size, max_radius)
    print(low, high)
    print('lims: ', lims, '\n')

    # Create a dataset of shape of num_frames * num_sequences x 1 x new_width x new_height
    # Eg : 3000000 x 1 x 64 x 64
    dataset = np.empty((num_frames * num_sequences, 1, width, height), dtype=np.uint8)

    print('Generating sequences...')
    for img_idx in tqdm(range(num_sequences)):
        # label = np.random.randint(len(TemporalShape))
        label = np.random.randint(3)
        # label = 1
        labels.append(label)

        if TemporalShape(label).name == 'CIRCLE':
            velocities = generate_circle(time_steps=num_frames, r=max_radius)

        if TemporalShape(label).name == 'SPIRAL':
            velocities = generate_spiral(time_steps=num_frames, max_radius=max_radius)
            low, high = get_limits(lims, original_size, max_radius=10, shape='SPIRAL')
            print('adjusted: ')
            print(low, high, '\n')

        if TemporalShape(label).name == 'LINE':
            velocities = generate_line(time_steps=num_frames)

        # Get a list containing three PIL images randomly sampled from the database
        mnist_images = [Image.fromarray(
            get_image_from_array(mnist, r, mean=0)).resize(
            (original_size, original_size), Image.ANTIALIAS)
            for r in np.random.randint(0, mnist.shape[0], nums_per_image)]
        # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
        # positions = np.asarray((np.random.rand() * lims[0], np.random.rand() * lims[1]))
        # import ipdb; ipdb.set_trace()
        positions = np.asarray((np.random.randint(low, high),
                                np.random.randint(low, high)))
        print(positions, '\n')

        # Generate the frames
        for frame_idx in range(num_frames):

            canvases = [Image.new('L', (width, height)) for _ in range(nums_per_image)]
            canvas = np.zeros((1, width, height), dtype=np.float32)

            for i, canv in enumerate(canvases):
                # In canv (an Image object), place the image at the respective positions
                # canv.paste(mnist_images[i], tuple(positions[i].astype(int)))
                canv.paste(mnist_images[i], tuple(positions.astype(int)))
                # Superimpose both images on the canvas (i.e., empty np-array)
                canvas += arr_from_img(canv, mean=0)

            # Get the next position by adding velocity
            next_pos = positions + velocities[frame_idx]

            # Iterate over velocity and see if we hit the wall
            # If we do then change the  (change direction)
            for j, coord in enumerate(next_pos):
                # if coord < -2 or coord > lims[j] + 2:
                if coord < -2 or coord > lims[j] + 2:
                    # One of list(veloc[i][:j]) or list(veloc[i][j + 1:])
                    # always gives an empty list [].
                    # Whereas [-1 * veloc[i][j]] reverses that component.
                    # list(list + list) is just concatenating lists.
                    future_velocities = [list(velocities[ind][:j]) + [-1 * velocities[ind][j]] + list(velocities[ind][j + 1:])
                        for ind in range(frame_idx+1, num_frames)]
                    velocities[frame_idx+1:] = future_velocities

            # Make the permanent change to position by adding updated velocity
            positions = positions + velocities[frame_idx]

            # Add the canvas to the dataset array
            dataset[img_idx * num_frames + frame_idx] = (canvas * 255).clip(0, 255).astype(np.uint8)

            col_headers = ['class']
            labels_df = pd.DataFrame(labels, columns=col_headers)

    return dataset, labels_df


def main(training, dest, filetype='jpg', frame_size=64, num_frames=30, num_sequences=2,
         original_size=14, nums_per_image=1, save_gifs=True):
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
                                   include_color_table=False, optimize=False, duration=80)


if __name__ == '__main__':
    num_frames = 30
    num_sequences = 10
    train_test = 'train'

    train = True if train_test == 'train' else False
    dest = f'../../data/classification_{train_test}_{num_sequences}seqs_{num_frames}_per_seq/'
    if not os.path.isdir(dest):
        os.mkdir(dest)
    main(training=train, dest=dest, num_frames=num_frames, num_sequences=num_sequences, save_gifs=True)

