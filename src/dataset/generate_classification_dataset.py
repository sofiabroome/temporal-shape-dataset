import math
import os

from PIL import Image
from tqdm import tqdm
from enum import Enum
import pandas as pd
import numpy as np
import argparse
import random

from perlin_noise import PerlinNoise

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


def random_flip(array):
    draw = random.uniform(0, 1)
    if draw > 0.5:
        array = np.flip(array)
    return array


def generate_circle(time_steps, r):
    angles = np.linspace(0, 2*np.pi, time_steps)
    angles = random_flip(angles)
    angles = np.roll(angles, shift=np.random.randint(time_steps))
    velocities = [np.array([r*np.cos(phi), r*np.sin(phi)])
                  for phi in angles]
    return velocities


def generate_arc(time_steps, r):
    random_start_angle = random.uniform(0, 2*np.pi)
    angles = np.linspace(random_start_angle, random_start_angle + np.pi, time_steps)
    angles = random_flip(angles)
    velocities = [np.array([r*np.cos(phi), r*np.sin(phi)])
                  for phi in angles]
    return velocities


def generate_spiral(time_steps, max_radius):
    t_array = np.linspace(0, max_radius, time_steps)
    t_array = random_flip(t_array)
    velocities = [np.array([0.5*t*np.cos(t), 0.5*t*np.sin(t)])
                  for t in t_array]
    return velocities


def generate_line(speed, time_steps):
    # Randomly generate direction, speed and velocity
    direc = np.pi * (np.random.rand() * 2 - 1)
    veloc = np.asarray((speed * math.cos(direc), speed * math.sin(direc)))
    velocities = [veloc for i in range(time_steps)]
    return velocities


def generate_rectangle(speed, time_steps):
    nb_sides = 4
    direcs = np.asarray([np.pi/2, np.pi, -np.pi/2, 0])
    direcs = random_flip(direcs)
    direcs = np.roll(direcs, shift=np.random.randint(time_steps))
    velocs = [np.asarray((speed * math.cos(direc), speed * math.sin(direc))) for direc in direcs]
    interval_length = int(time_steps/nb_sides)
    velocities = []
    for i in range(nb_sides):
        velocities.extend([velocs[i] for _ in range(interval_length)])
    return velocities


def generate_textured_background(canvas, octaves):
    noise = PerlinNoise(octaves=octaves)
    xpix = canvas.shape[1]
    ypix = canvas.shape[2]
    canvas_with_perlin_background = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]
    return np.asarray(canvas_with_perlin_background).reshape(1, xpix, ypix)


def get_starting_point_limits(image_size, symbol_width, max_radius):
    low = 2 * max_radius + symbol_width
    high = image_size - low
    return low, high


def generate_temporal_shape_dataset(training, shape, num_frames, num_sequences,
                                    symbol_size, nums_per_image, object_mode,
                                    textured_background, save_gifs=True):
    """
    Args:
        training: Boolean, used to decide if downloading/generating train set or test set
        shape: Shape we want for our moving images (new_width and new_height)
        num_frames: Number of frames in a particular movement/animation/gif
        num_sequences: Number of movement/animations/gif to generate
        symbol_size: Real size of the images (eg: MNIST is 28x28)
        nums_per_image: Digits per movement/animation/gif.
    Returns:
        Dataset of np.uint8 type with dimensions
        num_frames * num_sequences x 1 x new_width x new_height
    """
    width, height = shape

    labels = []
    if object_mode == 'dot':
        max_radius = 2 + symbol_size

    if object_mode == 'mnist':
        mnist = load_mnist(training)
        max_radius = 2

    # Get how many pixels can we move around a single image (to fit its width)
    wall_lims = width - symbol_size, height - symbol_size
    low, high = get_starting_point_limits(width, symbol_size, max_radius)
    print(low, high)
    print('wall_lims: ', wall_lims, '\n')

    gif_counter = 0

    print('Generating sequences...')
    for img_idx in tqdm(range(num_sequences)):
        max_radius = 2
        # Create an array of shape num_sequences x 1 x new_width x new_height
        sequence = np.empty((num_frames, 1, width, height), dtype=np.uint8)
        if textured_background:
            octaves = np.random.randint(1, 10)
        label = np.random.randint(num_classes)
        labels.append(label)

        if TemporalShape(label).name == 'CIRCLE':
            velocities = generate_circle(time_steps=num_frames, r=max_radius)
            max_radius = 8
            low, high = get_starting_point_limits(width, symbol_size, max_radius=max_radius)
            print('Circle, adjusted: ')
            print(low, high, '\n')

        if TemporalShape(label).name == 'ARC':
            max_radius = 2
            velocities = generate_arc(time_steps=num_frames, r=max_radius)
            max_radius = 8
            low, high = get_starting_point_limits(width, symbol_size, max_radius=max_radius)
            print('Arc, adjusted: ')
            print(low, high, '\n')

        if TemporalShape(label).name == 'SPIRAL':
            max_radius = 10
            if object_mode == 'mnist':
                max_radius = 8
            velocities = generate_spiral(time_steps=num_frames, max_radius=max_radius)
            low, high = get_starting_point_limits(width, symbol_size, max_radius=max_radius)
            print('Spiral, adjusted: ')
            print(low, high, '\n')

        if TemporalShape(label).name == 'LINE':
            speed = np.random.randint(1, 4)
            max_radius = speed * max_radius
            low, high = get_starting_point_limits(width, symbol_size, max_radius=max_radius)
            velocities = generate_line(speed=speed, time_steps=num_frames)

        if TemporalShape(label).name == 'RECTANGLE':
            speed = np.random.randint(1, 4)
            max_radius = speed * max_radius
            low, high = get_starting_point_limits(width, symbol_size, max_radius=max_radius)
            velocities = generate_rectangle(speed=speed, time_steps=num_frames)

        if object_mode == 'mnist':
            # Get a list containing three PIL images randomly sampled from the database
            mnist_images = [Image.fromarray(get_image_from_array(
                mnist, r, allow_nuances=False, mean=0)).resize(
                (symbol_size, symbol_size), Image.ANTIALIAS)
                for r in np.random.randint(0, mnist.shape[0], nums_per_image)]

        # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
        positions = np.asarray((np.random.randint(low, high),
                                np.random.randint(low, high)))
        # print(positions, '\n')

        # Generate the frames
        for frame_idx in range(num_frames):

            canvases = [Image.new('L', (width, height)) for _ in range(nums_per_image)]
            canvas = np.zeros((1, width, height), dtype=np.float32)

            if textured_background:
                canvas = generate_textured_background(canvas, octaves=octaves)

            for i, canv in enumerate(canvases):
                # In canv (an Image object), place the image at the respective positions
                if object_mode == 'mnist':
                    canv.paste(mnist_images[i], tuple(positions.astype(int)))
                    # Superimpose both images on the canvas (i.e., empty np-array)
                    canvas += arr_from_img(canv, mean=0)
                if object_mode == 'dot':
                    x_pos = int(positions[0])
                    y_pos = int(positions[1])
                    dot_radius = int(symbol_size / 2)
                    canvas[0,
                           x_pos-dot_radius:x_pos+dot_radius+1,
                           y_pos-dot_radius:y_pos+dot_radius+1] = 255

            # Get the next position by adding velocity
            next_pos = positions + velocities[frame_idx]
            # print('next pos: ', next_pos, '\n')

            # Iterate over velocity and see if we hit the wall
            # If we do then change the  (change direction)
            # if coord < -2 or coord > wall_lims[j] + 2:
            for j, coord in enumerate(next_pos):
                if coord < 0 or coord > wall_lims[1]:
                    # One of list(veloc[i][:j]) or list(veloc[i][j + 1:])
                    # always gives an empty list [].
                    # Whereas [-1 * veloc[i][j]] reverses that component.
                    # list(list + list) is just concatenating lists.
                    future_velocities = [list(velocities[ind][:j]) + [-1 * velocities[ind][j]] + list(velocities[ind][j + 1:])
                        for ind in range(frame_idx, num_frames)]
                    velocities[frame_idx:] = future_velocities

            # Make the permanent change to position by adding updated velocity
            positions = positions + velocities[frame_idx]

            # Add the canvas to the dataset array
            sequence[frame_idx] = (canvas * 255).clip(0, 255).astype(np.uint8)
            jpg_index = img_idx * num_frames + frame_idx
            image_to_save = Image.fromarray( get_image_from_array(
                sequence, frame_idx, allow_nuances=True, mean=0))
            image_to_save.save(os.path.join(dest, '{}.jpg'.format(jpg_index)))
        if save_gifs:
            print('Saving gifs...')
            if gif_counter > 100:
                continue
            start_index = img_idx * num_frames
            images_for_gif = [Image.fromarray(get_image_from_array(
                sequence, j, allow_nuances=True, mean=0)).convert('P') for j in
                              range(num_frames)]
            images_for_gif[0].save(os.path.join(dest, f'seq_{gif_counter}_start_{start_index}.gif'),
                                   save_all=True, append_images=images_for_gif[1:],
                                   include_color_table=False, optimize=False, duration=80)
            gif_counter += 1

    col_headers = ['class']
    labels_df = pd.DataFrame(labels, columns=col_headers)

    return labels_df


def main(training, dest, frame_size, num_frames, num_sequences,
         symbol_size, nums_per_image, object_mode,
         textured_background, save_gifs=True):
    labels_df = generate_temporal_shape_dataset(
        training, shape=(frame_size, frame_size), num_frames=num_frames,
        num_sequences=num_sequences, symbol_size=symbol_size,
        nums_per_image=nums_per_image, object_mode=object_mode,
        textured_background=textured_background, save_gifs=save_gifs)
    labels_df.to_csv(os.path.join(dest, 'labels.csv'))


if __name__ == '__main__':
    num_frames = 20
    num_classes = 5

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-sequences")
    parser.add_argument("--object-mode")
    parser.add_argument("--symbol-size")
    parser.add_argument("--train-test-mnist")
    parser.add_argument("--textured-background")
    args = parser.parse_args()

    num_sequences = int(args.num_sequences)
    object_mode = args.object_mode
    symbol_size = int(args.symbol_size)
    textured_background = int(args.textured_background)

    if object_mode == 'mnist':
        train_test_mnist = args.train_test_mnist
    else:
        train_test_mnist = ''

    train = True if train_test_mnist == 'train' else False

    dest = f'../../data/classification_{symbol_size}{object_mode}_{train_test_mnist}_bg{textured_background}_{num_classes}classes_{num_sequences}seqs_{num_frames}_per_seq/'

    if not os.path.isdir(dest):
        os.mkdir(dest)

    main(frame_size=64, nums_per_image=1, training=train, dest=dest, num_frames=num_frames, num_sequences=num_sequences,
         object_mode=object_mode, symbol_size=symbol_size, textured_background=textured_background, save_gifs=True)

