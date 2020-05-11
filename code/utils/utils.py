import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os

INPUT_INTERPOLATION = cv2.INTER_NEAREST

CONV_LAYERS2014 = ({'channel_num': 32, 'kernel_size': 8, 'stride': 4, 'padding': 0},
                   {'channel_num': 64, 'kernel_size': 4, 'stride': 2, 'padding': 0})

CONV_LAYERS2015 = ({'channel_num': 32, 'kernel_size': 8, 'stride': 4, 'padding': 0},
                   {'channel_num': 64, 'kernel_size': 4, 'stride': 2, 'padding': 0},
                   {'channel_num': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0})


def resize_torch(im: torch.Tensor, dims: tuple) -> torch.Tensor:
    p = transforms.Compose([transforms.Scale(dims)])
    return p(im)


def resize_image_numpy(im: np.ndarray, dims: tuple) -> np.ndarray:
    assert len(dims) == 2, 'cv2 resize takes in resize dims as tuple of length 2 (no channel dim).'
    im = im.astype(np.float)
    return cv2.resize(im, dims, interpolation=INPUT_INTERPOLATION)


def rgb_to_gray(rgb, channels_first=True):
    if channels_first:
        r, g, b = rgb[0], rgb[1], rgb[2]
        gray_im = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray_im = gray_im.reshape((1, *rgb.shape[1:]))
    else:
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray_im = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray_im = gray_im.reshape((*rgb.shape[:2], 1))
    return gray_im


def channel_first_numpy(im: np.ndarray) -> np.ndarray:
    assert len(tuple(im.shape)) == 3
    return np.transpose(im, (2, 0, 1))


def channel_last_numpy(im: np.ndarray) -> np.ndarray:
    assert len(tuple(im.shape)) == 3
    assert im.shape[0] == 1 or im.shape[0] == 3, 'Expected image to be in channel first form'
    return np.transpose(im, (1, 2, 0))


def clip_rewards(r_t):
    return np.clip(r_t, -1.0, 1.0)


def standardize_state(s_t: np.ndarray, input_shape: tuple, grayscale=True) -> np.ndarray:
    s_t = resize_image_numpy(s_t, input_shape[1:])
    if not (s_t.shape[0] == input_shape[0] or s_t.shape[0] == input_shape[0]*3):
        s_t = channel_first_numpy(s_t)
    else:
        assert input_shape[0] == input_shape[0] or \
               input_shape[0] == input_shape[0]*3, 'input shape must be channels first for torch to be happy'
    if grayscale and s_t.shape[0] == 3:
        s_t = rgb_to_gray(s_t)
    s_t /= 256.0
    return s_t


def transition_to_torch_no_r(s_t, a_t, s_tp1, d_t):
    # type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray) -> tuple
    s_t = torch.from_numpy(s_t).to(dtype=torch.float32)
    a_t = torch.from_numpy(a_t).to(dtype=torch.long)
    s_tp1 = torch.from_numpy(s_tp1).to(dtype=torch.float32)
    d_t = torch.from_numpy(d_t).to(dtype=torch.int8)
    return s_t, a_t, s_tp1, d_t


def transition_to_torch(s_t, a_t, r_t, s_tp1, d_t):
    # type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray) -> tuple
    s_t = torch.from_numpy(s_t).to(dtype=torch.float32)
    a_t = torch.from_numpy(a_t).to(dtype=torch.long)
    r_t = torch.from_numpy(r_t).to(dtype=torch.float32)
    s_tp1 = torch.from_numpy(s_tp1).to(dtype=torch.float32)
    d_t = torch.from_numpy(d_t).to(dtype=torch.int8)
    return s_t, a_t, r_t, s_tp1, d_t


def items_to_torch(items):
    # type: (tuple) -> tuple
    """"
    Takes any amount of columns to be transitioned to torch.
    By default it makes them into float32, thus one must be careful if actions are to be used as indices later.
    """
    out = []
    for i in items:
        out.append(torch.from_numpy(i).to(dtype=torch.float32))
    return tuple(out)


def plot_list_in_dict(lists, x_min=0, x_interval=1, y_low=None, y_high=None,
                      show=False, path=None, legend_loc=None,
                      title=None, x_label=None, y_label=None):
    # type: (dict, int, int, float, float, bool, str, str, str, str, str) -> None
    fig = plt.figure()
    for i in lists:
        name, li = i, lists[i]
        x = np.arange(x_min, x_min + len(li) * x_interval, x_interval)
        plt.plot(x, li, label=name)
    plt.legend(loc=legend_loc)
    plt.ylim(y_low, y_high)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if path is not None:
        directory = '../plots/'
        for i, c in enumerate(reversed(path)):
            if c == '/':
                directory = path[:-i]
                break
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(path, format='png')
        except:
            print('Plot saving failed. Path:', path)
    if show:
        plt.show()
    plt.close(fig)
