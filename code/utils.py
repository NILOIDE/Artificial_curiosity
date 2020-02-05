import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

INPUT_X = 84
INPUT_Y = 84
INPUT_C = 3
INPUT_DIM = (INPUT_C, INPUT_X, INPUT_Y)
INPUT_INTERPOLATION = cv2.INTER_NEAREST


def resize_torch(im: torch.Tensor,  dims:tuple) -> torch.Tensor:
    p = transforms.Compose([transforms.Scale(dims)])
    return p(im)


def resize_numpy(im: np.ndarray,  dims: tuple) -> np.ndarray:
    if len(dims) == 2:
        pass
    elif len(dims) == 3:
        if dims[0] == INPUT_C:
            dims = dims[1:]
        elif dims[2] == INPUT_C:
            dims = dims[:2]
        else:
            raise ValueError("Wrong target dimension channel format.")
    else:
        raise ValueError("Wrong target dimension vector length.")
    return cv2.resize(im, dims, interpolation=INPUT_INTERPOLATION)


def resize_to_standard_dim_numpy(im: np.ndarray) -> np.ndarray:
    im = im.astype(np.float)
    return cv2.resize(im, (INPUT_Y, INPUT_Y), interpolation=INPUT_INTERPOLATION)


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
    assert type(im) == np.ndarray
    assert len(tuple(im.shape)) == 3
    assert im.shape[0] == INPUT_C or im.shape[2] == INPUT_C
    if im.shape[0] == INPUT_C:
        raise ValueError("Image already in channel first form")
    return np.transpose(im, (2, 0, 1))


def channel_last_numpy(im: np.ndarray) -> np.ndarray:
    assert type(im) == np.ndarray
    assert len(tuple(im.shape)) == 3
    assert im.shape[0] == INPUT_C or im.shape[2] == INPUT_C
    if im.shape[2] == INPUT_C:
        raise ValueError("Image already in channel last form")
    return np.transpose(im, (1, 2, 0))


def clip_rewards(r_t):
    return np.clip(r_t, -1.0, 1.0)


def standardize_state(s_t, grayscale=False):
    s_t = resize_to_standard_dim_numpy(s_t)
    s_t = channel_first_numpy(s_t)
    if grayscale:
        s_t = rgb_to_gray(s_t)
    s_t /= 256
    return s_t


def transition_to_torch(s_t, a_t, r_t, s_tp1, d_t):
    # type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray) -> tuple
    s_t = torch.from_numpy(s_t).to(dtype=torch.float32)
    a_t = torch.from_numpy(a_t).to(dtype=torch.long)
    r_t = torch.from_numpy(r_t).to(dtype=torch.float32)
    s_tp1 = torch.from_numpy(s_tp1).to(dtype=torch.float32)
    d_t = torch.from_numpy(d_t).to(dtype=torch.int8)
    return s_t, a_t, r_t, s_tp1, d_t

