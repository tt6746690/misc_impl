import jax
import jax.numpy as np

color_im_mean = (0.485, 0.456, 0.406)
color_im_std  = (0.229, 0.224, 0.225)


def pil_to_ndarray(im):
    """ PIL image with `uint8` to np.ndarray(np.float32) """
    import PIL
    if not isinstance(im, PIL.Image.Image):
        raise ValueError('`im` not valid PIL.Image')
    im = np.asarray(im) / 255.
    return im


@jax.jit
def normalize(im, mean=color_im_mean, std=color_im_std):
    """ Assumes `im` has dimension (H, W, C) with dtype of `np.uint8` """
    if im.ndim != 3 or im.dtype != np.float32:
        raise ValueError('im not valid `np.ndarray(np.float32)`')
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (im - mean) / std


@jax.jit
def normalize_undo(im, mean=color_im_mean, std=color_im_std):
    """ Un-normalize by `mean` & `std` but also  normalize 
        s.t. `im` has range of [0, 1] for visualization purposes """
    if im.ndim != 3 or im.dtype != np.float32:
        raise ValueError('im not valid `np.ndarray(np.float32)`')
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return im*std + mean
