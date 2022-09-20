import jax.numpy as np
import equinox as eqx
import typing as t
from rotate import rotate
from dLux.utils import get_pixel_positions, get_polar_positions

Matrix = t.TypeVar("Matrix")


# TODO: Sought out the whole radius thing so that it is more idiomatic. 
# I think just remove it and padding can be done by another function. 
# This will probably reduce time ellapsed as well.
def nicmos(npix: int) -> Matrix:
    """
    Constructs a detailed model of the nicmos1 camera pupil on the hubble 
    space telescope. 

    Parameters
    ----------
    npix: int
        The number of pixels along one edge of the image that represents the 
        camera pupil.

    Returns
    -------
    nicmos: Matrix
        The pupil of the nicmos1 camera represented as a binary array.
    """
    pixel_scale = 2 / npix 
    cartesian = get_pixel_positions(npix, 0., 0.) * pixel_scale
    radial = (get_polar_positions(npix, 0., 0.) * pixel_scale)[0]
   
    pad_radius = .065 
    pad_1_centre = int(.89221 * npix / 2)
    pad_2_centre = int(.7555 * npix / 2), int(-.4615 * npix / 2)
    pad_3_centre = int(-.7606 * npix / 2), int(-.4564 * npix / 2)

    outer_radius = (radial <= 1)
    telescope_obstruction = (radial <= .33)
    horizontal_spider = (np.abs(cartesian[0]) < .011)
    vertical_spider = (np.abs(cartesian[1]) < .011)
    mirror_pad_1 = np.roll(radial, pad_1_centre, axis=1) <= pad_radius
    mirror_pad_2 = np.roll(radial, pad_2_centre, axis=(0, 1)) <= pad_radius
    mirror_pad_3 = np.roll(radial, pad_3_centre, axis=(0, 1)) <= pad_radius

    optical_telescope_assembly = np.zeros_like(radial)\
        .at[outer_radius].set(1.)\
        .at[telescope_obstruction].set(0.)\
        .at[horizontal_spider].set(0.)\
        .at[vertical_spider].set(0.)\
        .at[mirror_pad_1].set(0.)\
        .at[mirror_pad_2].set(0.)\
        .at[mirror_pad_3].set(0.)

    outer_radius = (radial <= .955)
    obstruction = (radial <= .372)
    vertical_spider = (np.abs(cartesian[0]) < .0335)
    horizontal_spider = (np.abs(cartesian[1]) < .0335)

    inner = .065  
    outer = .8921 - .065
    unrotated_x = cartesian[0]
    unrotated_y = np.abs(cartesian[1])
    rotated_right_x = rotate(cartesian[0], 121. * np.pi / 180)
    rotated_right_y = np.abs(rotate(cartesian[1], 121. * np.pi / 180))
    rotated_left_x = rotate(cartesian[0], -121.5 * np.pi / 180)
    rotated_left_y = np.abs(rotate(cartesian[1], -121.5 * np.pi / 180))
    
    mirror_pad_1 = (unrotated_x >= outer) * (unrotated_y <= inner)
    mirror_pad_2 = (rotated_right_x >= outer) * (rotated_right_y <= inner)
    mirror_pad_3 = (rotated_left_x >= outer) * (rotated_left_y <= inner)
 
    nicmos_cold_mask = np.zeros_like(radial)\
        .at[outer_radius].set(1.)\
        .at[obstruction].set(0.)\
        .at[vertical_spider].set(0.)\
        .at[horizontal_spider].set(0.)\
        .at[mirror_pad_1].set(0.)\
        .at[mirror_pad_2].set(0.)\
        .at[mirror_pad_3].set(0.)

    mask_shift = int(-.08 * npix / 2)
    nicmos_cold_mask = np.roll(nicmos_cold_mask, mask_shift, axis=0)

    return optical_telescope_assembly * nicmos_cold_mask

import matplotlib.pyplot as pyplot

mask = nicmos(1024)
pyplot.imshow(mask)
pyplot.show()
