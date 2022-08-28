import jax.numpy as np
import equinox as eqx
import typing as t
from rotate import rotate
from dLux.utils import get_pixel_positions, get_polar_positions

Matrix = t.TypeVar("Matrix")

def nicmos(npix: int, radius: int) -> Matrix:
    """
    References:
    -----------
    1. Observatory - Optics, NASA, May 27, 2022,
       https://www.nasa.gov/content/goddard/
       hubble-space-telescope-optics-system
    """
    pixel_scale = 2 * radius / npix 
    cartesian = get_pixel_positions(npix, 0., 0.) * pixel_scale
    radial = (get_polar_positions(npix, 0., 0.) * pixel_scale)[0]
   
    # The optical telescope assembly refers to the optical system 
    # frame (NASA, 2022). 
    pad_radius = .065 * radius
    pad_1_centre = int(.89221 * npix / 2)
    pad_2_centre = int(.7555 * npix / 2), int(-.4615 * npix / 2)
    pad_3_centre = int(-.7606 * npix / 2), int(-.4564 * npix / 2)

    outer_radius = (radial <= radius)
    telescope_obstruction = (radial <= .33 * radius)
    horizontal_spider = (np.abs(cartesian[0]) < .011 * radius)
    vertical_spider = (np.abs(cartesian[1]) < .011 * radius)
    mirror_pad_1 = np.roll(radial, pad_1_centre, axis=1) <= pad_radius
    mirror_pad_2 = np.roll(radial, pad_2_centre, axis=(0, 1)) <= pad_radius
    mirror_pad_3 = np.roll(radial, pad_3_centre, axis=(0, 1)) <= pad_radius

    pyplot.imshow(np.roll(radial, pad_1_centre, axis=1))
    pyplot.show()

    pyplot.imshow(mirror_pad_1)
    pyplot.show()

    pyplot.imshow(mirror_pad_2)
    pyplot.show()

    pyplot.imshow(mirror_pad_3)
    pyplot.show()

    optical_telescope_assembly = np.zeros_like(radial)\
        .at[outer_radius].set(1.)\
        .at[telescope_obstruction].set(0.)\
        .at[horizontal_spider].set(0.)\
        .at[vertical_spider].set(0.)\
        .at[mirror_pad_1].set(0.)\
        .at[mirror_pad_2].set(0.)\
        .at[mirror_pad_3].set(0.)\

    pyplot.imshow(optical_telescope_assembly)
    pyplot.show()


    # TODO: Remember that I have not copied exactly and I am 
    # multiplying but I should be using .set so that I can have 
    # it exactly the same.
    # TODO: I can use boolean masking and make this exactly the 
    # same which will be the way that I do this moving forward.
    outer_radius = (radial <= .955 * radius)
    obstruction = (radial <= .372 * radius)
    vertical_spider = (np.abs(cartesian[0]) < .0335 * radius)
    horizontal_spider = (np.abs(cartesian[1]) < .0335 * radius)

    inner = (.065) * radial
    outer = (.8921 - .065) * radial
    
    mirror_pads = ((cartesian[0] >= outer) * (cartesian[1] <= inner)\
        & ((rotate(cartesian[0], 121.) >= outer) *\
            (rotate(cartesian[1], 121.) <= inner))\
        & ((rotate(cartesian[0], -121.5) >= outer) *\
            (rotate(cartesian[1], -121.5) <= inner)))
    
    rotated_right = rotate(cartesian[0], -121.5)
    rotated_right = rotate(cartesian[1], -121.5)
     
    nicmos_cold_mask = np.zeros_like(radial)\
        .at[outer_radius].set(1.)\
        .at[obstruction].set(0.)\
        .at[vertical_spider].set(0.)\
        .at[horizontal_spider].set(0.)\
        .at[mirror_pads].set(0.)

    # mask shift
    nicmos_cold_mask = np.roll(
        np.roll(nicmos_cold_mask, int(-.0 * radius), axis=1),
        int(-.08 * radius), axis=0)

    pyplot.imshow(nicmos_cold_mask)
    pyplot.show()

    return optical_telescope_assembly * nicmos_cold_mask

import matplotlib.pyplot as pyplot

mask = nicmos(1024, 1.)
pyplot.imshow(mask)
pyplot.show()
