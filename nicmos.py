import jax.numpy as np
import equinox as eqx
import typing as t


Matrix = t.TypeVar("Matrix")


_pixels = lambda npix : np.arange(npix) - npix / 2 + .5
_cartesian = lambda npix : np.array(np.meshgrid(_pixels(npix), _pixels(npix)))
_radial = lambda npix : (_cartesian(npix) ** 2).sum(axis = 0)


def nicmos(npix: int, radius: int) -> Matrix:
    """
    References:
    -----------
    1. Observatory - Optics, NASA, May 27, 2022,
       https://www.nasa.gov/content/goddard/
       hubble-space-telescope-optics-system
    """
    pixel_scale = 2 * radius / npix 
    cartesian = _cartesian(npix) * pixel_scale
    radial = _radial(npix) * pixel_scale
   
    # The optical telescope assembly refers to the optical system 
    # frame (NASA, 2022). 
    mirror_pad_edges = (
        (int(.89221 * radius),), 
        (int(.7555 * radius), int(-.4615 * radius)),
        (int(-.7606 * radius), int(-.4615 * radius)))
    mirror_pad_outer_edge = .065 * radius

    outer_radius = (radial <= radius)
    telescope_obstruction = (radial <= .33 * radius)
    horizontal_spider = (np.abs(cartesian[0]) < .011 * radius)
    vertical_spider = (np.abs(cartesian[1]) < .011 * radius)
    mirror_pad = (np.roll(radial, mirror_pad_edges[0], axis=(1,))\
            <= mirror_pad_outer_edge)\
        && (np.roll(radial, mirror_pad_edges[1], axis=(0, 1))\
            <= mirror_pad_outer_edge)\
        && (np.roll(radial, mirror_pad_edges[2], axis=(0, 1))\
            <= mirror_pad_outer_edge)

    optical_telescope_assembly = np.zeros_like(radial)\
        .at[outer_radius].set(1.)\
        .at[telescope_obstruction].set(0.)\
        .at[horizontal_spider].set(0.)\
        .at[vertical_spider].set(0.)\
        .at[mirror_pad].set(0.)


    # TODO: Remember that I have not copied exactly and I am 
    # multiplying but I should be using .set so that I can have 
    # it exactly the same.
    # TODO: I can use boolean masking and make this exactly the 
    # same which will be the way that I do this moving forward.
    outer_radius = (radial <= .955 * radius)
    obstruction = (radial <= .372 * radius)
    vertical_spider = (np.abs(cartesian[0]) < .0335 * radius)
    horizontal_spider = (np.abs(cartesian[1]) < .0335 * radius)
    mirror_pads = 

    nicmos_cold_mask =\
        (radial <= .955 * radius)\
        .at[:]\
        .mul(radial <= .372 * radius)\
        .at[:]\
        .mul(np.abs(cartesian[0]) > .0335 * radius)\
        .at[:]\
        .mul(np.abs(cartesian[1]) > .0335 * radius)\


    
    
