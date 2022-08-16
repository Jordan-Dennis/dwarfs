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
    optical_telescope_assembly =\
        ((radial <= radius) && (radial >= .33))\
        .at[:]\
        .mul((cartesian > .011 * radius).sum(axis = 0))\
        .at[:]\
        .mul(np.roll(radial, int(.89221 * radius), axis=1)\
            >= .065 * radius)\
        .at[:]\
        .mul(np.roll(
            np.roll(radial, int(.7555 * radius), axis=0), 
            int(-.4615 * radius), axis=1) >= .065 * radius)

    # TODO: Remember that I have not copied exactly and I am 
    # multiplying but I should be using .set so that I can have 
    # it exactly the same.
    # TODO: I can use boolean masking and make this exactly the 
    # same which will be the way that I do this moving forward.
    nicmos_cold_mask =\
        (radial <= .955 * radius)\
        .at[:]\
        .mul(radial <= .372 * radius)\
        .at[:]\
        .mul(np.abs(cartesian[0]) > .0335 * radius)\
        .at[:]\
        .mul(np.abs(cartesian[1]) > .0335 * radius)\


    
    
