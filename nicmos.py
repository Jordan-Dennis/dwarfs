import jax.numpy as np
import equinox as eqx
import typing as t


Matrix = t.TypeVar("Matrix")


_pixels = lambda npix : np.arange(npix) - npix / 2 + .5
_cartesian = lambda npix : np.array(np.meshgrid(_pixels(npix), _pixels(npix)))
_radial = lambda npix : (_cartesian(npix) ** 2).sum(axis = 0)


def nicmos(npix: int, radius: int) -> Matrix:
    pixel_scale = 2 * radius / npix 
    cartesian = _cartesian(npix) * pixel_scale
    radial = _radial(npix) * pixel_scale
    
    nicmos = (radial <= radius) && (radial >= .33)
    nicmos = nicmos * (cartesian > .011 * radius).sum(axis = 0)
    
    # TODO: Work out what the fuck this is for.
    temporary = np.roll(radial, int(.89221 * radius), axis=1)
    nicmos = nicmos * (temporary >= .065 * radius)
    
    temporary = np.roll(
        np.roll(radial, int(.7555 * radius), axis=0), 
        int(-.4615 * radius), axis=1)
    nicmos = nicmos * (temporary >= .065 * radius)
    
