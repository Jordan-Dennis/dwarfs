import numpy as np
import matplotlib.pyplot as pyplot 
from scipy.ndimage import rotate

def _xyic(ys, xs, between_pix=False):
    ''' --------------------------------------------------------------
    Private utility: returns two arrays of (y, x) image coordinates
    Array values give the pixel coordinates relative to the center of
    the array, that can be offset by 0.5 pixel if between_pix is set
    to True
    Parameters:
    ----------
    - ys: (integer) vertical size of the array (in pixels)
    - xs: (integer) horizontal size of the array (in pixels)
    - between_pix: (boolean) places center of array between pixels
    Returns:
    -------
    A tuple of two arrays: (yy, xx) ... IN THAT ORDER!
    -------------------------------------------------------------- '''
    offset = 0
    if between_pix is True:
        offset = 0.5
    xx = np.outer(np.ones(ys), np.arange(xs)-xs//2+offset)
    yy = np.outer(np.arange(ys)-ys//2+offset, np.ones(xs))
    return (yy, xx)


def HST_NIC1(PSZ, rad, between_pix=True, ang=0):
    ''' ---------------------------------------------------------
    returns an array that draws the pupil of HST/NICMOS1 camera
    Parameters:
    ----------
    - PSZ:     size of the array (assumed to be square)
    - rad:     radius of the standard pupil (in pixels)
    - between_pix: flag
    - ang:     global rotation of the pupil (in degrees)
    Remarks:
    -------
    This fairly complex procedure attempts to reproduce the
    features of the telescope and that of the cold mask inside
    NICMOS, including the misalignment between the two.
    This was used in the following publication:
    https://ui.adsabs.harvard.edu/abs/2020A&A...636A..72M/abstract
    --------------------------------------------------------- '''
    yy, xx = _xyic(PSZ, PSZ, between_pix=between_pix)
    mydist = np.hypot(yy, xx)

    pyplot.imshow(mydist)
    pyplot.colorbar()
    pyplot.show()

    NCM = np.zeros_like(mydist)  # nicmos cold mask

    # --------------------------------
    # OTA: Optical Telescope Assembly
    # --------------------------------
    OTA = np.zeros_like(mydist)
    OTA[mydist <= 1.000 * rad] = 1.0     # outer radius
    OTA[mydist <= 0.330 * rad] = 0.0     # telescope obstruction
    OTA[np.abs(xx) < 0.011 * rad] = 0.0  # spiders
    OTA[np.abs(yy) < 0.011 * rad] = 0.0  # spiders

    tmp = np.roll(mydist, int(0.8921 * rad), axis=1)

    pyplot.imshow(tmp)
    pyplot.show()

    OTA[tmp <= 0.065 * rad] = 0.0        # mirror pad

    tmp = np.roll(
        np.roll(mydist, int(0.7555 * rad), axis=0),
        int(-0.4615 * rad), axis=1)
    OTA[tmp <= 0.065 * rad] = 0.0        # mirror pad

    tmp = np.roll(
        np.roll(mydist, int(-0.7606 * rad), axis=0),
        int(-0.4564 * rad), axis=1)
    OTA[tmp <= 0.065 * rad] = 0.0        # mirror pad

    pyplot.imshow(OTA)
    pyplot.show()

    # --------------------------------
    # NCM: NICMOS COLD MASK
    # --------------------------------
    NCM = np.zeros_like(mydist)           # nicmos cold mask
    NCM[mydist <= 0.955 * rad] = 1.0      # outer radius
    NCM[mydist <= 0.372 * rad] = 0.0      # obstruction 0.372
    NCM[np.abs(xx) < 0.0335 * rad] = 0.0  # fat spiders
    NCM[np.abs(yy) < 0.0335 * rad] = 0.0  # fat spiders

    # PADS
    cpadr = 0.065
    NCM[(xx >= (0.8921-cpadr)*rad) * (np.abs(yy) <= cpadr*rad)] = 0.0
    xx1 = rotate(xx, 121, order=0, reshape=False)
    yy1 = rotate(yy, 121, order=0, reshape=False)
    NCM[(xx1 >= (0.8921-cpadr)*rad) * (np.abs(yy1) <= cpadr*rad)] = 0.0
    xx1 = rotate(xx, -121.5, order=0, reshape=False)
    yy1 = rotate(yy, -121.5, order=0, reshape=False)
    NCM[(xx1 >= (0.8921-cpadr)*rad) * (np.abs(yy1) <= cpadr*rad)] = 0.0

    NCM = np.roll(
        np.roll(NCM, int(-0.0 * rad), axis=1),
        int(-0.08 * rad), axis=0)  # MASK SHIFT !!

    pyplot.imshow(NCM)
    pyplot.show()

    res = 1.0 * (OTA * NCM)

    if ang != 0:
        res = rotate(res, ang, order=0, reshape=False)
    return res


mask = HST_NIC1(1024, 512)
pyplot.imshow(mask)
pyplot.show()
