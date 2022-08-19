import numpy as np
from scipy import ndimage
try:
    import pyfftw.interfaces.scipy_fftpack as fftpack
except:
    from scipy import fftpack

def fft_3shear_rotate_pad(in_frame, alpha, pad=4, return_full = False):
    """
    3 FFT shear based rotation, following Larkin et al 1997

    in_frame: the numpy array which has to be rotated
    alpha: the rotation alpha in degrees
    pad: the padding factor

    The following options were removed because they didn't work:
        x1,x2: the borders of the original image in x
        y1,y2: the borders of the original image in y

    One side effect of this program is that the image gains two columns and two rows.
    This is necessary to avoid problems with the different choice of centre between
    GRAPHIC and numpy

    Return the rotated array
    """
    
    #################################################
    # Check alpha validity and correcting if needed
    #################################################
    alpha=1.*alpha-360*np.floor(alpha/360)

    # We need to add some extra rows since np.rot90 has a different definition of the centre
    temp = np.zeros((in_frame.shape[0]+3,in_frame.shape[1]+3))+np.nan
    temp[1:in_frame.shape[0]+1,1:in_frame.shape[1]+1]=in_frame
    in_frame = temp
    
    pyplot.imshow(in_frame)
    pyplot.show()

    # FFT rotation only work in the -45:+45 range
    if alpha > 45 and alpha < 135:
        in_frame=np.rot90(in_frame, k=1)
        alpha_rad=-np.deg2rad(alpha-90)
        print("K = 1")
    elif alpha > 135 and alpha < 225:
        in_frame=np.rot90(in_frame, k=2)
        alpha_rad=-np.deg2rad(alpha-180)
        print("K = 2")
    elif alpha > 225 and alpha < 315:
        in_frame=np.rot90(in_frame, k=3)
        alpha_rad=-np.deg2rad(alpha-270)
        print("K = 3")
    else:
        alpha_rad=-np.deg2rad(alpha)
        print("K = 0")

    print("Rotation", alpha_rad) 
         # Remove one extra row
    in_frame = in_frame[:-1,:-1]

    ###################################
    # Preparing the frame for rotation
    ###################################

    # Calculate the position that the input array will be in the padded array to simplify
    #  some lines of code later 
    px1=np.int(((pad-1)/2.)*in_frame.shape[0])
    px2=np.int(((pad+1)/2.)*in_frame.shape[0])
    py1=np.int(((pad-1)/2.)*in_frame.shape[1])
    py2=np.int(((pad+1)/2.)*in_frame.shape[1])

    # Make the padded array 
    pad_frame=np.ones((in_frame.shape[0]*pad,in_frame.shape[1]*pad))*np.NaN
    pad_mask=np.ones((pad_frame.shape), dtype=bool)
    pad_frame[px1:px2,py1:py2]=in_frame
    pad_mask[px1:px2,py1:py2]=np.where(np.isnan(in_frame),True,False)
    
    # Rotate the mask, to know what part is actually the image
    pad_mask=ndimage.interpolation.rotate(pad_mask, np.rad2deg(-alpha_rad),
          reshape=False, order=0, mode='constant', cval=True, prefilter=False)

    # Replace part outside the image which are NaN by 0, and go into Fourier space.
    pad_frame=np.where(np.isnan(pad_frame),0.,pad_frame)

    pyplot.imshow(pad_frame)
    pyplot.show()


    ###############################
    # Rotation in Fourier space
    ###############################
    a=np.tan(alpha_rad/2.)
    b=-np.sin(alpha_rad)

    M=-2j*np.pi*np.ones(pad_frame.shape)
    N=fftpack.fftfreq(pad_frame.shape[0])

    X=np.arange(-pad_frame.shape[0]/2.,pad_frame.shape[0]/2.)#/pad_frame.shape[0]
    pad_x=fftpack.ifft((fftpack.fft(pad_frame, axis=0,overwrite_x=True).T*
        np.exp(a*((M*N).T*X).T)).T, axis=0,overwrite_x=True)
    pad_xy=fftpack.ifft(fftpack.fft(pad_x,axis=1,overwrite_x=True)*
        np.exp(b*(M*X).T*N), axis=1,overwrite_x=True)
    pad_xyx=fftpack.ifft((fftpack.fft(pad_xy, axis=0,overwrite_x=True).T*
        np.exp(a*((M*N).T*X).T)).T,axis=0,overwrite_x=True)


    pyplot.imshow(np.real(np.exp(a*((M*N).T*X).T)).T)
    pyplot.show()

    pyplot.imshow(np.real(np.exp(b*(M*X).T*N)))
    pyplot.show()

    # Go back to real space
    # Put back to NaN pixels outside the image.

    pad_xyx[pad_mask]=np.NaN


    if return_full:
        return np.real(pad_xyx).copy()
    else:
        return np.real(pad_xyx[px1:px2,py1:py2]).copy()

shape = (10, 10)
quadrant_1 = np.full(shape, 1., dtype=float)
quadrant_2 = np.full(shape, 2., dtype=float)
quadrant_3 = np.full(shape, 3., dtype=float)
quadrant_4 = np.full(shape, 4., dtype=float)

top = np.vstack([quadrant_2, quadrant_1])
bottom = np.vstack([quadrant_3, quadrant_4])
image = np.hstack([top, bottom])

import matplotlib.pyplot as pyplot

pyplot.imshow(image)
pyplot.show()

image = fft_3shear_rotate_pad(image, 45)

pyplot.imshow(image)
pyplot.show()

