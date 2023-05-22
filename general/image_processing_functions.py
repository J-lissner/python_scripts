import numpy as np
from math import ceil, floor
from numpy.fft import fftn, ifftn
from matplotlib.image import imread, imsave



def pooling( image, kernel_size, pool_type='avg'):
    """
    sum up local windows of a 2d-array (without overlap) and return it as 
    a smaller grid. Will always assert the same stride as kernel_size
    Parameters:
    -----------
    image:          numpy 2d-array
                    image data
    kernel_size:    int or tuple of ints
                    what window size it should consider in each step
    Returns:
    --------
    feature_map:    numpy 2d-array
                    feature map of size ceil( image.shape/ kernel_size )
    """
    size = np.ceil( np.array(image.shape) / np.array( kernel_size) ).astype(int)
    feature = np.zeros( size)
    if isinstance( kernel_size, int):
        kernel_size = image.ndim*[ kernel_size] 
    for i in range( size[0] ):
        column = image[i*kernel_size[0]:(i+1)*kernel_size[0]]
        for j in range( size[1]):
            box = column[:,j*kernel_size[1]:(j+1)*kernel_size[1]]
            if 'avg' in pool_type:
                feature[i,j] = column[:,j*kernel_size[1]:(j+1)*kernel_size[1]].mean()
            elif 'max' in pool_type:
                feature[i,j] = column[:,j*kernel_size[1]:(j+1)*kernel_size[1]].max()
    return feature


def embed_kernel( kernel, image_size ):
    """
    Embeds the kernel in an image of size "image_size" and centers it
    periodically in the top left corner that the center of "kernel" is 
    positioned at the 0th index of the returned "embedded_kernel"
    Already flips the kernel for further convolution processing
    Parameters:
    -----------
    kernel:     numpy nd-array or nested list
                array containing the kernel in original form
                the kernel must be smaller or equal to "image_size"
    image_size: list like of ints
                size of the resulting embedded_kernel
    Returns:
    --------
    embedded_kernel:    numpy nd-array
                        periodically embedded and flipped kernel
                        of embedded_kernel.shape == image_size
    """
    kernel = np.array( kernel)
    if ( np.array( kernel.shape) > np.array( image_size) ).any():
        raise Exception( 'Specified image size too small, terminating program') 
    kernel_shift = np.array( [ floor( x/2) for x in kernel.shape ] )
    top_left = tuple( [ slice( x) for x in kernel.shape ] )
    embedded_kernel = np.zeros( image_size)
    embedded_kernel[ top_left] = np.flip( kernel )
    embedded_kernel = np.roll( embedded_kernel, -kernel_shift, axis=np.arange(kernel.ndim) )
    return embedded_kernel


def apply_filter( image, kernel):
    """
    Convolute a given image with a given kernel. The size of image and kernel do not need to match
    Only applicable for periodic images, using the convolution in fourier space
    The function assumes that the image and the kernel are real valued,
    CAREFUL: wrong results will be returned for non-real valued inputs
    Parameters:
    -----------
    image:      numpy nd-array
                periodic image to apply the convolution to
    kernel:     numpy nd-array or nested list
                kernel to convolute the image with
    Returns:
    --------
    convoluted: numpy nd-array
                convolution of image with kernel ( image {*} kernel)
    """ 
    f_image  = fftn ( image )
    f_kernel = fftn( embed_kernel( kernel, image.shape ) )
    return ( ifftn( f_image * f_kernel) ).real


def image_clipping( image, theta, kind='upper'):
    """
    element wise "clipping" of image data by the treshold 'theta'
    Does clip every value upper/lower than 'theta' to 'theta'
    Parameters:
    -----------
    image:  numpy array
            image data
    theta:  float (int)
            Threshold parameter 
    kind:   string (upper/lower)
            Specifies the upper or lower clipping
            elaboration: 'upper' -> np.max( image)= theta, 'lower' -> np.min( image)=theta
    Returns:
    --------
    image:  numpy array
            clipped image data, does return a copy
    """
    if kind=='upper':
        image = np.maximum( image, theta )
    else:
        image = np.minimum( image, theta )
    return image

def binary_segmentation( image, theta):
    """ 
    threshold a grayscale image and return a binary image, where values
    larger than theta are set to 1, and smaller to 0
    """
    segmented = np.zeros(image.shape)
    segmented[ image >= theta ] = 1
    return segmented

def binary_dilation( image, kernel):
    """ Dilate a binary image by a given kernel.  """
    image = apply_filter( image, kernel) #convolution of the image with the kernel
    image = binary_segmentation( image, 0.9999*np.min( kernel[kernel!= 0]) ) #segment the convolved image at the right value 
    # For the binary_dilation, we want to keep every value in "image" which is non zero
    # be careful with round off errors!! 
    return image


def binary_erosion( image, kernel ):
    image = apply_filter( image, kernel) #convolution of the image with the kernel
    image = binary_segmentation( image, 0.9999*kernel.sum() )  #cut off the convolved image at the right value using binary_segmentation
    # For the binary_erosion, we want to keep every value in which the kernel fully fits
    # -> threshold is the sum of the kernel (multiplied by a correction factor like e.g. 0.99999 )
    return image
    

def binary_opening( image, kernel, n=1):
    """ apply erosion and a consecutive dilation using the same kernel """
    for i in range( n):
        image = binary_erosion( image, kernel)
    for i in range( n):
        image = binary_dilation( image, kernel)
    return image


def binary_closing( image, kernel, n=1):
    """ apply dilation and a consecutive erosion using the same kernel """
    for i in range( n):
        image = binary_dilation( image, kernel)
    for i in range( n):
        image = binary_erosion( image, kernel)
    return image

#### loading and conversion of images 
def rgb_to_grayscale( image, coefficients='CCIR'):
    """ convert a 3 channel image (rgb) to single channel image (grayscale)"""
    if coefficients == 'CCIR': #also called REC601
        image = 0.299*image[:,:,0] + 0.587* image[:,:,1] + 0.114*image[:,:,2]
    elif coefficients == 'ITU-R': #also called rec 709
        image = 0.2125*image[:,:,0] + 0.7154* image[:,:,1] + 0.0721 *image[:,:,2]
    elif coefficients == 'SMPTE':
        image = 0.212*image[:,:,0] + 0.701* image[:,:,1] + 0.087 *image[:,:,2]
    return  image 



def float_to_u8( image):
    """
    converts any float image to a u1 scale
    assumes that the maximum amplitude in 'image' is "white" (255) and the lowest is black (0)
    """
    image = image - image.min() 
    image = (image / image.max() * 255).astype('u1')
    return image


def u8_to_float( image):
    """
    converts any float image to a u1 scale
    assumes that the maximum amplitude in 'image' is "white" (255) and the lowest is black (0)
    """
    return image.astype(float) /255


def load_grayscale( filename):
    """
    loads an image with matplotlib.image.imread, which returns a (x,x,3) float array
    this function converts it to u1
    """
    image = imread( filename )
    image = (255*image).astype('u1')
    return rgb_to_grayscale( image )


def load_rgb( filename, *args, **kwargs):
    """
    loads an image with matplotlib.image.imread, which returns a (x,x,3) float array
    this function converts it to u1
    """
    image = imread( filename, *args, **kwargs)
    image = (255*image).astype('u1')
    return image


