import numpy as np

def output_dimension( input_resolution, kernel_size, stride=1, channels=1, padding=True):
    """
    Compute the output dimension of a kernel with an image. For now it 
    is assumed that the kernels are quadratic and stride is equal in
    both directions. Padding is always as big as half the kernel. For 
    now it is also assumed that it is assumed that we are dealing with 
    2d convolutions. Also asserts that the convolution takes the sum of
    all input channels.
    Parameters:
    -----------
    input_resolution:   tuple of ints
                        dimension of the input image n_row, n_col, n_channels=1
    kernel_size:        int
                        size of the kernel
    stride:             int, default 1
                        stride of the kernel
    channels:           int, default 1
                        number of channels/kernels to convolute with
    padding:            bool, default True
                        whether or not padding is deployed
    Returns:
    --------
    output_dim:         array of 3 ints
                        n_row, n_col, n_channels
    """
    input_resolution = np.array( input_resolution)
    convolution_pad = kernel_size//2
    if padding is False:
        input_resolution[:2] = input_resolution[:2] - 2*convolution_pad
    output_resolution = input_resolution[:2] // stride
    if len( input_resolution) == 3 and channels == 1:
        output_resolution = np.hstack( (output_resolution, channels) )
    return output_resolution

if __name__ == '__main__':
    ke = [7,7,5,5]
    st = [2,2,2,2]
    output_dim = (400, 400)
    for k, s in zip( ke, st  ):

        output_dim  = output_dimension( output_dim, k, s)
    print( output_dim )
