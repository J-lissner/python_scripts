import time

initialized_times = dict()

def tic( tag='', silent=False):
    """
    initializes the tic timer
    different tags allow for tracking of different computations
    Parameters:
    -----------
    tag:        string, default ''
                name of tag to initialize the timer
    silent:     bool, default False
                Whether or not initialization should be printed
    """
    initialized_times[tag] = time.time()
    if not silent:
        print( 'Initializing timer for this tag:', tag)

def toc( tag='', precision=4, auxiliary='', *args, **kwargs ):
    """
    prints the time passed since the invocation of the tic tag 
    does not remove the tag on call, can be timed multiple times 
    since start
    Parameters:
    -----------
    tag:        string, default ''
                name of tag to initialize the timer
    precision:  int, default 4
                How many digits after ',' are printed
    auxiliary:  string, default ''
                what to add after the time statement, 
                allows for more consolue output
    """
    try:
        time_passed = time.time() - initialized_times[tag]
        print( '{1} -> elapsed time:{2: 0.{0}f}'.format( precision, tag, time_passed) + auxiliary )
    except:
        time_passed = -1.0
        print( 'tic( tag) not specified, command will be ignored!')
    return time_passed


class Timer:
    """
    used as: with Timer('tests'):
                 .... computations to be times
    initializes a timer which tracks the time required to do the indended computations
    input: tag - optional, tag of the computation
    """
    def __init__(self, tag=None):
        self.tag = tag

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.tag:
            print('%s -> ' % self.tag, end='')
        print('elapsed time: %.8f' % (time.time() - self.tstart))
