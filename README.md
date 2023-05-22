# Python scripts

Contains some python scripts with some good all purpose functionality, suitable for the `$PYTHONPATH` <br>
Each function is documented, and function purpose is split into larger subfolders.
Some scripts do contain cross references (between folders), it is recommended to have each folder in your pythonpath, by e.g. adding these lines into your `~/.bashrc`

```
for dir in /path/to/repo/pythonscripts/*; do
    if [ -d "$dir" ]; then
        PYTHONPATH="$PYTHONPATH:$dir"
    fi
done
export PYTHONPATH
```



General content per folder:<br>
__plotting/__: (using matplotlib.pyplot)

    - general default settings for any plot, i.e. styles of scatter/lines, color cycler, fontsize etc.
    - function to create a plot which is always equally spaced and can be stretched
    - auxiliary functionalities to improve the optics of a plot, custom legend, custom label entries, etc.
    - generation of plots, e.g. colored R$^2$ plot, scatter error relations

__hdf5/__:

    - data loaders to access files and return the requested data upon indexing
    - functions to move around datasets in files, file layout is slightly specific

__general/__:

    - timers, cycler
    - some data processing (binning), and image processing (periodic convolution, erosion, etc.)

__tensorflow_functions/__:

    - a model baseclass implementing freezing functions, batched predictions
        also baseclasses for more complex model layouts, e.g. hybrid models, fully conv models
    - custom layers, conv layers which deploy periodic padding, custom SnE layer
    - custom learning rate schedule
    - custom losses
    - custom models for convolutional (2D & 3D), or dense neural networks
    - literature models using periodic padding (e.g. Resnet, InceptionNet, etc.)
    - functionality to store custom models created by user source code. 
        also a class to reassemble the stored models
    - data processing/augmentation and batching 

__tikz_drawing/__:

    - collection of functions to generate LaTeX tikzpicture via scripts, mostly neural networks



 
