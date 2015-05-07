# nervana_theano
A rudimentary wrapper around the fast Maxwell kernels for GEMM and convolution operations provided by [nervanagpu](https://github.com/nervanasystems/nervanagpu)

This is a work in progress, help is welcome! (see below)

Note that the [Theano](https://github.com/Theano/Theano) team at the LISA lab are separately working on integrating these kernels into the library themselves. However, they are focusing their efforts on the half-precision (fp16) kernels for now, and they are implementing half precision support (and support for these kernels) using the new Theano backend. I implemented these wrappers because I wanted something that was readily usable using the current Theano backend, so I've only wrapped the single precision (fp32) kernels.

## Installation
check out this repository and add the `nervana_theano` module to your Python path.
You will need [Theano](https://github.com/Theano/Theano) (0.7 or the latest version from git) and [nervanagpu](https://github.com/nervanasystems/nervanagpu).

You will need a NVIDIA Maxwell GPU for this code to run. These include all 900 series cards, and a select number of 700 series cards (as well as 800M series cards). However, the most popular 700 series cards like the 780Ti and the Titan are Kepler-based cards, so this code will not run on them.

If you wish to use the provided layer classes for [Lasagne](https://github.com/Lasagne/Lasagne), you will need to install that as well.

## Usage
### with Theano
You can use the gemm kernel in Theano as follows:
```
import theano.tensor as T
from nervana_theano.gemm import nervana_dot
x, y = T.matrices('x', 'y')
prod_cublas = T.dot(x, y)
prod_nervana = nervana_dot(x, y)
# these should give the same result
```

The Nervana convolution kernels support 1D, 2D and 3D convolutions. They use **c01b** or **batch-size-last** axis ordering, like the cuda-convnet kernels. This is different form Theano's default **bc01** or **batch-size-first** ordering.

The `nervana_conv` wrapper function adds the necessary dimshuffles by default, so you can use it as a drop-in replacement for `theano.tensor.nnet.conv.conv2d`. However, this may degrade performance, so this behaviour can be turned off by passing the keyword argument `dimshuffle=False`. In that case, you will need to provide inputs with axes in **c01b** order.

You can use the convolution kernels in Theano as follows:
```
import theano.tensor as T
from nervana_theano.conv import nervana_conv

x = T.tensor4('x')
w = T.tensor4('w')

border_mode = 'valid' # or 'full', or...
# for nervana_conv this can also be a tuple of integers

conv_theano = T.nnet.conv.conv2d(x, w, border_mode=border_mode, subsample=strides)
conv_nervana = nervana_conv(x, w, padding=border_mode, strides=strides)

# or with manual shuffling:
x_shuffled = x.dimshuffle(1, 2, 3, 0) # batch size last
w_shuffled = w.dimshuffle(1, 2, 3, 0) # batch size last

conv_nervana = nervana_conv(x_shuffled, w_shuffled, padding=border_mode, strides=strides, dimshuffle=False)
```

Note that `nervana_conv` will perform a 1D, 2D or 3D convolution depending on whether the inputs are 3D, 4D or 5D tensors.

### with Lasagne
`nervana_theano.layers.NervanaConvLayer` is a Lasagne layer class that functions as a drop-in replacement for `lasagne.layers.Conv1DLayer` and `lasagne.layers.Conv2DLayer` (in addition it also supports 3D convolutions). Like `lasagne.layers.cuda_convnet.Conv2DCCLayer` it has a `dimshuffle` keyword argument to perform the necessary dimshuffles automatically. This is set to `True` by default and can be disabled as a performance optimization.

## To do
This is a work in progress, help / remarks / suggestions / pull requests are very welcome. I don't know if I will have much time to work on this in the near future. Here are some things left to do:

- implement the gradient for the GEMM wrapper
- add wrappers for the pooling kernels
- optimize the wrappers to avoid unnecessary computation and other slowdowns
- figure out what needs to change to optimize performance for the GTX 980 (instead of the GTX Titan X only), since more people have this card
- add support for the built-in ReLU operation of the kernels
- add Theano optimizations that replace theano.tensor.nnet.conv2d / theano.tensor.dot with the Nervana equivalents automatically

## Disclaimer

This code comes with no warranty or support, I just put it online because I figured other people might find it interesting / useful. Use it at your own risk. If something doesn't work you're welcome to submit an issue on GitHub, but I can't promise I'll be able to look at it or fix it.

## Acknowledgements

Thanks to Scott Gray, Jan Schlüter and the Theano team for their help, and to Nervana Systems for open sourcing nervanagpu.
