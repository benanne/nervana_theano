"""
Lasagne layers wrapping nervana_conv
"""

import lasagne
import conv


class NervanaConvLayer(lasagne.layers.Layer):
    """
    This layer supports 1D, 2D, 3D convolutions using nervana_conv
    """
    def __init__(self, incoming, num_filters, filter_size, stride=None,
                 border_mode=None, untie_biases=False, W=None,
                 b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 pad=None, dimshuffle=True, **kwargs):
        super(NervanaConvLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = lasagne.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        ndim = len(self.input_shape)
        cdim = ndim - 2

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.untie_biases = untie_biases
        self.dimshuffle = dimshuffle
        self.stride = (1,) * cdim if stride is None else stride

        if border_mode is not None and pad is not None:
            raise RuntimeError("You cannot specify both 'border_mode' and "
                               "'pad'. To avoid ambiguity, please specify "
                               "only one of them.")
        elif border_mode is None and pad is None:
            # no option specified, default to valid mode
            self.pad = (0,) * cdim
        elif border_mode is not None:
            if border_mode == 'valid':
                self.pad = (0,) * cdim
            elif border_mode == 'full':
                self.pad = tuple(s - 1 for s in self.filter_size)
            elif border_mode == 'same':
                self.pad = tuple((s - 1) // 2 for s in self.filter_size)
            else:
                raise RuntimeError("Unsupported border_mode for "
                                   "NervanaConvLayer: %s" % border_mode)
        else:
            self.pad = pad

        assert len(self.filter_size) == cdim  # messy, make it so this can be an integer as well
        assert len(self.stride) == cdim
        assert len(self.pad) == cdim

        if W is None:
            if dimshuffle:
                W = lasagne.init.GlorotUniform()
            elif ndim == 4:
                W = lasagne.init.GlorotUniform(c01b=True)
            else:
                raise RuntimeError("Please specify a weight initializer, there "
                                   "is no sensible default in this case.")

        self.W = self.create_param(W, self.get_W_shape())
        if b is None:
            self.b = None
        elif self.untie_biases:
            output_shape = self.get_output_shape()
            if self.dimshuffle:
                self.b = self.create_param(b, (num_filters,) + output_shape[2:])
            else:
                self.b = self.create_param(b, (num_filters,) + output_shape[1:-1])
        else:
            self.b = self.create_param(b, (num_filters,))

    def get_W_shape(self):
        if self.dimshuffle:
            num_input_channels = self.input_shape[1]
            return (self.num_filters, num_input_channels) + self.filter_size
        else:
            num_input_channels = self.input_shape[0]
            return (num_input_channels,) + self.filter_size + (self.num_filters,)

    def get_params(self):
        return [self.W] + self.get_bias_params()

    def get_bias_params(self):
        return [self.b] if self.b is not None else []

    def get_output_shape_for(self, input_shape):
        if self.dimshuffle:
            batch_size = input_shape[0]
            input_size = input_shape[2:]
        else:
            batch_size = input_shape[-1]
            input_size = input_shape[1:-1]


        output_size = []
        for i, f, s, p in zip(input_size, self.filter_size, self.stride, self.pad):
            o = lasagne.layers.conv_output_length(i, f, s, 'pad', p)
            output_size.append(o)

        if self.dimshuffle:
            return (batch_size, self.num_filters) + tuple(output_size)
        else:
            return (self.num_filters,) + tuple(output_size) + (batch_size,)

    def get_output_for(self, input, **kwargs):
        conved = conv.nervana_conv(input, self.W, padding=self.pad,
                                   strides=self.stride,
                                   dimshuffle=self.dimshuffle)


        if self.b is not None:
            ndim = conved.ndim
            cdim = ndim - 2

            if self.dimshuffle:
                if self.untie_biases:
                    axes = ['x'] + range(conved.ndim - 1)
                else:
                    axes = ['x', 0] + (['x'] * cdim)
            else:
                if self.untie_biases:
                    axes = range(conved.ndim - 1) + ['x']
                else:
                    axes = [0, 'x'] + (['x'] * cdim)

            biases = self.b.dimshuffle(*axes)
            conved += biases

        conved = self.nonlinearity(conved)
        return conved
