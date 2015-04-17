from theano import Op, Apply

import theano.sandbox.cuda as cuda

try:
    from nervanagpu.nervanagpu import GPUTensor
except ImportError:
    GPUTensor = None


def to_gputensor(a):
    assert a.is_c_contiguous
    return GPUTensor(a.shape, dtype=a.dtype, base=a, gpudata=a.gpudata,
                     strides=a.strides, is_trans=False)



class NervanaOp(cuda.GpuOp): # base class for shared code between scikits.cuda-based ops
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def output_type(self, inp):
        raise NotImplementedError

    def make_node(self, inp):
        inp = cuda.basic_ops.gpu_contiguous(
           cuda.basic_ops.as_cuda_ndarray_variable(inp))

        assert inp.dtype == "float32"

        return theano.Apply(self, [inp], [self.output_type(inp)()])



class NervanaDot(NervanaOp):
    __props__ = ('relu')

    def __init__(self, relu=False):
        self.relu = relu

    def output_type(self, inp):
        return cuda.CudaNdarrayType(broadcastable=[False, False])

    def make_thunk(self, node, storage_map, _, _2):
        if GPUTensor is None:
            raise RuntimeError("nervanagpu not found")

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            input1_shape = inputs[0][0].shape
            input2_shape inputs[1][0].shape

            assert input1_shape[1] == input2_shape[0]

            output_shape = (input1_shape[0], input2_shape[1])

            z = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if z[0] is None or z[0].shape != output_shape:
                z[0] = cuda.CudaNdarray.zeros(output_shape)

            input1_nervana = to_gputensor(inputs[0][0])
            input2_nervana = to_gputensor(inputs[1][0])
            output_nervana = to_gputensor(z[0])

            input1_nervana.dot(input1_nervana, input2_nervana, output_nervana,
                               alpha=1, beta=0, relu=self.relu)

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

nervana_dot = NervanaDot()


if __name__ == "__main__":
    import numpy as np
    import theano
    import theano.tensor as T

    x = theano.shared(np.random.randn(200, 300).astype(theano.config.floatX))
    y = theano.shared(np.random.randn(300, 100).astype(theano.config.floatX))

    prod1 = T.dot(x, y)
    prod2 = nervana_dot(x, y)

    val1 = prod1.eval()
    val2 = prod2.eval()

    assert np.allclose(val1, val2a, b)
