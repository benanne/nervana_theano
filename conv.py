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


class Gemm16(Op):
    __props__ = ('relu', 'inplace')

    def __init__(self, relu=False, inplace=False):
        self.relu = relu
        self.inplace = inplace

    def make_node(self, C, alpha, A, B, beta):
        if GPUTensor is None:
            raise RuntimeError("Can't use Gemm16: nervanagpu not found")

        A = cuda.as_cuda_ndarray_variable(A)
        B = cuda.as_cuda_ndarray_variable(B)
        C = cuda.as_cuda_ndarray_variable(C)

        assert C.dtype == A.dtype == B.dtype == 'float16'

        return Apply(self, [C, alpha, A, B, beta], [C.type()])

    def perform(self, node, inputs, outputs):
        C, alpha, A, B, beta = inputs
        inplace = self.inplace
        if inplace and not C.flags.forc:
            inplace = False
        if not inplace:
            C = C.copy()
        At = to_gputensor(A)
        Bt = to_gputensor(B)
        Ct = to_gputensor(C)
        outputs[0][0] = At.dot(At, Bt, Ct, alpha=alpha, beta=beta,
                               relu=self.relu)