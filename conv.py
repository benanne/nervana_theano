import numpy as np
import theano
import theano.sandbox.cuda as cuda
from theano.sandbox.cuda.basic_ops import host_from_gpu

import theano.misc.pycuda_init

from nervanagpu import nervanagpu
from nervanagpu.layers import _magic32, _flatten
from math import ceil
from operator import mul

from gemm import to_gputensor, NervanaOp, lib


# size: layer.fprop_size
# grid: layer.fprop_grid
# block: layer.fprop_block
# args: layer.kernel_args
# shared: layer.lut_size
# A: I (input)
# B: F (filters)
# C: O (output)
# alpha: alpha
# relu: relu
# zero: False


# interface to aim for:
# fprop_conv(I, F, O, alpha=1.0, relu=False):

# need to eliminate: size, grid, block, args, shared



def _compute_kernel_settings(N, C, K,
                             D=1, H=1, W=1,
                             T=1, R=1, S=1,
                             pad_d=0, pad_h=0, pad_w=0,
                             str_d=1, str_h=1, str_w=1,
                             grid_P=0, grid_Q=0, update_size=None):

    assert N % 8 == 0, "N dim must be multiple of 8"
    assert K % 8 == 0, "K dim must be multiple of 8"

    # Compute the output spatial dimensions
    M = int(ceil(float(D - T + 1 + 2*pad_d) / str_d))
    P = int(ceil(float(H - R + 1 + 2*pad_h) / str_h))
    Q = int(ceil(float(W - S + 1 + 2*pad_w) / str_w))


    NCK = (N,C,K)
    TRS = (T,R,S)
    DHW = (D,H,W)
    MPQ = (M,P,Q)
    padding = (pad_d, pad_h, pad_w)
    strides = (str_d, str_h, str_w)

    dimI   = (C,D,H,W,N)
    dimF   = (C,T,R,S,K)
    dimO   = (K,M,P,Q,N)
    dimI2  = (C*D*H*W,N)
    dimF2  = (C*T*R*S,K)
    dimO2  = (K*M*P*Q,N)
    dimIew = (C*D*H,W*N)
    dimFew = (C*T*R,S*K)
    dimOew = (K*M*P,Q*N)
    sizeI  = reduce(mul, dimI, 1)
    sizeF  = reduce(mul, dimF, 1)
    sizeO  = reduce(mul, dimO, 1)
    nOut   = reduce(mul, MPQ,  1) * K

    # precompute some multiplications for fast constant memory access
    WN   = W*N
    HWN  = H*WN
    DHWN = D*HWN
    RS   = R*S
    RST  = T*RS
    CRST = C*RST
    PQ   = P*Q
    PM   = P*M
    PQM  = M*PQ
    QN   = Q*N
    PQN  = P*QN
    MPQN = M*PQN

    # I can easily get the kernels working with larger values here.. 
    # But this is what version 1 is coded to support.
    assert PQM < 2**16, "Integer division is faster with 16bit numerators"

    # Kernels can be recoded to support 32bit numerators at
    # some performance loss.
    assert CRST+8 < 2**16, "Integer division is faster with 16bit numerators"

    # precompute grid dimensions
    grid_N64  = N    // 64 + (N    % 64 != 0)
    grid_K64  = K    // 64 + (K    % 64 != 0)
    grid_C64  = CRST // 64 + (CRST % 64 != 0)

    grid_N128 = N    // 128 + (N    % 128 != 0)
    grid_K128 = K    // 128 + (K    % 128 != 0)
    grid_C128 = CRST // 128 + (CRST % 128 != 0)

    #TODO: add more 128x128 kernels for better performance at fp32.
    fprop_grid = (PQM, grid_K64,  grid_N64)
    bprop_grid = (PQM, grid_C128, grid_N64)
    fprop_block = (64,  1, 1)
    bprop_block = (128, 1, 1)
    fprop_size = "K64_N64"
    bprop_size = "C128_N64"

    #TODO: tune this further
    if  (update_size is None or update_size == "C64_K64" or update_size == "C128_K64") and \
        (CRST <= 64 or K <= 64 or (K % 64 == 0 and K % 128 != 0)):
            updat_size = "C128_K64"
            updat_grid  = [0, grid_C128, grid_K64]
            updat_block = 128
    else:
        updat_size = "C128_K128"
        updat_grid  = [0, grid_C128, grid_K128]
        updat_block = 256

    if grid_P == 0 or grid_Q == 0:
        grid_P = P
        grid_Q = Q // 4

        # TitanX optimization: make grid multiple of 24 for small grids
        # TODO: explore L2 utilization here:
        # TODO: add 980, 750, etc optimizations
        if nervanagpu._get_sm_count() == 24:
            grid_PQ  = grid_P * grid_Q
            if   grid_PQ < 30:
                grid_P = 6
                grid_Q = 4
            elif grid_PQ < 54:
                grid_P = 8
                grid_Q = 6
            elif grid_PQ < 78:
                grid_P = 9
                grid_Q = 8
            elif grid_PQ <= 108:
                grid_P = 12
                grid_Q = 8

    if grid_P >= P: grid_P = P
    if grid_Q >= Q: grid_Q = Q

    grid_PQ  = grid_P * grid_Q
    grid_PQM = updat_grid[0] = grid_PQ * M

    updat_grid  = tuple(updat_grid)
    updat_block = (updat_block,1,1)

    # precompute the magic numbers and shift amounts for integer division
    magic_RST = _magic32(CRST+8, RST)
    magic_RS  = _magic32(RST+32, RS)
    magic_S   = _magic32(RS+32, S)
    magic_PQ  = _magic32(PQM, PQ)
    magic_Q   = _magic32(PQ, Q)
    magic_PQu = _magic32(grid_PQM, grid_PQ)
    magic_Qu  = _magic32(grid_PQ, grid_Q)

    # generate the convolution kernel args for fprop and bprop
    kernel_args = _flatten([
        N, K, D, H, W, WN, HWN, DHWN,
        C, CRST, RST, magic_RST, RS, magic_RS, S, magic_S,
        pad_d, pad_h, pad_w, str_d, str_h, str_w,
        P, Q, PQ, QN, PQN, MPQN, magic_Q, magic_PQ,
        grid_P, grid_Q, grid_PQ])

    # update uses slightly different args
    update_args = _flatten([
        N, K, D, H, W, WN, HWN, DHWN,
        C, CRST, RST, magic_RST, RS, magic_RS, S, magic_S,
        pad_d, pad_h, pad_w, str_d, str_h, str_w,
        P, Q, PQ, QN, PQN, MPQN, magic_Qu, magic_PQu,
        grid_P, grid_Q, grid_PQ])

    # shared lookup table size
    lut_size = (RST // 32 + (RST % 32 != 0)) * 32 * 4

    return {
        'fprop': (fprop_size, fprop_grid, fprop_block),
        'bprop': (bprop_size, bprop_grid, bprop_block),
        'updat': (updat_size, updat_grid, updat_block),
        'kernel_args': kernel_args,
        'update_args': update_args,
        'lut_size': lut_size,
    }


def _conv(settings, A, B, C, alpha=1.0, relu=False, op="fprop"):
    """
    Adapted from the nervanagpu code to avoid using the Layer classes.
    A lot of copied code!

    settings is generated by _compute_kernel_settings().
    """
    assert B.dtype == C.dtype == np.dtype('float32')
    assert op in ["fprop", "bprop", "updat"]
    clss = "sconv"  # hardcode fp32 for now

    flags = 0
    if C.rounding:
        flags |= 1
    if relu:
        flags |= 2

    # find the correct settings for this operation
    size, grid, block = settings[op]

    if op in ["fprop", "bprop"]:
        args = settings['kernel_args']
        shared = settings['lut_size']
    elif op == "updat":
        args = settings['update_args']
        shared = 0

    kernel = nervanagpu._get_conv_kernel(lib.cubin_path, clss, op, size)
    params = [grid, block, nervanagpu._get_rand_state(),
              C.gpudata, A.gpudata, B.gpudata,
              alpha, flags]
    params.extend(args)

    kernel.prepared_call(*params, shared_size=shared)




# class NervanaConv(NervanaOp):
#     __props__ = ('relu')

#     def __init__(self, relu=False):
#         self.relu = relu

#     def make_node(self, inp1, inp2):
#         inp1 = cuda.basic_ops.gpu_contiguous(
#            cuda.basic_ops.as_cuda_ndarray_variable(inp1))
#         inp2 = cuda.basic_ops.gpu_contiguous(
#            cuda.basic_ops.as_cuda_ndarray_variable(inp2))

#         assert inp1.dtype == "float32"
#         assert inp2.dtype == "float32"
#         assert inp1.ndim == 2
#         assert inp2.ndim == 2

#         return theano.Apply(self, [inp1, inp2], [self.output_type(inp1)()])

#     def output_type(self, inp):
#         return cuda.CudaNdarrayType(broadcastable=[False, False])

#     def make_thunk(self, node, storage_map, _, _2):
#         inputs = [storage_map[v] for v in node.inputs]
#         outputs = [storage_map[v] for v in node.outputs]

#         def thunk():
#             input1_shape = inputs[0][0].shape
#             input2_shape = inputs[1][0].shape

#             assert input1_shape[1] == input2_shape[0]

#             output_shape = (input1_shape[0], input2_shape[1])

#             z = outputs[0]

#             # only allocate if there is no previous allocation of the right size.
#             if z[0] is None or z[0].shape != output_shape:
#                 z[0] = cuda.CudaNdarray.zeros(output_shape)

#             input1_nervana = to_gputensor(inputs[0][0])
#             input2_nervana = to_gputensor(inputs[1][0])
#             output_nervana = to_gputensor(z[0])

#             lib.dot(input1_nervana, input2_nervana, output_nervana,
#                                alpha=1, beta=0, relu=self.relu)

#         thunk.inputs = inputs
#         thunk.outputs = outputs
#         thunk.lazy = False

#         return thunk

# nervana_conv = NervanaConv()


# if __name__ == "__main__":
#     import theano.tensor as T

#     x = theano.shared(np.random.randn(2000, 3000).astype(theano.config.floatX))
#     y = theano.shared(np.random.randn(3000, 1000).astype(theano.config.floatX))

#     prod1 = T.dot(x, y)
#     prod2 = host_from_gpu(nervana_dot(x, y))

#     val1 = prod1.eval()
#     val2 = prod2.eval()

#     assert np.allclose(val1, val2)
