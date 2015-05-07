"""
This file contains code from nervanagpu, which is covered by the following
license:


                     Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


import numpy as np
import theano
import theano.sandbox.cuda as cuda
from theano.sandbox.cuda.basic_ops import (host_from_gpu, gpu_from_host,
                                           gpu_contiguous, gpu_alloc_empty)

import theano.misc.pycuda_init

from nervanagpu import nervanagpu
from nervanagpu.layers import _magic32, _flatten
from math import ceil
from operator import mul

from gemm import to_gputensor, NervanaOp, lib


def _compute_kernel_settings(N, C, K,
                             D=1, H=1, W=1,
                             T=1, R=1, S=1,
                             pad_d=0, pad_h=0, pad_w=0,
                             str_d=1, str_h=1, str_w=1,
                             grid_P=0, grid_Q=0, update_size=None):
    """
    Most of this has been copy-pasted from nervanagpu's ConvLayer class.
    It exists to avoid having to instantiate the layer classes inside the
    Theano Ops.
    """

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
        'output_size': (M, P, Q),
    }


def _conv(settings, A, B, C, alpha=1.0, relu=False, op="fprop"):
    """
    Adapted from the nervanagpu code to avoid using the Layer classes.
    A lot of copied code!

    settings is generated by _compute_kernel_settings().
    """
    assert B.dtype == C.dtype == np.float32
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


class NervanaConvBase(NervanaOp):
    __props__ = ('padding', 'strides')

    def __init__(self, padding=(0, 0, 0), strides=(0, 0, 0)):
        self.padding = padding
        self.strides = strides


class NervanaConv(NervanaConvBase):
    def make_node(self, img, kern):
        img = cuda.basic_ops.gpu_contiguous(
            cuda.basic_ops.as_cuda_ndarray_variable(img))
        kern = cuda.basic_ops.gpu_contiguous(
            cuda.basic_ops.as_cuda_ndarray_variable(kern))

        if img.type.ndim != 5:
            raise TypeError('img must be 5D tensor')
        if kern.type.ndim != 5:
            raise TypeError('kern must be 5D tensor')

        broadcastable = [kern.type.broadcastable[-1], False, False, False, img.type.broadcastable[-1]]
        return theano.Apply(self, [img, kern], [cuda.CudaNdarrayType(broadcastable)()])

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        bottom, weights = inputs
        top, = outputs

        settings_shapes = [None]
        settings = [None]

        def thunk():
            bottom_shape = bottom[0].shape
            weights_shape = weights[0].shape

            C , D, H, W, N = bottom_shape
            C_, T, R, S, K = weights_shape

            if self.padding == 'valid':
                pad_d, pad_h, pad_w = 0, 0, 0
            elif self.padding == 'full':
                pad_d, pad_h, pad_w = T - 1, R - 1, S - 1
            elif self.padding == 'half':
                pad_d, pad_h, pad_w = T // 2, R // 2, S // 2
            else:
                pad_d, pad_h, pad_w = self.padding

            str_d, str_h, str_w = self.strides

            assert C_ == C

            if (settings_shapes[0] is None or
                    settings_shapes[0] != (N, C, K, D, H, W, T, R, S)):
                # shape change, recompute settings
                settings_shapes[0] = (N, C, K, D, H, W, T, R, S)
                settings[0] = _compute_kernel_settings(N, C, K,
                                                       D, H, W,
                                                       T, R, S,
                                                       pad_d, pad_h, pad_w,
                                                       str_d, str_h, str_w)

            top_shape = (K,) + settings[0]['output_size'] + (N,)

            # only allocate if there is no previous allocation of the right size.
            if top[0] is None or top[0].shape != top_shape:
                top[0] = cuda.CudaNdarray.zeros(top_shape)

            bottom_nervana = to_gputensor(bottom[0])
            weights_nervana = to_gputensor(weights[0])
            top_nervana = to_gputensor(top[0])

            _conv(settings[0], bottom_nervana, weights_nervana, top_nervana,
                  alpha=1.0, relu=False, op="fprop")

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

    def grad(self, inp, grads):
        bottom, weights = inp
        top, = grads
        top = gpu_contiguous(top)

        d_bottom = NervanaConvGradI(self.padding, self.strides)(weights, top, bottom.shape[1:-1])
        d_weights = NervanaConvGradW(self.padding, self.strides)(bottom, top, weights.shape[1:-1])

        return d_bottom, d_weights


class NervanaConvGradI(NervanaConvBase):
    def make_node(self, kern, topgrad, shape):
        kern = cuda.basic_ops.as_cuda_ndarray_variable(kern)
        topgrad = cuda.basic_ops.as_cuda_ndarray_variable(topgrad)

        if kern.type.ndim != 5:
            raise TypeError('kern must be 5D tensor')
        if topgrad.type.ndim != 5:
            raise TypeError('topgrad must be 5D tensor')

        depth_height_width = [shape[0], shape[1], shape[2]]

        broadcastable = [kern.type.broadcastable[0], False, False, False, topgrad.type.broadcastable[-1]]
        return theano.Apply(self, [kern, topgrad] + depth_height_width, [cuda.CudaNdarrayType(broadcastable)()])

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        weights, top = inputs[:2]
        bottom, = outputs

        settings_shapes = [None]
        settings = [None]

        def thunk():
            weights_shape = weights[0].shape
            top_shape = top[0].shape

            D, H, W = int(inputs[2][0]), int(inputs[3][0]), int(inputs[4][0])

            C, T, R, S, K = weights_shape
            K_, M, P, Q, N = top_shape

            if self.padding == 'valid':
                pad_d, pad_h, pad_w = 0, 0, 0
            elif self.padding == 'full':
                pad_d, pad_h, pad_w = T - 1, R - 1, S - 1
            elif self.padding == 'half':
                pad_d, pad_h, pad_w = T // 2, R // 2, S // 2
            else:
                pad_d, pad_h, pad_w = self.padding

            str_d, str_h, str_w = self.strides

            assert K_ == K

            if (settings_shapes[0] is None or
                    settings_shapes[0] != (N, C, K, D, H, W, T, R, S)):
                # shape change, recompute settings
                settings_shapes[0] = (N, C, K, D, H, W, T, R, S)
                settings[0] = _compute_kernel_settings(N, C, K,
                                                       D, H, W,
                                                       T, R, S,
                                                       pad_d, pad_h, pad_w,
                                                       str_d, str_h, str_w)

            
            bottom_shape = (C, D, H, W, N)

            # only allocate if there is no previous allocation of the right size.
            if bottom[0] is None or bottom[0].shape != bottom_shape:
                bottom[0] = cuda.CudaNdarray.zeros(bottom_shape)

            bottom_nervana = to_gputensor(bottom[0])
            weights_nervana = to_gputensor(weights[0])
            top_nervana = to_gputensor(top[0])

            _conv(settings[0], weights_nervana, top_nervana, bottom_nervana,
                  alpha=1.0, relu=False, op="bprop")

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk


class NervanaConvGradW(NervanaConvBase):
    def make_node(self, img, topgrad, shape):
        img = cuda.basic_ops.as_cuda_ndarray_variable(img)
        topgrad = cuda.basic_ops.as_cuda_ndarray_variable(topgrad)

        if img.type.ndim != 5:
            raise TypeError('img must be 5D tensor')
        if topgrad.type.ndim != 5:
            raise TypeError('topgrad must be 5D tensor')

        depth_height_width = [shape[0], shape[1], shape[2]]

        broadcastable = [img.type.broadcastable[0], False, False, False, topgrad.type.broadcastable[0]]
        return theano.Apply(self, [img, topgrad] + depth_height_width, [cuda.CudaNdarrayType(broadcastable)()])

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        bottom, top = inputs[:2]
        weights, = outputs

        settings_shapes = [None]
        settings = [None]

        def thunk():
            bottom_shape = bottom[0].shape
            top_shape = top[0].shape

            T, R, S = int(inputs[2][0]), int(inputs[3][0]), int(inputs[4][0])

            C , D, H, W, N = bottom_shape
            K, M, P, Q, N_ = top_shape

            if self.padding == 'valid':
                pad_d, pad_h, pad_w = 0, 0, 0
            elif self.padding == 'full':
                pad_d, pad_h, pad_w = T - 1, R - 1, S - 1
            elif self.padding == 'half':
                pad_d, pad_h, pad_w = T // 2, R // 2, S // 2
            else:
                pad_d, pad_h, pad_w = self.padding

            str_d, str_h, str_w = self.strides

            assert N_ == N

            if (settings_shapes[0] is None or
                    settings_shapes[0] != (N, C, K, D, H, W, T, R, S)):
                # shape change, recompute settings
                settings_shapes[0] = (N, C, K, D, H, W, T, R, S)
                settings[0] = _compute_kernel_settings(N, C, K,
                                                       D, H, W,
                                                       T, R, S,
                                                       pad_d, pad_h, pad_w,
                                                       str_d, str_h, str_w)

            
            weights_shape = (C, T, R, S, K)

            # only allocate if there is no previous allocation of the right size.
            if weights[0] is None or weights[0].shape != weights_shape:
                weights[0] = cuda.CudaNdarray.zeros(weights_shape)

            bottom_nervana = to_gputensor(bottom[0])
            weights_nervana = to_gputensor(weights[0])
            top_nervana = to_gputensor(top[0])

            _conv(settings[0], bottom_nervana, top_nervana, weights_nervana,
                  alpha=1.0, relu=False, op="updat")

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

# TODO: test how much of a problem the dimshuffles are in a real network (does Theano avoid copy operations? It seems like it does for the cuda-convnet wrappers...)
# TODO: implement an optimization to swap it in so T.nnet.conv.conv2d can be used?
# TODO: built in relu support (with optimization to enable it?)


def nervana_conv(input, filters, padding=None, strides=1, dimshuffle=True):
    ndim = input.ndim
    if ndim not in [3, 4, 5]:
        raise RuntimeError("inputs should be 3D, 4D or 5D")

    if ndim != filters.ndim:
        raise RuntimeError("inputs and filters should have the same dimensionality")

    cdim = ndim - 2  # actual convolution dimensionality

    # modify padding and strides tuples for 3D convolution
    if isinstance(padding, str):
        if padding == "same":
            padding = "half"
        assert padding in ['full', 'valid', 'half']
    else:
        if isinstance(padding, int):
            padding = (padding,) * cdim
        elif isinstance(padding, tuple):
            assert len(padding) == cdim
        padding = ((0,) * (3 - cdim)) + padding

    if isinstance(strides, int):
        strides = (strides,) * cdim
    elif isinstance(strides, tuple):
        assert len(strides) == cdim
    strides = ((1,) * (3 - cdim)) + strides

    if dimshuffle:
        axes = range(1, ndim) + [0]
        input = input.dimshuffle(*axes)
        filters = filters.dimshuffle(*axes)

    # go from ndim dimensions to 5 dimensions by 1-padding
    if ndim == 3:
        new_input_shape = (input.shape[0], 1, 1, input.shape[1], input.shape[2])
        new_filters_shape = (filters.shape[0], 1, 1, filters.shape[1], filters.shape[2])
    elif ndim == 4:
        new_input_shape = (input.shape[0], 1, input.shape[1], input.shape[2], input.shape[3])
        new_filters_shape = (filters.shape[0], 1, filters.shape[1], filters.shape[2], filters.shape[3])
    elif ndim == 5:
        new_input_shape = input.shape
        new_filters_shape = filters.shape

    input = input.reshape(new_input_shape)
    filters = filters.reshape(new_filters_shape)
    
    op = NervanaConv(padding=padding, strides=strides)
    out = op(input, filters)

    # go from 5 dimensions back to ndim dimensions by removing the added ones
    # using dimshuffle and slicing for this instead leads to hard-to-debug errors
    if ndim == 3:
        new_out_shape = (out.shape[0], out.shape[3], out.shape[4])
    elif ndim == 4:
        new_out_shape = (out.shape[0], out.shape[2], out.shape[3], out.shape[4])
    elif ndim == 5:
        new_out_shape = out.shape

    out = out.reshape(new_out_shape)

    if dimshuffle:
        axes = [ndim - 1] + range(0, ndim - 1)
        out = out.dimshuffle(*axes)

    return out


if __name__ == "__main__":
    import theano.tensor as T
    from theano.sandbox.cuda import dnn

    input_shape = (128, 8, 96, 96)
    filter_shape = (64, 8, 3, 3)
    padding = "valid" # (1, 1)
    strides = (1, 1)

    # input_shape = (32, 16, 48, 48)
    # filter_shape = (24, 16, 3, 3)
    # padding = (1, 1)
    # strides = (1, 1)

    print "fprop"
    x = theano.shared(np.random.normal(0, 1, input_shape).astype(theano.config.floatX))
    w = theano.shared(np.random.normal(0, 1, filter_shape).astype(theano.config.floatX))

    y_cudnn = dnn.dnn_conv(x, w, border_mode=padding, subsample=strides, conv_mode='cross')
    y_nervana_raw = nervana_conv(x, w, padding=padding, strides=strides)
    y_nervana = gpu_from_host(y_nervana_raw)

    val_cudnn = np.array(y_cudnn.eval())
    val_nervana = np.array(y_nervana.eval())

    assert np.allclose(val_cudnn, val_nervana)

    print "fprop without dimshuffle"
    x_nodimshuffle = theano.shared(x.get_value().transpose(1, 2, 3, 0)) # c01b
    w_nodimshuffle = theano.shared(w.get_value().transpose(1, 2, 3, 0)) # c01b

    y_nervana_nodimshuffle = gpu_from_host(nervana_conv(x_nodimshuffle, w_nodimshuffle, padding=padding, strides=strides, dimshuffle=False))

    val_nervana_nodimshuffle = np.array(y_nervana_nodimshuffle.eval()).transpose(3, 0, 1, 2)

    assert np.allclose(val_nervana, val_nervana_nodimshuffle)


    print "backprop inputs"
    gi_cudnn = T.grad(T.mean(y_cudnn**2), x)
    gi_nervana = T.grad(T.mean(y_nervana_raw**2), x)

    gival_cudnn = np.array(gi_cudnn.eval())
    gival_nervana = np.array(gi_nervana.eval())

    assert np.allclose(gival_cudnn, gival_nervana)


    print "backprop weights"
    gw_cudnn = T.grad(T.mean(y_cudnn**2), w)
    gw_nervana = T.grad(T.mean(y_nervana_raw**2), w)

    gwval_cudnn = np.array(gw_cudnn.eval())
    gwval_nervana = np.array(gw_nervana.eval())

    assert np.allclose(gwval_cudnn, gwval_nervana)



# %timeit y_cudnn.eval()                -> 47.0 ms
# %timeit y_nervana.eval()              -> 61.3 ms
# %timeit y_nervana_nodimshuffle.eval() -> 23.6 ms
