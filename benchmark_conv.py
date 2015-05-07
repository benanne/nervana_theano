#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmarks Nervana System's convolution against cuDNN.
Based on Theano convolution benchmark script from Soumith's convnet-benchmarks.

Author: Jan Schlüter
"""

import os
import sys
import numpy as np
import math

try:
    import theano.misc.pycuda_init
    import pycuda.driver
except ImportError:
    print "Note: pycuda not available, no timing via CUDA events possible"
    import time
    pycuda = None
import theano

try:
    import theano.sandbox.cuda.dnn
    if not theano.sandbox.cuda.dnn.dnn_available():
        del theano.sandbox.cuda.dnn
        raise ImportError
except (ImportError, NameError):
    print "Note: cuDNN not available"

try:
    from nervana_theano.conv import nervana_conv
except ImportError:
    print "Note: nervanaGPU not available"
    nervana_conv = None


number = 10  # nb of steps in loop to average over
repeat = 1   # nb of trials to pick the minimum of

runs = [
   {
      'ni': 3,
      'no': 96,
      'kw': 11,
      'kh': 11,
      'iw': 128,
      'ih': 128,
      'bs': 128,
      'dw': 1,
      'dh': 1,
   },
   {
      'ni': 64,
      'no': 128,
      'kw': 9,
      'kh': 9,
      'iw': 64,
      'ih': 64,
      'bs': 128,
      'dw': 1,
      'dh': 1,
   },
   {
      'ni': 128,
      'no': 128,
      'kw': 9,
      'kh': 9,
      'iw': 32,
      'ih': 32,
      'bs': 128,
      'dw': 1,
      'dh': 1,
   },
   {
      'ni': 128,
      'no': 128,
      'kw': 7,
      'kh': 7,
      'iw': 16,
      'ih': 16,
      'bs': 128,
      'dw': 1,
      'dh': 1,
   },
   {
      'ni': 384,
      'no': 384,
      'kw': 3,
      'kh': 3,
      'iw': 13,
      'ih': 13,
      'bs': 128,
      'dw': 1,
      'dh': 1,
   }
]

def time_run(fn):
    times = []
    fn()  # warm-up call, not timed
    if pycuda:
        theano.sandbox.cuda.synchronize()
        start = pycuda.driver.Event()
        end = pycuda.driver.Event()
        for _ in range(repeat):
            start.record()
            for _ in range(number):
                fn()
            end.record()
            end.synchronize()
            times.append(start.time_till(end) / 1e3 / number)
    else:
        for _ in range(repeat):
            theano.sandbox.cuda.synchronize()
            start = time.time()
            for _ in range(number):
                fn()
            theano.sandbox.cuda.synchronize()
            times.append((time.time() - start) / number)
    return min(times)

def print_graph(fn):
    if int(os.environ.get('PRINT_GRAPH', 0)):
        # debugprint of graph (in blue text)
        print '\033[1;34m'
        theano.printing.debugprint(fn)
        print '\033[1;m'

def benchmark_three_ways(name, sharedX, sharedY, sharedW, X, Y, gW, gX, mode=None):
    # benchmark fprop
    try:
        fprop = theano.function([], [],
                                givens=[(X, sharedX)],
                                updates=[(sharedY, Y)],
                                mode=mode,
                                name=name + " fprop")
        tm = time_run(fprop)
        print '{: <50} ==> {: <13} ==> {: >7}'.format(name, 'fprop', int(tm*1000))
        print_graph(fprop)
        del fprop
    except Exception, e:
        print name, 'fprop: FAILED', str(e).split('\n', 1)[0]

    # benchmark bprop wrt input
    try:
        bprop = theano.function([], [],
                                # the nvidia wrapper need this (in fact could be optional for subsample==(1, 1)
                                givens=[(X, sharedX)],
                                updates=[(sharedX, gX)],
                                mode=mode,
                                name=name + " bprop inputs")
        tm = time_run(bprop)
        print '{: <50} ==> {: <13} ==> {: >7}'.format(name, 'bprop inputs', int(tm*1000))
        print_graph(bprop)
        del bprop
    except Exception, e:
        print name, 'bprop inputs: FAILED', str(e).split('\n', 1)[0]

    # benchmark bprop wrt weights
    try:
        bprop = theano.function([], [],
                                givens=[(X, sharedX)],
                                updates=[(sharedW, gW)],
                                mode=mode,
                                name=name + " bprop weights")
        tm = time_run(bprop)
        print '{: <50} ==> {: <13} ==> {: >7}'.format(name, 'bprop weights', int(tm*1000))
        print_graph(bprop)
        del bprop
    except Exception, e:
        print name, 'bprop weights: FAILED', str(e).split('\n', 1)[0]
    print ''

def parse_custom_config(s):
    # parses a custom configuration string of the format:
    # iAxBxC,kDxExF,bG,sHxJ where A: input channels, B: input width, C: input height,
    # D: output channels, E: kernel width, F: kernel height, G: batchsize,
    # H: horizontal stride, J: vertical stride (with G, H, J being optional)
    run = {'bs': 128, 'dw': 1, 'dh': 1}
    defs = {'i': ['ni', 'iw', 'ih'],
            'k': ['no', 'kw', 'kh'],
            'b': ['bs'],
            's': ['dw', 'dh']}
    for part in s.split(','):
        p, args = part[0], map(int, part[1:].split('x'))
        run.update(zip(defs[p], args))
    return run

if len(sys.argv) > 1:
    # allow specifying the runs on command line, 1-indexed (i.e., 1 2 5)
    runs = [runs[int(r) - 1] for r in sys.argv[1:] if r[0] != 'i']
    # allow specifying custom configurations on command line (e.g., i3x80x15,k32x3x7,b256)
    runs.extend([parse_custom_config(r) for r in sys.argv[1:] if r[0] == 'i'])

for run in runs:
    # params for run:
    # (input channels, output channels, kernel width, kernel height, batchsize, image width, image height, horizontal stride, vertical stride)
    ni, no, kw, kh, bs, iw, ih, dw, dh = run['ni'], run['no'], run['kw'], run['kh'], run['bs'], run['iw'], run['ih'], run['dw'], run['dh']
    print ''
    print 'CONFIG: input =', ni, 'x', iw, 'x', ih, '* ker =', ni, 'x', no, 'x', kw, 'x', kh, '( bs =', bs, ', stride =', dw, ')'
    ops = 2  # ops per point
    mode = theano.compile.get_default_mode().including('gpu')

    # general setup
    input_shape = (bs, ni, ih, iw)
    filter_shape = (no, ni, kh, kw)
    try:
        sharedX = theano.shared(np.random.randn(*input_shape).astype('float32'), name='sharedX')
        sharedY = theano.shared(np.random.randn(bs, no, (ih-kh)/dh+1, (iw-kw)/dw+1).astype('float32'), name='sharedY')
        sharedW = theano.shared(np.random.randn(*filter_shape).astype('float32'), name='sharedW')
    except MemoryError, e:
        print "SKIPPING config due to the memory error below"
        print e
        continue
    X = theano.tensor.tensor4('X')

    # benchmark nvidia convolution (directly, not via graph optimizer)
    if hasattr(theano.sandbox.cuda, 'dnn'):
        Y = theano.sandbox.cuda.dnn.dnn_conv(X, sharedW, 'valid',
                                             subsample=(dh, dw))
        gW = theano.grad(None, wrt=sharedW, known_grads={Y: sharedY})
        gX = theano.grad(None, wrt=X, known_grads={Y: sharedY})
        benchmark_three_ways(
            '(manual conv) theano.sandbox.cuda.dnn.GpuDnnConv',
            sharedX, sharedY, sharedW, X, Y, gW, gX)

    # benchmark nervana convolution (directly, not via graph optimizer)
    if nervana_conv is not None:
        # input in bc01, dimshuffled on GPU
        Y = nervana_conv(X, sharedW, 'valid', (dh, dw))
        gW = theano.grad(None, wrt=sharedW, known_grads={Y: sharedY})
        gX = theano.grad(None, wrt=X, known_grads={Y: sharedY})
        benchmark_three_ways(
            '(manual conv) nervana_conv(dimshuffle=True)',
            sharedX, sharedY, sharedW, X, Y, gW, gX)

        # input in c01b (native layout for Nervana System's kernels)
        sharedX = theano.shared(sharedX.get_value().transpose(1, 2, 3, 0))
        sharedY = theano.shared(sharedY.get_value().transpose(1, 2, 3, 0))
        sharedW = theano.shared(sharedW.get_value().transpose(1, 2, 3, 0))
        Y = nervana_conv(X, sharedW, 'valid', (dh, dw), dimshuffle=False)
        gW = theano.grad(None, wrt=sharedW, known_grads={Y: sharedY})
        gX = theano.grad(None, wrt=X, known_grads={Y: sharedY})
        benchmark_three_ways(
            '(manual conv) nervana_conv(dimshuffle=False)',
            sharedX, sharedY, sharedW, X, Y, gW, gX)

    del sharedX
    del sharedY
    del sharedW

