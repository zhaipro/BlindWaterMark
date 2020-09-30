#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import cv2
import numpy as np


def encode(im, wm, alpha=30.0, seed=20160930):
    h, w, c = im.shape
    rwm = np.zeros((h - 1, w // 2, c))
    wm = np.rot90(wm)
    rwm[:wm.shape[0], :wm.shape[1]] = wm

    np.random.seed(seed)
    m = np.random.permutation(rwm.shape[0])
    n = np.random.permutation(rwm.shape[1])
    rwm = rwm[m][:, n]

    f1 = np.fft.rfft2(im, axes=(0, 1))
    f1[1:, 1:] += f1[1:, 1:] / np.abs(f1[1:, 1:]) * alpha * rwm
    img_wm = np.fft.irfft2(f1, axes=(0, 1))

    return img_wm


def decode(img, img_wm, alpha=30.0, seed=20160930):
    np.random.seed(seed)
    m = np.random.permutation(img.shape[0] - 1)
    n = np.random.permutation(img.shape[1] // 2)

    f1 = np.fft.rfft2(img, axes=(0, 1))
    f2 = np.fft.rfft2(img_wm, axes=(0, 1))

    rwm = (f2 - f1) / (f1 / np.abs(f1) * alpha)
    rwm = np.real(rwm)[1:, 1:]

    wm = np.zeros(rwm.shape)
    for i in range(rwm.shape[0]):
        for j in range(rwm.shape[1]):
            wm[m[i], n[j]] = rwm[i, j]
    return wm


if __name__ == '__main__':
    cmd = None
    seed = 20160930
    alpha = 30.0
    if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) < 2:
        print('Usage: python bwm.py <cmd> [arg...] [opts...]')
        print('  cmds:')
        print('    encode <image> <watermark> <image(encoded)>')
        print('           image + watermark -> image(encoded)')
        print('    decode <image> <image(encoded)> <watermark>')
        print('           image + image(encoded) -> watermark')
        print('  opts:')
        print('    --seed <int>,     Manual setting random seed (default is 20160930)')
        print('    --alpha <float>,  Manual setting alpha (default is 30.0)')
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd != 'encode' and cmd != 'decode':
        print('Wrong cmd %s' % cmd)
        sys.exit(1)
    if '--seed' in sys.argv:
        p = sys.argv.index('--seed')
        if len(sys.argv) <= p+1:
            print('Missing <int> for --seed')
            sys.exit(1)
        seed = int(sys.argv[p+1])
        del sys.argv[p+1]
        del sys.argv[p]
    if '--alpha' in sys.argv:
        p = sys.argv.index('--alpha')
        if len(sys.argv) <= p+1:
            print('Missing <float> for --alpha')
            sys.exit(1)
        alpha = float(sys.argv[p+1])
        del sys.argv[p+1]
        del sys.argv[p]
    if len(sys.argv) < 5:
        print('Missing arg...')
        sys.exit(1)
    fn1 = sys.argv[2]
    fn2 = sys.argv[3]
    fn3 = sys.argv[4]

    if cmd == 'encode':
        im = cv2.imread(fn1)
        wm = cv2.imread(fn2)
        img_wm = encode(im, wm, alpha=alpha)
        assert cv2.imwrite(fn3, img_wm)
    elif cmd == 'decode':
        img = cv2.imread(fn1)
        img_wm = cv2.imread(fn2)
        wm = decode(img, img_wm, alpha=alpha)
        assert cv2.imwrite(fn3, wm)
