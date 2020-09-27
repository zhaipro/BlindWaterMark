#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import random

import cv2
import numpy as np


def encode(im, wm, alpha=30.0, seed=20160930):
    h, w, c = im.shape
    hwm = np.zeros((int(h * 0.5), w, c))
    assert hwm.shape[0] > wm.shape[0]
    assert hwm.shape[1] > wm.shape[1]
    hwm[:wm.shape[0], :wm.shape[1]] = wm

    random.seed(seed)
    m, n = np.arange(hwm.shape[0]), np.arange(hwm.shape[1])
    random.shuffle(m)
    random.shuffle(n)
    rwm = np.zeros(im.shape)
    for i in range(hwm.shape[0]):
        for j in range(hwm.shape[1]):
            rwm[i, j] = hwm[m[i], n[j]]

    _h = hwm.shape[0]
    rwm[-_h:] = np.rot90(rwm[:_h], 2)

    f1 = np.fft.fft2(im, axes=(0, 1))
    t = np.abs(f1)
    print(t.min(), t.max(), t.mean())
    t = t / t.max() * 255
    cv2.imshow('f1', t)
    cv2.waitKey()
    f2 = f1 + alpha * rwm
    _im = np.fft.ifft2(f2, axes=(0, 1))

    img_wm = np.real(_im)

    return img_wm


def decode(img, img_wm, alpha=30.0, seed=20160930):
    random.seed(seed)
    m, n = np.arange(int(img.shape[0] * 0.5)), np.arange(img.shape[1])
    random.shuffle(m)
    random.shuffle(n)

    f1 = np.fft.fft2(img, axes=(0, 1))
    f2 = np.fft.fft2(img_wm, axes=(0, 1))

    rwm = (f2 - f1) / alpha
    rwm = np.real(rwm)

    wm = np.zeros(rwm.shape)
    for i in range(int(rwm.shape[0] * 0.5)):
        for j in range(rwm.shape[1]):
            wm[m[i], n[j]] = rwm[i, j]

    for i in range(int(rwm.shape[0] * 0.5)):
        for j in range(rwm.shape[1]):
            wm[-m[i], -n[j]] = rwm[-1 - i, -1 - j]
    return wm


if __name__ == '__main__':
    cmd = None
    seed = 20160930
    alpha = 3.0
    if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) < 2:
        print('Usage: python bwm.py <cmd> [arg...] [opts...]')
        print('  cmds:')
        print('    encode <image> <watermark> <image(encoded)>')
        print('           image + watermark -> image(encoded)')
        print('    decode <image> <image(encoded)> <watermark>')
        print('           image + image(encoded) -> watermark')
        print('  opts:')
        print('    --seed <int>,     Manual setting random seed (default is 20160930)')
        print('    --alpha <float>,  Manual setting alpha (default is 3.0)')
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
        img_wm = encode(im, wm)
        assert cv2.imwrite(fn3, img_wm, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    elif cmd == 'decode':
        img = cv2.imread(fn1)
        img_wm = cv2.imread(fn2)
        wm = decode(img, img_wm)
        assert cv2.imwrite(fn3, wm)
