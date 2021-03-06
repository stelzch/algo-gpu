#!/usr/bin/python3
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters
from reikna import cluda
from reikna.cluda import Snippet, dtypes
from reikna.core import Transformation, Type, Annotation, Parameter, Computation
from reikna.helpers import template_for
from reikna.algorithms import PureParallel
from reikna.transformations import mul_param
from reikna.fft.fftshift import FFTShift
from reikna.fft.fft import FFT



def r2s(size):
    """Return an r² array.
    
    Keyword arguments:
    size -- half the size of the array
    """
    x1 = np.roll(np.arange(-size, size)**2, size)
    y1 = np.roll(np.arange(-size, size)**2, size)
    return np.add.outer(x1, y1)+1e-6


class PhaseComputation(Computation):
    """Calculate and reconstruct phase information"""

    def __init__(self, output, input_, rect_x, rect_y, rect_r):
        """Initialize the computation and its kernels.
        
        Keyword arguments:
        output -- a CLUDA array where the reconstructed phase will be stored
        input_ -- a CLUDA array with the input data
        rect_x -- initial x position of the satellite's center
        rect_y -- initial y position of the satellite's center
        rect_r -- initial radius of the satellite
        """
        Computation.__init__(self, [
            Parameter('output', Annotation(output, 'o')),
            Parameter('magspec', Annotation(output, 'o')),
            Parameter('input', Annotation(input_, 'i')),
            Parameter('rect_x', Annotation(Type(np.uint16, shape=()), 's')),
            Parameter('rect_y', Annotation(Type(np.uint16, shape=()), 's')),
            Parameter('rect_r', Annotation(Type(np.uint16, shape=()), 's'))])

    def _build_plan(self, plan_factory, device_params, output, magspec, input_, rect_x, rect_y, rect_r):
        """Assemble the Computation."""

        """Assert that the input/output size is quadratic and the same"""
        assert output.shape[0] == output.shape[1] == input_.shape[0] == input_.shape[1]

        plan = plan_factory()

        """ Initialize the temporary arrays.
        mirrored contains the mirrored variant of the input image, thus has to
        be four times the size.
        """
        mirrored = np.empty(shape=[2*x for x in input_.shape], dtype=np.complex128)
        temp = plan.temp_array_like(input_)
        temp2 = plan.temp_array_like(input_)
        mirror = plan.temp_array_like(mirrored)
        mirror_temp1 = plan.temp_array_like(mirrored)
        mirror_temp2 = plan.temp_array_like(mirrored)
        mirror_sin = plan.temp_array_like(mirrored)
        mirror_cos = plan.temp_array_like(mirrored)
        r2s_c = plan.persistent_array(r2s(output.shape[0]).astype(np.float64))

        """Initialize computations for the fourier transforms, always on both
        axis."""
        fft = FFT(temp, axes=(0,1))
        fft_mirror = FFT(mirror, axes=(0,1))
        fftshift = FFTShift(temp, axes=(0,1))


        template = template_for("kernels") # Read template from kernels.mako
        
        plan.computation_call(fft, temp, input_, 0)
        plan.computation_call(fftshift, temp, temp)

        plan.kernel_call(
            template.get_def('magspec'),
            [magspec, temp],
            global_size=magspec.shape)

        plan.kernel_call(
            template.get_def('cropcomp'),
            [temp2, temp, rect_x, rect_y, rect_r],
            global_size=output.shape)

        plan.computation_call(fftshift, temp2, temp2)
        plan.computation_call(fft, temp, temp2, 1)

        plan.kernel_call(
            template.get_def('angle'),
            [output, temp],
            global_size=output.shape)

        plan.kernel_call(
            template.get_def('zeroarr'),
            [mirror],
            global_size=mirror.shape)

        plan.kernel_call(
            template.get_def('mirror4'),
            [mirror, output],
            global_size=output.shape)

        print(mirror.dtype, output.dtype)

        plan.kernel_call(
            template.get_def('sine'),
            [mirror_sin, mirror],
            global_size=mirror.shape)

        plan.kernel_call(
            template.get_def('cosine'),
            [mirror_cos, mirror],
            global_size=mirror.shape)

        plan.computation_call(fft_mirror, mirror_temp1, mirror_sin, 0)
        plan.computation_call(fft_mirror, mirror_temp2, mirror_cos, 0)

        plan.kernel_call(
            template.get_def('mulmat'),
            [mirror_temp1, r2s_c, mirror_temp1],
            global_size=mirror.shape)
        plan.kernel_call(
            template.get_def('mulmat'),
            [mirror_temp2, r2s_c, mirror_temp2],
            global_size=mirror.shape)

        plan.computation_call(fft_mirror, mirror_temp1, mirror_temp1, 1)
        plan.computation_call(fft_mirror, mirror_temp2, mirror_temp2, 1)

        plan.kernel_call(
            template.get_def('mulmat'),
            [mirror_temp1, mirror_cos, mirror_temp1],
            global_size=mirror.shape)

        plan.kernel_call(
            template.get_def('mulmat'),
            [mirror_temp2, mirror_sin, mirror_temp2],
            global_size=mirror.shape)

        plan.kernel_call(
            template.get_def('submat'),
            [mirror_temp1, mirror_temp1, mirror_temp2],
            global_size=mirror.shape)

        plan.computation_call(fft_mirror, mirror_temp1, mirror_temp1, 0)

        plan.kernel_call(
            template.get_def('divmat'),
            [mirror_temp1, mirror_temp1, r2s_c],
            global_size=mirror.shape)

        plan.computation_call(fft_mirror, mirror_temp1, mirror_temp1, 1)

        plan.kernel_call(
            template.get_def('phicalc'),
            [output, mirror_temp1, output],
            global_size=output.shape)
        return plan

if __name__ == '__main__':
    """Read from given path and calculate phase."""

    usage = """phasecomputation.py imagefile rect_x rect_y rect_r

    imagefile - path to a quadratic grayscale image file
    """

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(-1)

    interactive = True
    if len(sys.argv) == 5:
        interactive = False


    api = cluda.ocl_api()
    thr = api.Thread.create()

    try:
        img = plt.imread(sys.argv[1])
    except:
        print("[ERROR] Could not read image from file")
        sys.exit(-1)

    print(img.ndim, img.shape)
    if img.ndim == 3:
        size = min(img.shape[0], img.shape[1])
        print(size)
        img = img[:size, :size, 0]
    img = img[:2048, :2048]
    width = 2048
    height = 2048
    print(img.ndim, img.shape)


    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        print("[ERROR] Image must be grayscale with same width and height")
        sys.exit(-1)

    data = img.astype(np.complex128)
    outp = np.zeros(data.shape, dtype=np.float64)
    magspec = np.zeros_like(outp)

    phasecomp = PhaseComputation(outp, data, 50.0, 50, 50)
    phasecompc = phasecomp.compile(thr)

    data_dev = thr.to_device(data)
    mgsp_dev = thr.to_device(magspec)
    outp_dev = thr.to_device(outp)

    # Show user magnitude spectrum to select satellite
    phasecompc(outp_dev, mgsp_dev, data_dev, 975, 528, 24)
    if interactive:
        magspec = mgsp_dev.get()


        center_patch_size = 200
        kernel_size = 50
        kernel = np.ones((kernel_size, kernel_size))
        magspec[height // 2 - center_patch_size // 2:
                height // 2 + center_patch_size // 2,
                width // 2 - center_patch_size // 2:
                width // 2 + center_patch_size // 2] = np.zeros((center_patch_size,
                    center_patch_size))

        # Apply Gaussian filter
        magspec = scipy.ndimage.filters.convolve(magspec, kernel)

        nmax = 5
        maxindex = magspec.argsort(axis=None)[-nmax:][::-1]
        for x in maxindex:
            print("Maximum at {} {}".format(x%2048, x//2048))


        """
        print(magspec.dtype, magspec.shape)
        plt.figure()
        plt.imshow(mgsp_dev.get())
        plt.show()
        rect_x = input("Rect X:")
        rect_y = input("Rect Y:")
        rect_r = input("Rect R:")
        """
        rect_x = maxindex[0] % 2048
        rect_y = maxindex[0] // 2048
        rect_r = 50
    else:
        rect_x = sys.argv[2]
        rect_y = sys.argv[3]
        rect_r = sys.argv[4]

    t0 = time.time()
    phasecompc(outp_dev, mgsp_dev, data_dev, rect_x, rect_y, rect_r)
    outp_t = outp_dev.get()

    t1 = time.time()
    if interactive:
        plt.figure()
        plt.imshow(outp_t)
        plt.show()
    else:
        plt.imsave("output.png", outp_t)

    print("Took {} secs".format(t1-t0))
