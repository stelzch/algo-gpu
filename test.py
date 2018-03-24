#!/usr/bin/python3
import time
import numpy as np
import matplotlib.pyplot as plt
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

api = cluda.ocl_api()
thr = api.Thread.create()

img = plt.imread('/home/cstelz/holmos_raw.png')

data = img.astype(np.complex128)
magspec = np.zeros_like(data)
outp = np.zeros_like(data).astype(np.float64)

phasecomp = PhaseComputation(outp, data, 50.0, 50, 50)
phasecompc = phasecomp.compile(thr)

data_dev = thr.to_device(data)
mgsp_dev = thr.to_device(magspec)
outp_dev = thr.to_device(outp)

t0 = time.time()
phasecompc(outp_dev, mgsp_dev, data_dev, 975, 528, 24)
outp_t = outp_dev.get()

t1 = time.time()
plt.figure()
plt.imshow(outp_t)
plt.imsave('output.png', outp_t)
plt.show()

print("Took {} secs".format(t1-t0))
