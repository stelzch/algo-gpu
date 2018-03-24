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
    x1 = np.roll(np.arange(-size, size)**2, size)
    y1 = np.roll(np.arange(-size, size)**2, size)
    return np.add.outer(x1, y1)+1e-6


class PhaseComputation(Computation):
    def __init__(self, output, input_, rect_x, rect_y, rect_r, square_size):
        Computation.__init__(self, [
            Parameter('output', Annotation(output, 'o')),
            Parameter('magspec', Annotation(output, 'o')),
            Parameter('input', Annotation(input_, 'i')),
            Parameter('r2s', Annotation(Type(np.float64, shape=(output.shape[0]*2, output.shape[1]*2)), 'io')),
            Parameter('rect_x', Annotation(Type(np.uint16, shape=()), 's')),
            Parameter('rect_y', Annotation(Type(np.uint16, shape=()), 's')),
            Parameter('rect_r', Annotation(Type(np.uint16, shape=()), 's'))])
        self.square_size = square_size

    def _build_plan(self, plan_factory, device_params, output, magspec, input_, r2s, rect_x, rect_y, rect_r):
        assert output.shape[0] == output.shape[1] == input_.shape[0] == input_.shape[1]

        mirrored = np.empty(shape=[2*x for x in input_.shape], dtype=np.complex128)
        plan = plan_factory()

        temp = plan.temp_array_like(input_)
        temp2 = plan.temp_array_like(input_)
        mirror = plan.temp_array_like(mirrored)
        mirror_temp1 = plan.temp_array_like(mirrored)
        mirror_temp2 = plan.temp_array_like(mirrored)
        mirror_sin = plan.temp_array_like(mirrored)
        mirror_cos = plan.temp_array_like(mirrored)
        r2s_c = plan.persistent_array(r2s(output.shape[0]).astype(np.float64))

        fft = FFT(temp, axes=(0,1))
        fft_mirror = FFT(mirror, axes=(0,1))
        fftshift = FFTShift(temp, axes=(0,1))
        fftshift_mirror = FFTShift(mirror, axes=(0,1))

        plan.computation_call(fft, temp, input_, 0)
        plan.computation_call(fftshift, temp, temp)

        template = template_for("kernels") # Read template from kernels.mako
        
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
            [mirror_temp1, r2s, mirror_temp1],
            global_size=mirror.shape)
        plan.kernel_call(
            template.get_def('mulmat'),
            [mirror_temp2, r2s, mirror_temp2],
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
            [mirror_temp1, mirror_temp1, r2s],
            global_size=mirror.shape)

        plan.computation_call(fft_mirror, mirror_temp1, mirror_temp1, 1)

        plan.kernel_call(
            template.get_def('magspec'),
            [r2s, mirror],
            global_size=mirror.shape)
        plan.kernel_call(
            template.get_def('phicalc'),
            [output, mirror_temp1, output],
            global_size=output.shape)
        
        

        return plan




api = cluda.ocl_api()
thr = api.Thread.create()


img = plt.imread('/home/cstelz/holmos_raw.png')
n = 2
#data = np.kron(img, np.ones((n,n))).astype(np.complex128)
data = img.astype(np.complex128)
magspec = np.zeros_like(data)
print(data.shape)
outp = np.zeros((1024, 1024)).astype(np.float64)
phasecomp = PhaseComputation(outp, data, 50.0, 50, 50, 1024)
phasecompc = phasecomp.compile(thr)

data_dev = thr.to_device(data)
mgsp_dev = thr.to_device(magspec)
outp_dev = thr.to_device(outp)
r2s_dev = thr.to_device(r2s(1024).astype(np.float64))

print(np.min(r2s_dev.get()))
print("Dev: ",r2s_dev.shape)

t0 = time.time()
phasecompc(outp_dev, mgsp_dev, data_dev, r2s_dev, 975, 528, 24)
outp_t = outp_dev.get()
magspec_t = r2s_dev.get()

t1 = time.time()
print("Min: {}, Max: {}".format(np.min(outp_t), np.max(outp_t)))
print(outp_t.dtype, outp_t.shape, np.max(outp_t), np.min(outp_t), outp_t)
#print(magspec_t)
plt.figure()
plt.subplot(121)
#plt.imshow(magspec_t)
plt.subplot(122)
plt.imshow(outp_t)
#outp_t = (outp_t + np.min(outp_t)) / np.max(outp_t) / 2 * 255
plt.imsave('output.png', outp_t)
plt.show()

print("Took {} secs".format(t1-t0))
