#!/usr/bin/python3
import time
import numpy as np
import matplotlib.pyplot as plt
from reikna import cluda
from reikna.cluda import Snippet, dtypes
from reikna.core import Transformation, Type, Annotation, Parameter, Computation
from reikna.helpers import template_for
from reikna.algorithms import PureParallel
from reikna.fft.fftshift import FFTShift
from reikna.fft.fft import FFT


width = height = 2048



class PhaseComputation(Computation):
    def __init__(self, output, input_, rect_x, rect_y, rect_r, square_size):
        Computation.__init__(self, [
            Parameter('output', Annotation(output, 'o')),
            Parameter('input', Annotation(input_, 'i')),
            Parameter('rect_x', Annotation(Type(np.uint16, shape=()), 's')),
            Parameter('rect_y', Annotation(Type(np.uint16, shape=()), 's')),
            Parameter('rect_r', Annotation(Type(np.uint16, shape=()), 's'))])
        self.square_size = square_size

    def _build_plan(self, plan_factory, device_params, output, input_, rect_x, rect_y, rect_r):
        print(output.shape)
        assert output.shape[0] == output.shape[1] == input_.shape[0] == input_.shape[1]
        plan = plan_factory()

        temp = plan.temp_array_like(input_)
        temp2 = plan.temp_array_like(input_)
        fft = FFT(temp, axes=(0,1))
        fftshift = FFTShift(temp, axes=(0,1))

        plan.computation_call(fft, temp, input_, 0)
        plan.computation_call(fftshift, temp, temp)

        template = template_for("kernels") # Read template from kernels.mako
        
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

        return plan



api = cluda.ocl_api()
thr = api.Thread.create()

img = plt.imread('/home/cstelz/holmos_raw.png')
n = 2
data = np.kron(img, np.ones((n,n))).astype(np.complex128)
print(data.shape)
outp = np.zeros((2048, 2048)).astype(np.float64)
phasecomp = PhaseComputation(outp, data, 50.0, 50, 50, 2048)
phasecompc = phasecomp.compile(thr)

data_dev = thr.to_device(data)
outp_dev = thr.to_device(outp)

t0 = time.time()
for i in range(0, 24):
    phasecompc(outp_dev, data_dev, 1585, 1007, 24)
    #outp_t = outp_dev.get()
#print(outp_t)


t1 = time.time()

print("Took {} secs".format(t1-t0))
