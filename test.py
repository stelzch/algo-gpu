#!/usr/bin/python3
import time
import numpy as np
import matplotlib.pyplot as plt
from reikna import cluda
from reikna.cluda import Snippet, dtypes
from reikna.core import Transformation, Type, Annotation, Parameter, Computation
from reikna.helpers import template_from
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

        crop_template = template_from(
            """
            <%def name='cropcomp(kernel_declaration, k_output, k_input_, k_rect_x, k_rect_y, k_rect_r)'>
            ${kernel_declaration}
            {
                VIRTUAL_SKIP_THREADS;
                const VSIZE_T square_size = virtual_global_size(0);
                const VSIZE_T idx0 = virtual_global_id(0);
                const VSIZE_T idx1 = virtual_global_id(1);

                const VSIZE_T new_idx0 = (idx0 - ${k_rect_y} + square_size / 2) % square_size;
                const VSIZE_T new_idx1 = (idx1 - ${k_rect_x} + square_size / 2) % square_size;

                if(new_idx0 < square_size / 2 - ${k_rect_r} ||
                   new_idx0 > square_size / 2 + ${k_rect_r} ||
                   new_idx1 < square_size / 2 - ${k_rect_r} ||
                   new_idx1 > square_size / 2 + ${k_rect_r}) {
                   ${k_output.store_idx}(new_idx0, new_idx1, 0);

                } else {
                    ${k_input_.ctype} value = ${k_input_.load_idx}(idx0, idx1);
                    ${k_output.store_idx}(new_idx0, new_idx1, value);
                }

            }
            </%def>
            """)
        magspec_template = template_from(
        """
        <%def name='magspec(kernel_declaration, k_output, k_input_)'>
        ${kernel_declaration}
        {
            VIRTUAL_SKIP_THREADS;
            const VSIZE_T idx0 = virtual_global_id(0);
            const VSIZE_T idx1 = virtual_global_id(1);

            ${k_input_.ctype} complex_value = ${k_input_.load_idx}(idx0, idx1);
            ${k_output.ctype} absolute_val = log(sqrt(pow(complex_value.x, 2) + pow(complex_value.y, 2)));
            ${k_output.store_idx}(idx0, idx1, absolute_val);
        }
        </%def>

        """)
        
        plan.kernel_call(
            crop_template.get_def('cropcomp'),
            [temp2, temp, rect_x, rect_y, rect_r],
            global_size=output.shape)
        """
        plan.kernel_call(
            magspec_template.get_def('magspec'),
            [output, temp2],
            global_size=output.shape)
        """
        plan.computation_call(fftshift, temp2, temp2)
        plan.computation_call(fft, temp, temp2, 1)
        angle_template = template_from(
        """
        <%def name='angle(kernel_declaration, k_output, k_input_)'>
        ${kernel_declaration}
        {
            VIRTUAL_SKIP_THREADS;
            const VSIZE_T idx0 = virtual_global_id(0);
            const VSIZE_T idx1 = virtual_global_id(1);


            ${k_input_.ctype} complex_value = ${k_input_.load_idx}(idx0, idx1);
            ${k_output.ctype} n_angle = atan2(complex_value.y, complex_value.x);
            ${k_output.store_idx}(idx0, idx1, n_angle);
        }
        </%def>
        """)
        plan.kernel_call(
            angle_template.get_def('angle'),
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
plt.figure()
phasecompc(outp_dev, data_dev, 1585, 1007, 24)
outp_t = outp_dev.get()
print(outp_t)


plt.imshow(outp_t)
plt.show()
t1 = time.time()

print("Took {} secs".format(t1-t0))
