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

<%def name='sine(kernel_declaration, k_output, k_input_)'>
${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;
    const VSIZE_T idx0 = virtual_global_id(0);
    const VSIZE_T idx1 = virtual_global_id(1);

    ${k_input_.ctype} val = ${k_input_.load_idx}(idx0, idx1);
    ${k_output.store_idx}(idx0, idx1, sin(val));
}
</%def>

<%def name='cosine(kernel_declaration, k_output, k_input_)'>
${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;
    const VSIZE_T idx0 = virtual_global_id(0);
    const VSIZE_T idx1 = virtual_global_id(1);

    ${k_input_.ctype} val = ${k_input_.load_idx}(idx0, idx1);
    ${k_output.store_idx}(idx0, idx1, cos(val));
}
</%def>
