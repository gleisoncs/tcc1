import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt

def mandelbrot_opencl(width, height, xmin, xmax, ymin, ymax, max_iterations):
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    real = np.linspace(xmin, xmax, width).astype(np.float32)
    imag = np.linspace(ymin, ymax, height).astype(np.float32)
    real, imag = np.meshgrid(real, imag)
    complex_plane = real + 1j * imag

    mandelbrot = np.empty(complex_plane.shape, dtype=np.uint32)

    real_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=real)
    imag_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=imag)
    mandelbrot_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, mandelbrot.nbytes)

    prg = cl.Program(ctx, """
        #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
        __kernel void mandelbrot(__global const float *real,
                                 __global const float *imag,
                                 __global uint *mandelbrot,
                                 const uint max_iterations)
        {
            int gid_x = get_global_id(0);
            int gid_y = get_global_id(1);
            float cx = real[gid_x];
            float cy = imag[gid_y];
            float x = 0.0f;
            float y = 0.0f;
            uint iteration = 0;
            while (x*x + y*y <= 4.0 && iteration < max_iterations) {
                float x_new = x*x - y*y + cx;
                float y_new = 2*x*y + cy;
                x = x_new;
                y = y_new;
                iteration++;
            }
            mandelbrot[gid_y * get_global_size(0) + gid_x] = iteration;
        }
    """).build()

    prg.mandelbrot(queue, mandelbrot.shape, None, real_buf, imag_buf, mandelbrot_buf, np.uint32(max_iterations))

    cl.enqueue_copy(queue, mandelbrot, mandelbrot_buf).wait()

    return mandelbrot

if __name__ == "__main__":
    width = 15360
    height = 8640
    xmin, xmax = -2, 1
    ymin, ymax = -1.5, 1.5
    max_iterations = 5000

    mandelbrot = mandelbrot_opencl(width, height, xmin, xmax, ymin, ymax, max_iterations)

    plt.imshow(np.log(mandelbrot), extent=(xmin, xmax, ymin, ymax))
    plt.title("Mandelbrot Set (OpenCL)")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.colorbar(label="Log Iterations")
    plt.show()
