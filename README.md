# Gaussian Blur Image Convolution using CUDA

This project implements a high-performance 2D Gaussian Blur using NVIDIA’s CUDA C++ API. By offloading the computationally intensive convolution operation to the GPU, the system achieves significant speedups compared to traditional CPU-based processing. The smoothing effect is achieved through a Multivariate Gaussian Kernel. The kernel samples values from a normalized probability density function (PDF) across two dimensions since we are working with images.

## Technical Architecture
To optimize throughput and minimize latency, the implementation utilizes several specific CUDA memory hierarchies and strategies:

- The Gaussian weights are cached in Constant/Texture Memory with a configurable `KERNEL_SIZE` defined in the program's header files. This leverages the dedicated hardware-level cache on the GPU, significantly reducing global memory traffic during the frequent read operations required for the convolution stencil.
- The convolution is implemented as a "valid" transformation, meaning no artificial padding is applied. Consequently, the output image dimensions are reduced by a factor proportional to the kernel radius.
- The workload is partitioned into a 2D grid of thread blocks, where each thread calculates the weighted average for a single pixel, maximizing the occupancy of the GPU's Streaming Multiprocessors (SMs).
## Before 
<img src="data/pug.jpg" width=400>

## After 21 x 21 convolution
<img src="output/pug_blurred.jpg" width=400>


## Code Organization

```bin/```
This folder holds the binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder holds all example images used for image processing.

```src/```
Contains the source code to blur the images defined in `cudaBlur.cuh` and `cudaBlur.cu`.

```Makefile or CMAkeLists.txt or build.sh```
There should be some rudimentary scripts for building the project's code in an automatic fashion.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments. To execute this, open the command line prompt and type: 
```
source run.sh
```

## Key Concepts

Performance Strategies, Image Processing, Cuda Events

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, ppc64le, armv7l


## Dependencies needed to build/run
[FreeImage](../../README.md#freeimage), [NPP](../../README.md#npp)

## Prerequisites

Download and install the [CUDA Toolkit 11.4](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run

### Windows
The Windows samples are built using the Visual Studio IDE. Solution files (.sln) are provided for each supported version of Visual Studio, using the format:
```
*_vs<version>.sln - for Visual Studio <version>
```
Each individual sample has its own set of solution files in its directory:

To build/examine all the samples at once, the complete solution files should be used. To build/examine a single sample, the individual sample solution files should be used.
> **Note:** Some samples require that the Microsoft DirectX SDK (June 2010 or newer) be installed and that the VC++ directory paths are properly set up (**Tools > Options...**). Check DirectX Dependencies section for details."

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
```
The samples makefiles can take advantage of certain options:
*  **TARGET_ARCH=<arch>** - cross-compile targeting a specific architecture. Allowed architectures are x86_64, ppc64le, armv7l.
    By default, TARGET_ARCH is set to HOST_ARCH. On a x86_64 machine, not setting TARGET_ARCH is the equivalent of setting TARGET_ARCH=x86_64.<br/>
`$ make TARGET_ARCH=x86_64` <br/> `$ make TARGET_ARCH=ppc64le` <br/> `$ make TARGET_ARCH=armv7l` <br/>
    See [here](http://docs.nvidia.com/cuda/cuda-samples/index.html#cross-samples) for more details.
*   **dbg=1** - build with debug symbols
    ```
    $ make dbg=1
    ```
*   **SMS="A B ..."** - override the SM architectures for which the sample will be built, where `"A B ..."` is a space-delimited list of SM architectures. For example, to generate SASS for SM 50 and SM 60, use `SMS="50 60"`.
    ```
    $ make SMS="50 60"
    ```

*  **HOST_COMPILER=<host_compiler>** - override the default g++ host compiler. See the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements) for a list of supported host compilers.
```
    $ make HOST_COMPILER=g++
```


## Running the Program
After building the project, you can run the program using the following command:

```bash
Copy code
make run
```

This command will execute the compiled binary, rotating the input image (Lena.png) by 45 degrees, and save the result as Lena_rotated.png in the data/ directory.

If you wish to run the binary directly with custom input/output files, you can use:

```bash
- Copy code
./bin/cudaBlur
```

- Cleaning Up
To clean up the compiled binaries and other generated files, run:


```bash
- Copy code
make clean
```

This will remove all files in the bin/ directory.
