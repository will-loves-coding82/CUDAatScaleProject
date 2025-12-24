#include "cudaBlur.cuh"

__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void blurKernel(uchar* d_img, uchar* d_blur, int H_in, int W_in, int H_out, int W_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    if (x < W_out && y < H_out) {
        for (int j=0; j < KERNEL_SIZE; j++) {
            for (int k=0; k < KERNEL_SIZE; k++) {
                int dx = x + k;
                int dy = y + j;
                sum += d_kernel[j * KERNEL_SIZE + k] * d_img[dy* W_in + dx];
            }
        }
        if (sum < 0.0f) sum = 0.0f;
        if (sum > 255.0f) sum = 255.0f;
        d_blur[y * W_out + x] = (uchar) sum;
    }
}

void printKernel(float* kernel, int size) {
    printf("Gaussian Kernel %dx%d:\n", size, size);
    printf("[\n");
    
    for (int i = 0; i < size; i++) {
        printf("  ");
        for (int j = 0; j < size; j++) {
            printf("%8.5f", kernel[i * size + j]);
            if (j < size - 1) {
                printf(", ");
            }
        }
        if (i < size - 1) {
            printf(",\n");
        } else {
            printf("\n");
        }
    }
    
    printf("]\n");
    
    // Verify sum = 1
    float sum = 0.0f;
    for (int i = 0; i < size * size; i++) {
        sum += kernel[i];
    }
    printf("Sum of kernel values: %f (should be ~1.0)\n", sum);
}

__host__ float* make_gaussian_window() {
    float* h_gaussian = (float*) malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float sigma = (float) KERNEL_SIZE / 2.0f;

    float sum = 0.0f;
    int radius = KERNEL_SIZE / 2;

    // (0,0) is in the center of the gaussian kernel
    for (int j = -radius; j <= radius; j++) {
        for (int k = -radius; k <= radius; k++) {
            float exponent = -(k*k + j*j) / (2.0f * sigma * sigma);
            float gauss_pdf = expf(exponent) / (2 * M_PI * sigma * sigma);
            h_gaussian[(j+radius) * KERNEL_SIZE + (k+radius)] = gauss_pdf;
            sum += gauss_pdf;
        }
    }

    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
        h_gaussian[i] /= sum;
    }

    return h_gaussian;
}

namespace fs = std::filesystem;

int main() {
    string inputFolder = "./data"; 
    fs::create_directories("./output");
    freopen("logs.txt", "w", stdout);

    if (!fs::exists(inputFolder)) {
        fprintf(stderr, "Error: Input folder '%s' does not exist!\n", inputFolder.c_str());
        return 1;
    }

    // Create 5 x 5 gaussian window for convolution
    float* h_kernel = make_gaussian_window();
    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    printKernel(h_kernel, KERNEL_SIZE);
    printf("==================================================================\n");

    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        string name = entry.path().filename().string();
        printf("Reading input image %s\n", name.c_str());

        cv::Mat img = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            fprintf(stderr, "Error: Could not load image\n");
            return 1;
        }
        int width = img.cols;
        int height = img.rows;
        size_t num_pixels = width * height;
        printf("Input image: %d x %d\n", width, height);

        int H_out = height - KERNEL_SIZE + 1;
        int W_out = width - KERNEL_SIZE + 1;   


        // Initialize device memory for input and output images
        uchar * d_img;
        uchar * d_blur;

        uchar* h_img = (uchar*) malloc(sizeof(uchar) * num_pixels);
        cudaMalloc(&d_img, num_pixels * sizeof(uchar));
        cudaMalloc(&d_blur, H_out * W_out * sizeof(uchar));

        // Load host image data into device input image
        cudaMemcpy(d_img, img.data, num_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Initialize kernel dimensions for output image
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 blocksPerGrid((W_out + BLOCK_SIZE - 1) / BLOCK_SIZE, (H_out + BLOCK_SIZE - 1) , 1);

        // Call the blur kernel and track execution time
        printf("Running Kernel\n");
        
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        blurKernel<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_blur, height, width, H_out, W_out);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("Kernel execution time:  %3.1f ms \n", time);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {  
            fprintf(stderr, "Failed to launch blurKernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy device output image back to host output
        uchar* h_blur = (uchar*) malloc(sizeof(uchar) * H_out * W_out);
        cudaMemcpy(h_blur, d_blur, H_out * W_out * sizeof(uchar), cudaMemcpyDeviceToHost);

        // Create blurred output image as a png
        fs::path p = entry.path();
        string outputFilePath = "./output/" + p.stem().string() + "_blurred" + p.extension().string();
        printf("Writing results to output image with dimensions: %d x %d\n", W_out, H_out);
        cv::Mat outputImg(H_out, W_out, CV_8UC1, h_blur);
        cv::imwrite(outputFilePath, outputImg);

        // Free memory
        cudaFree(d_img);
        cudaFree(d_blur);
        
        printf("Freed device memory\n");    
        printf("==================================================================\n");

    }
    printf("Program completed\n");
    fclose(stdout);
    return 0;
}

