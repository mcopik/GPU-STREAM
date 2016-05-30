/*=============================================================================
*------------------------------------------------------------------------------
* Copyright 2015: Tom Deakin, Simon McIntosh-Smith, University of Bristol HPC
* Based on John D. McCalpin’s original STREAM benchmark for CPUs
*------------------------------------------------------------------------------
* License:
*  1. You are free to use this program and/or to redistribute
*     this program.
*  2. You are free to modify this program for your own use,
*     including commercial use, subject to the publication
*     restrictions in item 3.
*  3. You are free to publish results obtained from running this
*     program, or from works that you derive from this program,
*     with the following limitations:
*     3a. In order to be referred to as "GPU-STREAM benchmark results",
*         published results must be in conformance to the GPU-STREAM
*         Run Rules published at
*         http://github.com/UoB-HPC/GPU-STREAM/wiki/Run-Rules
*         and incorporated herein by reference.
*         The copyright holders retain the
*         right to determine conformity with the Run Rules.
*     3b. Results based on modified source code or on runs not in
*         accordance with the GPU-STREAM Run Rules must be clearly
*         labelled whenever they are published.  Examples of
*         proper labelling include:
*         "tuned GPU-STREAM benchmark results"
*         "based on a variant of the GPU-STREAM benchmark code"
*         Other comparable, clear and reasonable labelling is
*         acceptable.
*     3c. Submission of results to the GPU-STREAM benchmark web site
*         is encouraged, but not required.
*  4. Use of this program or creation of derived works based on this
*     program constitutes acceptance of these licensing restrictions.
*  5. Absolutely no warranty is expressed or implied.
*———————————————————————————————————-----------------------------------------*/


#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cfloat>
#include <cmath>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
//#include "CL/cl2.hpp"
#include "common.h"

using namespace cl::sycl;

std::string getDeviceName(const sycl::device & device);
std::string getDeviceDriver(const sycl::device& device);
std::vector<sycl::device> getDeviceList(;


// Print error and exit
void die(std::string msg, sycl::exception & e)
{
    std::cerr
            << "Error: "
            << msg
            << ": " << e.what()
            << std::endl;
    exit();
}


int main(int argc, char *argv[])
{

    // Print out run information
    std::cout
        << "GPU-STREAM" << std::endl
        << "Version: " << VERSION_STRING << std::endl
        << "Implementation: Khronos SYCL" << std::endl;

    std::string status;

    try
    {
        parseArguments(argc, argv);
        if (NTIMES < 2)
            throw std::runtime_error("Chosen number of times is invalid, must be >= 2");


        std::cout << "Precision: ";
        if (useFloat) std::cout << "float";
        else std::cout << "double";
        std::cout << std::endl << std::endl;

        std::cout << "Running kernels " << NTIMES << " times" << std::endl;

        if (ARRAY_SIZE % 1024 != 0)
        {
            unsigned int OLD_ARRAY_SIZE = ARRAY_SIZE;
            ARRAY_SIZE -= ARRAY_SIZE % 1024;
            std::cout
                << "Warning: array size must divide 1024" << std::endl
                << "Resizing array from " << OLD_ARRAY_SIZE
                << " to " << ARRAY_SIZE << std::endl;
            if (ARRAY_SIZE == 0)
                throw std::runtime_error("Array size must be >= 1024");
        }

        // Get precision (used to reset later)
        std::streamsize ss = std::cout.precision();

        size_t DATATYPE_SIZE;

        if (useFloat)
        {
            DATATYPE_SIZE = sizeof(float);
        }
        else
        {
            DATATYPE_SIZE = sizeof(double);
        }

        // Display number of bytes in array
        std::cout << std::setprecision(1) << std::fixed
            << "Array size: " << ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0 << " MB"
            << " (=" << ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0/1024.0 << " GB)"
            << std::endl;
        std::cout << "Total size: " << 3.0*ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0 << " MB"
            << " (=" << 3.0*ARRAY_SIZE*DATATYPE_SIZE/1024.0/1024.0/1024.0 << " GB)"
            << std::endl;

        // Reset precision
        std::cout.precision(ss);


        // Setup OpenCL

        // Get list of devices
        std::vector<cl::sycl::device> devices = getDeviceList();

        // Check device index is in range
        if (deviceIndex >= devices.size())
            throw std::runtime_error("Chosen device index is invalid");

        cl::sycl::device device = devices[deviceIndex];

        status = "Creating queue";
        cl::sycl::queue queue(device);

        // Print out device name
        std::string name = getDeviceName(device);
        std::cout << "Using OpenCL device " << device.get_info( << std::endl;

        // Print out OpenCL driver version for this device
        std::string driver = getDeviceDriver(device);
        std::cout << "Driver: " << driver << std::endl;

        // Check device can do double precision if requested
        if (!useFloat && !device.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>())
            throw std::runtime_error("Device does not support double precision, please use --float");

        // Check buffers fit on the device
        status = "Getting device memory sizes";
        cl_ulong totalmem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
        cl_ulong maxbuffer = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        if (maxbuffer < DATATYPE_SIZE*ARRAY_SIZE)
            throw std::runtime_error("Device cannot allocate a buffer big enough");
        if (totalmem < 3*DATATYPE_SIZE*ARRAY_SIZE)
            throw std::runtime_error("Device does not have enough memory for all 3 buffers");

        try
        {
            std::string options = "";
            if (useFloat)
                options = "-DFLOAT";
            program.build(options.c_str());
        }
        catch (cl::Error& e)
        {
            std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
            std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cerr
                << "Build error:"
                << buildlog
                << std::endl;
            throw e;
        }

        status = "Making kernel copy";
        auto copy = cl::KernelFunctor<cl::Buffer&, cl::Buffer&>(program, "copy");
        status = "Making kernel mul";
        auto mul = cl::KernelFunctor<cl::Buffer&, cl::Buffer&>(program, "mul");
        status = "Making kernel add";
        auto add = cl::KernelFunctor<cl::Buffer&, cl::Buffer&, cl::Buffer&>(program, "add");
        status = "Making kernel triad";
        auto triad = cl::KernelFunctor<cl::Buffer&, cl::Buffer&, cl::Buffer&>(program, "triad");

        // Create host vectors
        void *h_a = malloc(ARRAY_SIZE * DATATYPE_SIZE);
        void *h_b = malloc(ARRAY_SIZE * DATATYPE_SIZE);
        void *h_c = malloc(ARRAY_SIZE * DATATYPE_SIZE);

        // Initilise arrays
        for (unsigned int i = 0; i < ARRAY_SIZE; i++)
        {
            if (useFloat)
            {
                ((float*)h_a)[i] = 1.0f;
                ((float*)h_b)[i] = 2.0f;
                ((float*)h_c)[i] = 0.0f;
            }
            else
            {
                ((double*)h_a)[i] = 1.0;
                ((double*)h_b)[i] = 2.0;
                ((double*)h_c)[i] = 0.0;
            }
        }

        // Create device buffers
        status = "Creating buffers";
        cl::Buffer d_a(context, CL_MEM_READ_WRITE, DATATYPE_SIZE * ARRAY_SIZE);
        cl::Buffer d_b(context, CL_MEM_READ_WRITE, DATATYPE_SIZE * ARRAY_SIZE);
        cl::Buffer d_c(context, CL_MEM_READ_WRITE, DATATYPE_SIZE * ARRAY_SIZE);


        // Copy host memory to device
        status = "Copying buffers";
        queue.enqueueWriteBuffer(d_a, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_a);
        queue.enqueueWriteBuffer(d_b, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_b);
        queue.enqueueWriteBuffer(d_c, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_c);

        // Make sure the copies are finished
        queue.finish();


        // List of times
        std::vector< std::vector<double> > timings;

        // Declare timers
        std::chrono::high_resolution_clock::time_point t1, t2;

        // Main loop
        for (unsigned int k = 0; k < NTIMES; k++)
        {
            status = "Executing copy";
            std::vector<double> times;
            t1 = std::chrono::high_resolution_clock::now();
            copy(
                cl::EnqueueArgs(
                queue,
                cl::NDRange(ARRAY_SIZE)),
                d_a, d_c);
            queue.finish();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


            status = "Executing mul";
            t1 = std::chrono::high_resolution_clock::now();
            mul(
                cl::EnqueueArgs(
                queue,
                cl::NDRange(ARRAY_SIZE)),
                d_b, d_c);
            queue.finish();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


            status = "Executing add";
            t1 = std::chrono::high_resolution_clock::now();
            add(
                cl::EnqueueArgs(
                queue,
                cl::NDRange(ARRAY_SIZE)),
                d_a, d_b, d_c);
            queue.finish();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


            status = "Executing triad";
            t1 = std::chrono::high_resolution_clock::now();
            triad(
                cl::EnqueueArgs(
                queue,
                cl::NDRange(ARRAY_SIZE)),
                d_a, d_b, d_c);
            queue.finish();
            t2 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

            timings.push_back(times);

        }

        // Check solutions
        status = "Copying back buffers";
        queue.enqueueReadBuffer(d_a, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_a);
        queue.enqueueReadBuffer(d_b, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_b);
        queue.enqueueReadBuffer(d_c, CL_FALSE, 0, ARRAY_SIZE*DATATYPE_SIZE, h_c);
        queue.finish();


        if (useFloat)
        {
            check_solution<float>(h_a, h_b, h_c);
        }
        else
        {
            check_solution<double>(h_a, h_b, h_c);
        }

        // Crunch results
        size_t sizes[4] = {
            2 * DATATYPE_SIZE * ARRAY_SIZE,
            2 * DATATYPE_SIZE * ARRAY_SIZE,
            3 * DATATYPE_SIZE * ARRAY_SIZE,
            3 * DATATYPE_SIZE * ARRAY_SIZE
        };
        double min[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
        double max[4] = {0.0, 0.0, 0.0, 0.0};
        double avg[4] = {0.0, 0.0, 0.0, 0.0};
        // Ignore first result
        for (unsigned int i = 1; i < NTIMES; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                avg[j] += timings[i][j];
                min[j] = std::min(min[j], timings[i][j]);
                max[j] = std::max(max[j], timings[i][j]);
            }
        }
        for (int j = 0; j < 4; j++)
            avg[j] /= (double)(NTIMES-1);

        // Display results
        std::string labels[] = {"Copy", "Mul", "Add", "Triad"};
        std::cout
            << std::left << std::setw(12) << "Function"
            << std::left << std::setw(12) << "MBytes/sec"
            << std::left << std::setw(12) << "Min (sec)"
            << std::left << std::setw(12) << "Max"
            << std::left << std::setw(12) << "Average"
            << std::endl;
        for (int j = 0; j < 4; j++)
        {
            std::cout
                << std::left << std::setw(12) << labels[j]
                << std::left << std::setw(12) << std::setprecision(3) << 1.0E-06 * sizes[j]/min[j]
                << std::left << std::setw(12) << std::setprecision(5) << min[j]
                << std::left << std::setw(12) << std::setprecision(5) << max[j]
                << std::left << std::setw(12) << std::setprecision(5) << avg[j]
                << std::endl;
        }

        // Free host vectors
        free(h_a);
        free(h_b);
        free(h_c);

    }
    catch (cl::Error &e)
    {
        die(status, e);
    }
    catch (std::exception& e)
    {
        std::cerr
            << "Error: "
            << e.what()
            << std::endl;
        exit(EXIT_FAILURE);
    }

}


std::vector<sycl::device> getDeviceList()
{
    // Get list of platforms
    try
    {
        return sycl::device::get_devices(sycl::info::device_type::all);
    }
    catch (sycl::exception &e)
    {
        die("Getting platforms", e);
    }
}


std::string getDeviceName(const cl::Device& device)
{
    try
    {
        return device.get_info(sycl::info::device::name);
    }
    catch (sycl::exception &e)
    {
        die("Getting device name", e);
    }
}

std::string getDeviceDriver(const sycl::device& device)
{
    try
    {
        return device.get_info(sycl::info::device::version);
    }
    catch (sycl::exception &e)
    {
        die("Getting device driver", e);
    }
}


void listDevices(void)
{
    // Get list of devices
    std::vector<sycl::device> devices = getDeviceList();

    // Print device names
    if (devices.size() == 0)
    {
        std::cout << "No devices found." << std::endl;
    }
    else
    {
        std::cout << std::endl;
        std::cout << "Devices:" << std::endl;
        for (unsigned i = 0; i < devices.size(); i++)
        {
            std::cout << i << ": " << getDeviceName(devices[i]) << std::endl;
        }
        std::cout << std::endl;
    }
}

