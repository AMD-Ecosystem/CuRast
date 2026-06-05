
#pragma once

#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <stacktrace>
#include <iostream>

#include "OrbitControls.h"
#include "unsuck.hpp"

#include "glm/common.hpp"

#if defined(USE_HIP)
#include "cuda_to_hip.h"
#else
#include "cuda.h"
#include "cuda_runtime.h"
#endif

using namespace std;

struct CURuntime{

#if defined(USE_HIP)
	inline static hipDevice_t device;
#else
	inline static CUdevice device;
#endif

	CURuntime(){

	}

	static int getNumSMs(){
#if defined(USE_HIP)
		hipDevice_t device;
		int numSMs;
		hipCtxGetDevice(&device);
		hipDeviceGetAttribute(&numSMs, hipDeviceAttributeMultiprocessorCount, device);
#else
		CUdevice device;
		int numSMs;
		cuCtxGetDevice(&device);
		cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
#endif

		return numSMs;
	}


#if defined(USE_HIP)
	static void assertCudaSuccess(hipError_t result, std::stacktrace trace = std::stacktrace::current()){

		if(result == hipSuccess) return;

		println("ERROR: HIP result != hipSuccess.");

		const char* name = hipGetErrorName(result);
		const char* desc = hipGetErrorString(result);

		std::cerr << "HIP error " << int(result) << " ("
			<< (name ? name : "unknown") << "): "
			<< (desc ? desc : "unknown") << "\n";

		std::cerr << to_string(trace) << std::endl;

		__builtin_trap();

		exit(6123453456);
	}
#else
	static void assertCudaSuccess(CUresult result, std::stacktrace trace = std::stacktrace::current()){

		if(result == CUDA_SUCCESS) return;

		println("ERROR: CUDA result != CUDA_SUCCESS.");

		const char* name = nullptr;
		const char* desc = nullptr;
		cuGetErrorName(result, &name);
		cuGetErrorString(result, &desc);

		println(stderr, "CUDA error {} ({}): {}\n ",
			int(result),
			name ? name : "unknown",
			desc ? desc : "unknown");

		std::cerr << to_string(trace) << std::endl;

		__debugbreak();

		exit(6123453456);
	}
#endif

};