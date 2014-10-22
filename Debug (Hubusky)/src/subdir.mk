################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/particleSystem.cpp \
../src/particles.cpp \
../src/render_particles.cpp \
../src/shaders.cpp 

CU_SRCS += \
../src/particleSystem_cuda.cu 

CU_DEPS += \
./src/particleSystem_cuda.d 

OBJS += \
./src/particleSystem.o \
./src/particleSystem_cuda.o \
./src/particles.o \
./src/render_particles.o \
./src/shaders.o 

CPP_DEPS += \
./src/particleSystem.d \
./src/particles.d \
./src/render_particles.d \
./src/shaders.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/5_Simulations" -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/common/inc" -I"/home/rkeedy/Dropbox/cuda/particles" -G -g -O0 -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/5_Simulations" -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/common/inc" -I"/home/rkeedy/Dropbox/cuda/particles" -G -g -O0 -m64 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/5_Simulations" -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/common/inc" -I"/home/rkeedy/Dropbox/cuda/particles" -G -g -O0 -m64 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/5_Simulations" -I"/home/rkeedy/NVIDIA_CUDA-5.0_Samples/common/inc" -I"/home/rkeedy/Dropbox/cuda/particles" -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


