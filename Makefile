# Default target
# Run when you call "make"
all: compile_run
 
# 						v------------------v dependencies
# Make will check if these changed before building 
compile_run : main.cu ray.h vec3.h
	nvcc -o gpu_rt main.cu 
	./gpu_rt > out.ppm

clean:
	rm out.ppm gpu_rt