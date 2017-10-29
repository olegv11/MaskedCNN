rm -f ./main.o ./main.dlink.o libmaskedcnncuda.a
#ordinary rdc compilation of CUDA source
nvcc -ccbin g++-5 -m64 -arch=sm_60 -dc -o main.o -c main.cu
#separate device link step, necessary for rdc flow
nvcc -ccbin g++-5 -m64 -arch=sm_60 -dlink -o main.dlink.o main.o



#creation of library - note we need ordinary linkable object and device-link object!
nvcc -ccbin g++-5 -m64 -arch=sm_60 -lib -o libmaskedcnncuda.a main.o main.dlink.o



#host code compilation
#g++ -m64 -o main.o -c main.cpp
#host (final) link phase - the order of entries on this line is important!!
#g++ -m64 main.o MWE.a -o test -L/usr/local/cuda/lib64 -lcudart
