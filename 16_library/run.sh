nvcc -c -o obj/cudnn_sample.o cudnn_sample.cu -isystem /home/pinming/miniconda3/include -I/usr/local/cuda/samples/c
ommon/inc/ `pkg-config opencv --cflags --libs`
# for nvcc -Xlinker -rpath=/usr/lib/x86_64-linux-gnu/
# for gcc -Wl,-rpath=/usr/local/lib,
