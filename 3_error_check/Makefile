dst=a.out
src=error_check.cu
cpr = /usr/local/cuda/bin/nvcc
$(dst):$(src)
	cpr $(src) -o $(dst);
	nsys profile --stats=true ./$(dst) >nsys.log
.PHONY:clean
clean:
	-rm -f *.out