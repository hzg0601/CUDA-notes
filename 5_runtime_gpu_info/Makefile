src=query_runtime_gpu_info.cu
dst=a.out
cpr=/usr/local/cuda/bin/nvcc

$(dst):$(src)
	$(cpr) $(src) -o $(dst)
	# nsys profile --stats=true ./$(dst) > $(dst).log
.PHONY:clean
clean:
	rm -f *.out
	