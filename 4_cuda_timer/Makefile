src=record_time.cu
dst=a.out
cpr=/usr/local/cuda/bin/nvcc

$(dst):$(src)
	$(cpr) -o $(dst) $(src)
	nsys profile --stats=true ./$(dst)
.PHONY:clean
clean:
	rm -f *.out
