src=get_topk.cu
dst=a.out
cpr=/usr/local/cuda/bin/nvcc

${dst}:${src}
	${cpr} -o ${dst} ${src}
	# nsys profile --stats=true ${dst} > log.log
.PHONY:clean
clean:
	rm -f ${dst}
