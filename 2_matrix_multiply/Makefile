TARGET=a.out
SOURCE=gemm.cu
COMPILER=/usr/local/cuda/bin/nvcc

$(TARGET):$(SOURCE)
	$(COMPILER) -o $(TARGET) $(SOURCE) --run

.PHONY: clean
clean:
	-rm -f $(TARGET)