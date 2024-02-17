SOURCE=hello_cuda.cu 
TARGET=hello_cuda.exe
COMPILER=/usr/local/cuda/bin/nvcc

$(TARGET):$(SOURCE)
	$(COMPILER) $(SOURCE) -o $(TARGET)
	
.PHONY:clean
clean:
	-rm $(TARGET)

