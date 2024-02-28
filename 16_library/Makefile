IDIR = /usr/local/cuda/samples/common/inc/
CC=nvcc
# CFLAGS+=-I$(IDIR) `pkg-config opencv --cflags --libs` 
CFLAGS=-I$(IDIR) `pkg-config opencv --cflags --libs` -isystem /home/pinming/miniconda3/include
# LDFLAGS+= -lcufft -lopencv_imgproc -lopencv_core -lopencv_highgui
# LDFLAGS=-Xlinker,-rpath,/home/pinming/miniconda3/lib -Xlinker,-rpath-link,/home/pinming/miniconda3/lib -L/home/pinming/miniconda3/lib -lcudnn -lopencv_imgproc -lopencv_core -lopencv_highgui
LDFLAGS= -Xlinker -O2 -Xlinker --sort-common -Xlinker --as-needed -Xlinker -z,relro -Xlinker -z,now -Xlinker --disable-new-dtags -Xlinker --gc-sections -Xlinker --allow-shlib-undefined -Xlinker -rpath,/home/pinming/miniconda3/lib -Xlinker -rpath-link,/home/pinming/miniconda3/lib -L/home/pinming/miniconda3/lib -lcudnn -lopencv_imgproc -lopencv_core -lopencv_highgui
ODIR=obj
dummy_build_folder := $(shell mkdir -p $(ODIR))
# OBJ 使用 patsubst 函数将 _OBJ 中的每个文件名前加上 $(ODIR)/ 的前缀，生成了目标文件的完整路径。
_OBJ = cudnn_sample.o  
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))
# $@ 表示目标文件的名称，$< 表示第一个依赖文件（即对应的 .cu 文件），$(CFLAGS) 包含了编译选项。
$(ODIR)/%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)


convolution: $(OBJ)
	nvcc -o $@ $^ $(CFLAGS) $(LDFLAGS)
.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
 
