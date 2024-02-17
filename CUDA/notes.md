# 一、预备知识

## 1.1 nvcc

- nvcc就是CUDA的编译器，cuda程序有两种代码，在cpu上的host代码和在gpu上的device代码。
- .cu后缀：cuda源代码，包括host和device代码。

### 1.1.1 常用命令



### 1.1.2 利用Makefile执行nvcc命令：

```makefile
TEST_SOURCE = vectorAdd.cu 

TARGETBIN := ./vectorAdd

CC = /usr/local/cuda/bin/nvcc #指定编译器

$(TARGETBIN):$(TEST_SOURCE)
	$(CC)  $(TEST_SOURCE) -o $(TARGETBIN)

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)
```

## 1.2 Makefile

### 1.2.1 Makefile文件是什么？

Makefile 文件描述了 Linux 系统下 C/C++ 工程的编译规则，它用来自动化编译 C/C++ 项目。一旦写编写好 Makefile 文件，只需要一个 make 命令，整个工程就开始自动编译，不再需要手动执行 GCC 命令。

一个中大型 C/C++ 工程的源文件有成百上千个，它们按照功能、模块、类型分别放在不同的目录中，Makefile 文件定义了一系列规则，指明了源文件的编译顺序、依赖关系、是否需要重新编译等。

[Makefile文件是什么？ (biancheng.net)](http://c.biancheng.net/view/7097.html)

### 1.2.2 Makefile文件编写规则

其结构如下：

```makefile
targets : prerequisites
    command
```

或者是

```makefile
targets : prerequisites; command
    command
```

> targets：规则的目标，可以是Object File（中间文件），也可以是可执行文件，还可以是一个标签。
>
> prerequisites：是我们的依赖文件，要生产targets需要的文件，可以有多个，也可以没有。
>
> command：make需要执行的命令（任意shell命令），可以有多条命令，每一条命令占一行。
>
> **注意：我们的目标和依赖文件之间要使用冒号分隔开，命令的开始一定要使用`Tab`键。**

通过下面的例子来具体使用一下 Makefile 的规则，Makefile文件中添代码如下：

```makefile
test:test.c
    gcc -o test test.c
```

上述代码实现的功能就是编译 test.c 文件，通过这个实例可以详细的说明 Makefile 的具体的使用。其中 test 是的目标文件，也是我们的最终生成的可执行文件。依赖文件就是 test.c 源文件，重建目标文件需要执行的操作是`gcc -o test test.c`。

> 使用 Makefile 的方式：首先需要编写好 Makefile 文件，然后在 shell 中执行 make 命令，程序就会自动执行，得到最终的目标文件。

简单的概括一下Makefile 中的内容，它主要包含有五个部分，分别是：

1)  **显式规则**

显式规则说明了，如何生成一个或多的的目标文件。这是由 Makefile 的书写者明显指出，要生成的文件，文件的依赖文件，生成的命令。

2)  **隐晦规则**

由于我们的 make 命名有自动推导的功能，所以隐晦的规则可以让我们比较粗糙地简略地书写 Makefile，这是由 make 命令所支持的。

3)  **变量的定义**

在 Makefile 中我们要定义一系列的变量，变量一般都是字符串，这个有点像C语言中的宏，当 Makefile 被执行时，其中的变量都会被扩展到相应的引用位置上。

4) **文件指示**

其包括了三个部分，一个是在一个 Makefile 中引用另一个 Makefile，就像C语言中的 include 一样；另一个是指根据某些情况指定 Makefile 中的有效部分，就像C语言中的预编译 #if 一样；还有就是定义一个多行的命令。有关这一部分的内容，我会在后续的部分中讲述。

5)  **注释**

Makefile 中只有行注释，和 UNIX 的 Shell 脚本一样，其注释是用“#”字符，这个就像 C/[C++](http://c.biancheng.net/cplus/) 中的“//”一样。如果你要在你的 Makefile 中使用“#”字符，可以用反斜框进行转义，如：“\#”。

### 1.2.3 工作流程

当我们在执行 make 条命令的时候，make 就会去当前文件下找要执行的编译规则，也就是 Makefile 文件。我们编写 Makefile 的时可以使用的文件的名称 "GNUmakefile" 、"makefile" 、"Makefile" ，make 执行时回去寻找 Makefile 文件，找文件的顺序也是这样的。

我们推荐使用 Makefile（一般在工程中都这么写，大写的会比较的规范）。

Makefile 的具体工作流程可以通过例子来看一下：创建一个包含有多个源文件和 Makefile 的目录文件，源文件之间相互关联。在 Makefile 中添加下面的代码：

```makefile
main:main.o test1.o test2.o
gcc main.o test1.o test2.o -o main
main.o:main.c test.h
gcc -c main.c -o main.o
test1.o:test1.c test.h
gcc -c test1.c -o test1.o
test2.o:test2.c test.h
gcc -c test2.c -o test2.o
```

在我们编译项目文件的时候，默认情况下，make 执行的是 Makefile 中的第一规则（Makefile 中出现的第一个依赖关系），**此规则的第一目标称之为“最终目标”或者是“终极目标”。**

在 shell 命令行执行的 make 命令，就可以得到可执行文件 main 和中间文件 main.o、test1.o 和 test2.o，main 就是我们要生成的最终文件。**通过 Makefile 我们可以发现，目标 main"在 Makefile 中是第一个目标，因此它就是 make 的终极目标，当修改过任何 C 文件后，执行 make 将会重建终极目标 main。**

它的具体工作顺序是：

- 当在 shell 提示符下输入 make 命令以后。 make 读取当前目录下的 Makefile 文件，并将 Makefile 文件中的第一个目标作为其执行的“终极目标”；
- 开始处理第一个规则（终极目标所在的规则）。在我们的例子中，第一个规则就是目标 "main" 所在的规则。规则描述了 "main" 的依赖关系，并定义了链接 ".o" 文件生成目标 "main" 的命令；
- make 在执行这个规则所定义的命令之前，首先处理目标 "main" 的所有的依赖文件的更新规则（以这些 ".o" 文件为目标的规则）。

对这些 ".o" 文件为目标的规则处理有下列三种情况：

- 目标 ".o" 文件不存在，使用其描述规则创建它；
- 目标 ".o" 文件存在，目标 ".o" 文件所依赖的 ".c" 源文件 ".h" 文件中的任何一个比目标 ".o" 文件“更新”（在上一次 make 之后被修改）。则根据规则重新编译生成它；
- 目标 ".o" 文件存在，目标 ".o" 文件比它的任何一个依赖文件（".c" 源文件、".h" 文件）“更新”（它的依赖文件在上一次 make 之后没有被修改），则什么也不做。


通过上面的更新规则我们可以了解到中间文件的作用，也就是编译时生成的 ".o" 文件。作用是检查某个源文件是不是进行过修改，最终目标文件是不是需要重建。我们执行 make 命令时，只有修改过的源文件或者是不存在的目标文件会进行重建，而那些没有改变的文件不用重新编译，这样在很大程度上节省时间，提高编程效率。小的工程项目可能体会不到，项目工程文件越大，效果才越明显。

当然 make 命令能否顺利的执行，还在于我们是否制定了正确的的依赖规则，当前目录下是不是存在需要的依赖文件，只要任意一点不满足，我们在执行 make 的时候就会出错。所以完成一个正确的 Makefile 不是一件简单的事情。

### 1.2.4 清除工作目录中的过程文件

我们在使用的时候会产生中间文件会让整个文件看起来很乱，所以在编写 Makefile 文件的时候会在末尾加上这样的规则语句：

```makefile
.PHONY:clean
clean:
    rm -rf *.o test
```

其中 "*.o" 是执行过程中产生的中间文件，"test" 是最终生成的执行文件。我们可以看到 clean 是独立的，它只是一个伪目标，不是具体的文件，不会与第一个目标文件相关联，所以我们在执行 make 的时候也不会执行下面的命令。在shell 中执行 "make clean" 命令，编译时的中间文件和生成的最终目标文件都会被清除，方便我们下次的使用。

### 1.2.5 Makefile通配符的使用

Makefile 是可以使用 shell 命令的。 shell 中使用的通配符有："*"，"?"，"[...]"。具体看一下这些通配符的表示含义和具体的使用方法。

| 通配符 | 使用说明                           |
| ------ | ---------------------------------- |
| *      | 匹配0个或者是任意个字符            |
| ？     | 匹配任意一个字符                   |
| []     | 我们可以指定匹配的字符放在 "[]" 中 |

通配符可以出现在模式的规则中，也可以出现在命令中，详细的使用情况如下：

实例 1：

```makefile
.PHONY:clean
clean:
	rm -rf *.o test
```

这是在 Makefile 中经常使用的规则语句。这个实例可以说明通配符可以使用在规则的命令当中，表示的是任意的以 .o 结尾的文件。
实例 2：

```makefile
test:*.c
	gcc -o $@ $^
```

这个实例可以说明我们的通配符不仅可以使用在规则的命令中，还可以使用在规则中。用来表示生所有的以 .c 结尾的文件。

但是如果我们的通配符使用在依赖的规则中的话一定要注意这个问题：**不能在引用变量中使用**，如下所示。

```makefile
OBJ=*.c
test:$(OBJ)
    gcc -o $@ $^
```

我们去执行这个命令的时候会出现错误，提示我们没有 "*.c" 文件，实例中我们相要表示的是当前目录下所有的 ".c" 文件，但是我们在使用的时候并没有展开，而是直接识别成了一个文件。文件名是 "*.c"。

如果我们就是相要通过引用变量的话，我们要使用一个函数 "wildcard"，这个函数在我们引用变量的时候，会帮我们展开。我们把上面的代码修改一下就可以使用了。

```makefile
OBJ=$(wildcard *.c)
test:$(OBJ)
	gcc -o $@ $^
```

这样我们再去使用的时候就可以了。调用函数的时候，会帮我们自动展开函数。

还有一个和通配符 "*" 相类似的字符，这个字符是 "%"，也是匹配任意个字符，使用在我们的的规则当中。

```makefile
test:test.o test1.o
	gcc -o $@ $^
%.o:%.c
	gcc -o $@ $^
```

 "%.o" 把我们需要的所有的 ".o" 文件组合成为一个列表，从列表中挨个取出的每一个文件，"%" 表示取出来文件的文件名（不包含后缀），然后找到文件中和 "%"名称相同的 ".c" 文件，然后执行下面的命令，直到列表中的文件全部被取出来为止。

这个属于 Makefile 中静态模规则：规则存在多个目标，并且不同的目标可以根据目标文件的名字来自动构造出依赖文件。跟我们的多规则目标的意思相近，但是又不相同。

### 1.2.6 变量的定义和使用

**变量的定义**

 Makefile 文件中定义变量的基本语法如下：

```
变量的名称=值列表
```

Makefile 中的变量的使用其实非常的简单，因为它并没有像其它语言那样定义变量的时候需要使用数据类型。变量的名称可以由大小写字母、阿拉伯数字和下划线构成。等号左右的空白符没有明确的要求，因为在执行 make 的时候多余的空白符会被自动的删除。至于值列表，既可以是零项，又可以是一项或者是多项。如：

```makefile
VALUE_LIST = one two three
```

调用变量的时候可以用 "\$(VALUE_LIST)" 或者是 "${VALUE_LIST}" 来替换，这就是变量的引用。实例：

```makefile
OBJ=main.o test.o test1.o test2.o
test:$(OBJ)
      gcc -o test $(OBJ)
```

这就是引用变量后的 Makefile 的编写，比我们之前的编写方式要简单的多。当要添加或者是删除某个依赖文件的时候，我们只需要改变变量 "OBJ" 的值就可以了。

**变量的基本赋值**

知道了如何定义，下面我们来说一下 Makefile 的变量的四种基本赋值方式：

-  简单赋值 ( := ) 编程语言中常规理解的赋值方式，只对当前语句的变量有效。
-  递归赋值 ( = ) 赋值语句可能影响多个变量，所有目标变量相关的其他变量都受影响。
-  条件赋值 ( ?= ) 如果变量未定义，则使用符号中的值定义变量。如果该变量已经赋值，则该赋值语句无效。
-  追加赋值 ( += ) 原变量用空格隔开的方式追加一个新值。

#### 简单赋值

```makefile
x:=foo
y:=$(x)b
x:=new
test：
      @echo "y=>$(y)"
      @echo "x=>$(x)"
```

# 二、CUDA一些概念

## 2.1 术语

Host：CPU和内存；

Device：GPU和显存；

CUDA中计算分为两部分，串行部分在Host上执行，即CPU，而并行部分在Device上执行，即GPU。

## 2.2 显卡硬件架构：SM、SP、Warp

具体到nvidia硬件架构上，有以下两个重要概念：

**SP（streaming processor）：**最基本的处理单元，也称为CUDA core。最后具体的指令和任务都是在SP上处理的。GPU进行并行计算，也就是很多个SP同时做处理。

**SM（streaming multiprocessor）：**多个SP加上其他的一些资源组成一个SM，也叫GPU大核，其他资源如：warp scheduler，register，shared memory等。SM可以看做GPU的心脏（对比CPU核心），register和shared memory是SM的稀缺资源。CUDA将这些资源分配给所有驻留在SM中的threads。因此，这些有限的资源就使每个SM中active warps有非常严格的限制，也就限制了并行能力。如下图是一个SM的基本组成，其中每个绿色小块代表一个SP。

![img](笔记.assets/v2-e51fe81b6f8808158b58e895cc4d3e09_1440w.jpg)

每个SM包含的SP数量依据GPU架构而不同，Fermi架构GF100是32个，GF10X是48个，Kepler架构都是192个，Maxwell都是128个。当一个kernel启动后，thread会被分配到很多SM中执行。大量的thread可能会被分配到不同的SM，但是同一个block中的thread必然在同一个SM中并行执行。

**Warp调度**

一个SP可以执行一个thread，但是实际上并不是所有的thread能够在同一时刻执行。**Nvidia把32个threads组成一个warp，warp是调度和运行的基本单元**。warp中所有threads并行的执行相同的指令。一个warp需要占用一个SM运行，多个warps需要轮流进入SM。由SM的硬件warp scheduler负责调度。目前每个warp包含32个threads（Nvidia保留修改数量的权利）。所以，**一个GPU上resident thread最多只有 SM\*warp个**。

同一个warp中的thread可以以任意顺序执行，active warps被SM资源限制。当一个warp空闲时，SM就可以调度驻留在该SM中另一个可用warp。在并发的warp之间切换是没什么消耗的，因为硬件资源早就被分配到所有thread和block，所以新调度的warp的状态已经存储在SM中了。

每个SM有一个32位register集合放在register file中，还有固定数量的shared memory，这些资源都被thread瓜分了，由于资源是有限的，所以，如果thread比较多，那么每个thread占用资源就叫少，thread较少，占用资源就较多，这需要根据自己的要求作出一个平衡。

## 2.3 软件架构：Kernel、Grid、Block

我们如何调用GPU上的线程实现我们的算法，则是通过Kernel实现的。在GPU上调用的函数成为CUDA核函数（Kernel function），核函数会被GPU上的多个线程执行。我们可以通过如下方式来定义一个kernel：

```text
func_name<<<grid, block>>>(param1, param2, param3....);
```

这里的grid与block是CUDA的线程组织方式，具体如下图所示：

![img](笔记.assets/v2-23d684f165319d30eb7fb4d0b669f055_1440w.jpg)

**Grid：**由一个单独的kernel启动的所有线程组成一个grid，grid中所有线程共享global memory。Grid由很多Block组成，可以是一维二维或三维。

**Block：**一个grid由许多block组成，block由许多线程组成，同样可以有一维、二维或者三维。block内部的多个线程可以同步（synchronize），可访问共享内存（share memory）。

CUDA中可以创建的网格数量跟GPU的计算能力有关，可创建的Grid、Block和Thread的最大数量如下所示：

![img](笔记.assets/v2-44a59a417750691c2f0ae78803daa750_1440w.jpg)

所有CUDA kernel的启动都是异步的，当CUDA kernel被调用时，控制权会立即返回给CPU。在分配Grid、Block大小时，我们可以遵循这几点原则：

- 保证block中thread数目是32的倍数。这是因为同一个block必须在一个SM内，而SM的Warp调度是32个线程一组进行的。
- 避免block太小：每个blcok最少128或256个thread。
- 根据kernel需要的资源调整block，多做实验来挖掘最佳配置。
- 保证block的数目远大于SM的数目。

# 三、CUDA程序编写

## 3.1 三个函数

| 函数名(重点是前面的修饰符)     | 执行位置 | 调用位置                  |
| ------------------------------ | -------- | ------------------------- |
| \__device__ float DeviceFunc() | device   | device                    |
| \__global__ void KernelFunc()  | device   | host & device（arch>3.0） |
| \__host__ float HostFunc()     | host     | host                      |

> \__global__定义一个kernel函数，入口函数，CPU上调用，GPU上执行，必须返回void。
>
> 调用核函数的方式：KernelFunc<<<block num, thread num>>>(参数1，……)，三个尖括号里面用于指定block数量和thread数量，这样我们的总线程数=block*thread

## 3.2 CUDA并行计算基础

### 3.2.1 CUDA线程层次

CUDA编程是一个多线程编程，数个线程(Thread)组成一个线程块(Block)，所有线程块组成一个线程网格(Grid)，如下图所示：

<img src="笔记.assets/v2-ca7030c0d6e2702b88271e8b0d9c1e38_1440w.jpg" alt="img" style="zoom: 67%;" />CUDA线程层级

图中的线程块，以及线程块中的线程，是按照2维的方式排布的。实际上，CUDA编程模型允许使用1维、2维、3维三种方式来排布。另外，即使线程块使用的是1维排布，线程块中的线程也不一定要按照1维排，而是可以任意排布。

> Thread：sequential execution unit，所有线程执行相同的核函数，并行执行。
>
> Thread Block：a group of threads，执行在一个Streaming Multiprocessor（SM），同一个Block中的线程可以协作。
>
> Thread Grid：a collection of thread blocks，一个Grid中的Blockj可以在多个SM中执行。

软件和硬件对应：

![image-20220110235516078](笔记.assets/image-20220110235516078.png)

### 3.2.2 CUDA线程索引 

block、thread可以是一维、二维或三维，因此x，y，z分别代表对应的维度。

threadIdx.[x y z]：执行当前核函数的线程在block中的索引值。

blockIdx.[x y z]：执行当前核函数的线程所在的block在grid中的索引值。

gridDim.[x y z]：表示一个grid中包含多少个block。

blockDIm.[x y z]：表示一个block中包含多少个线程。

> 最后两个值，就是<<<>>>中的值，gridDim*blockDim是总线程个数。
>
> 上面四个变量可以在核函数中直接使用。
>
> int index = threadIdx.x + blockIdx.x * blockDim.x 计算当前线程的总索引值。

### 3.2.3 CUDA线程分配

<img src="笔记.assets/image-20220110235712825.png" alt="image-20220110235712825" style="zoom:67%;" />

> block中有多个Warp，因此block中的线程个数一定是32的倍数，如果你指定31，也要花费一个Wrap。

```c
N=100000; // 总计算量
block_size = 128; // block中的线程数
grid_size = (N+block_size -1)/block_size; //grid中的block数
```

x,y,z三个维度如果我们没有设置y和z，那么默认为1。

 # 四、CUDA矩阵相乘

## 4.1 GPU的存储单元

每一个线程可以：

读/写每一个线程registers；

读/写每一个线程local memory；

读/写每一个block shared memory；

读/写每一个grid global memory;

只读每一个grid constant memory；

只读每一个grid texture memory。

<img src="笔记.assets/image-20220111132228688.png" alt="image-20220111132228688" style="zoom: 67%;" />

## 4.2 memory allocation/release

CPU：

- malloc()
- memset()
- free()

gpu:

| 函数                                                         | 形参                                                         | 含义                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------- |
| \__host__  _\_device__ cudaError_t cudaMalloc ( void** devPtr, size_t size ) | devPtr:指向分配内存size:需要分配的大小                       | 在设备上分配内存          |
| \__host__  _\_device__ cudaError_t cudaFree ( void* devPtr ) | devPtr:需要释放的设备内存指针                                | 释放设备内存              |
| \__host__ cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) | dst:目的地址；src:源地址；count:字节数；kind:方向，包括： cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault | 从src拷贝count个字节到dst |

- cudaMemset()

下面是使用案例：

```c
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#define BLOCK_SIZE 16

__global__ void gpu_matrix_mult(int *a,int *b,int *c,int m,int n,int k)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int sum=0;
    if(col<k&&row<m){
        for(int i=0;i<n;i++){
            sum+=a[row*n+i]*b[i*k+col];
        }
        c[row*k+col]=sum;
    }
}

void cpu_matrix_mult(int *c_a,int *c_b,int *c_c,int m,int n,int k)
{
    for(int i=0;i<m;i++){
        for(int j=0;j<k;j++){
            int tmp=0;
            for(int h=0;h<n;h++){
                tmp+=c_a[i*n+h]*c_b[h*k+j];
            }
            c_c[i*k+j]=tmp;
        }
    }
}

int main()
{
    int m=1000;
    int n=1000;
    int k=1000;
    
    int *c_a,*c_b,*c_c,*c_cc;
    cudaMallocHost((void **) &c_a, sizeof(int)*m*n);
    cudaMallocHost((void **) &c_b, sizeof(int)*n*k);
    cudaMallocHost((void **) &c_c, sizeof(int)*m*k);
    cudaMallocHost((void **) &c_cc, sizeof(int)*m*k);
    
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            c_a[i*n+j]=rand()%1024;
        }
    }
    
    for(int i=0;i<n;i++){
        for(int j=0;j<k;j++){
            c_b[i*k+j]=rand()%1024;
        }
    }
    
    int *g_a,*g_b,*g_c;
    cudaMalloc((void **)&g_a,sizeof(int)*m*n);
    cudaMalloc((void **)&g_b,sizeof(int)*n*k);
    cudaMalloc((void **)&g_c,sizeof(int)*m*k);
    
    cudaMemcpy(g_a,c_a,sizeof(int)*m*n,cudaMemcpyHostToDevice);
    cudaMemcpy(g_b,c_b,sizeof(int)*n*k,cudaMemcpyHostToDevice);
    
    unsigned int grid_rows=(m + BLOCK_SIZE-1)/BLOCK_SIZE;
    unsigned int grid_cols=(k + BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 dimGrid(grid_cols,grid_rows);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    gpu_matrix_mult<<<dimGrid,dimBlock>>>(g_a,g_b,g_c,m,n,k);
    cudaMemcpy(c_c,g_c,sizeof(int)*m*k,cudaMemcpyDeviceToHost);
    
    cpu_matrix_mult(c_a,c_b,c_cc,m,n,k);
    int ok = 1;
    for(int i=0;i<m;i++){
        for(int j=0;j<k;j++){
            if(fabs(c_cc[i*k+j]-c_c[i*k+j])>(1.0e-10)){
                ok = 0;
            }
        }
    }
    
    if(ok){
        printf("Pass!!!\n");
    }
    else{
        printf("Error!!!\n");
    }
    
    cudaFree(g_a);
    cudaFree(g_b);
    cudaFree(g_c);
    cudaFreeHost(c_a);
    cudaFreeHost(c_b);
    cudaFreeHost(c_c);
    cudaFreeHost(c_cc);
    
    
    return 0;
}
```

## 4.3 矩阵相乘

首先需要注意，无论是一维、二维或者三维矩阵，在存储时都是一维方式存储。

例如：矩阵$P_{mk}=M_{mn}*N_{nk}$。

1. 我们在CUDA中需要以P的大小来创建m*k个线程（二维的），每一个线程计算该位置的值；

2. 在调用CUDA函数时，需要指定gridsize和blocksize，而gridsize=（num*blocksize-1）/blocksize;(num表示该维度的大小)

3. 在每一个线程内部，我们可以获得该线程所在的行和列，进而求出该线程需要计算的数据的位置。

   ```c
   __global__ void gpu_matrix_mult(int *a,int *b,int *c,int m,int n,int k)
   {
       int row=blockIdx.y*blockDim.y+threadIdx.y;
       int col=blockIdx.x*blockDim.x+threadIdx.x;
       int sum=0;
       if(col<k&&row<m){
           for(int i=0;i<n;i++){
               sum+=a[row*n+i]*b[i*k+col];
           }
           c[row*k+col]=sum;
       }
   }
   ```

4. 下面是计算的过程图：

   <img src="笔记.assets/微信图片_20220111161815.jpg" alt="微信图片_20220111161815" style="zoom:50%;" />

​	从图中我们可知block的边缘线程是没有使用的，我们在计算时需要判断是否越界。**(col<k&&row<m)**

# 五、多种CUD存储单元

## 5.1 CUDA中的存储单元种类

![image-20220111194747957](笔记.assets/image-20220111194747957.png)

## 5.2 CUDA中的各种存储单元的使用方法

![gpu_memory](笔记.assets/gpu_memory.png)

**Registers**

寄存器是GPU中最快的memory，kernel中没有什么特殊声明的自动变量都是放在寄存器中的。当数组的索引是constant类型且在编译期能被确定的话，就是内置类型，数组也是放在寄存器中。

- 寄存器变量是每个线程私有的，一旦thread执行结束，寄存器变量就会失效。
- 不同结构，数量不同。

**Shared Memory**

用\__shared__修饰符修饰的变量存放在shared memory中。Shared Memory位于GPU芯片上，访问延迟仅次于寄存器。Shared Memory是可以被一个Block中的所有Thread来进行访问的，可以实现Block内的线程间的低开销通信。在SMX中，L1 Cache跟Shared Memory是共享一个64KB的高速存储单元的，他们之间的大小划分，不同的GPU结构不太一样；

- 要使用\__syncthread()同步；
- 比较小，要节省着使用，不然会限制活动warp的数量。

**Local Memory**

Local Memory本身在硬件中没有特定的存储单元，而是从Global Memory虚拟出来的地址空间。Local Memory是为寄存器无法满足存储需求的情况而设计的，主要是用于存放单线程的大型数组和变量。Local Memory是线程私有的，线程之间是不可见的。由于GPU硬件单位没有Local Memory的存储单元，所以，针对它的访问是比较慢的，跟Global Memory的访问速度是接近的。

在以下情况使用Local Memory：

- 无法确定其索引是否为常量的数组；
- 会消耗太多寄存器空间的大型结构或数组；
- 如果内核使用了多于寄存器的任何变量（这也称为寄存器溢出）；
- --ptxas-options=-v

**Constant Memory**

固定内存空间驻留在设备内存中，并缓存在固定缓存中（constant cache）

- constant的范围是全局的，针对所有kernel；
- kernel只能从constant Memory中读取数据，因此其初始化必须在host端使用下面的function调用：cudaError_t cudaMemcpyToSymbol(const void* symbol,const void* src,size_t count);
- 当一个warp中所有线程都从同一个Memory地址读取数据时，constant Memory表现会非常好，会触发广播机制。

**Global Memory**

Global Memory在某种意义上等同于GPU显存，kernel函数通过Global Memory来读写显存。Global Memory是kernel函数输入数据和写入结果的唯一来源。

**Texture Memory**

Texture Memory是GPU的重要特性之一，也是GPU编程优化的关键。Texture Memory实际上也是Global Memory的一部分，但是它有自己专用的只读cache。这个cache在浮点运算很有用，Texture Memory是针对2D空间局部性的优化策略，所以thread要获取2D数据就可以使用texture Memory来达到很高的性能。从读取性能的角度跟Constant Memory类似。

**Host Memory**

主机端存储器主要是内存可以分为两类：可分页内存（Pageable）和页面 （Page-Locked 或 Pinned）内存。

可分页内存通过操作系统 API(malloc/free) 分配存储器空间，该内存是可以换页的，即内存页可以被置换到磁盘中。可分页内存是不可用使用DMA（Direct Memory Acess)来进行访问的，普通的C程序使用的内存就是这个内存。

## 5.3 CUDA中的各种存储单元的适用条件

## 5.4 利用Shared Memory优化程序

### 5.4.1 Shared Memory详细介绍

Shared Memory是目前最快的可以让多个线程沟通的地方。那么，就有可能出现同时有很多线程访问Shared Memory上的数据。为了克服这个同时访问的瓶颈，Shared Memory被分成32个逻辑块（banks）。

![image-20220111203017177](笔记.assets/image-20220111203017177.png)

常用的两个场景：

① 用于两个线程之间的数据交换；

② 用于线程需要多次从Global Memory中读取的数据。

![image-20220111203654331](笔记.assets/image-20220111203654331.png)

### 5.4.2 Bank Conflict

为了获得高带宽，shared Memory被分成32（对应warp中的thread）个相等大小的内存块，他们可以被同时访问。如果warp访问shared Memory，对于每个bank只访问不多于一个内存地址，那么只需要一次内存传输就可以了，否则需要多次传输，因此会降低内存带宽的使用。

**当多个地址请求落在同一个bank中就会发生bank conflict**，从而导致请求多次执行。硬件会把这类请求分散到尽可能多的没有conflict的那些传输操作 里面，降低有效带宽的因素是被分散到的传输操作个数。

warp有三种典型的获取shared memory的模式：

· Parallel access：多个地址分散在多个bank。

· Serial access：多个地址落在同一个bank。

· Broadcast access：一个地址读操作落在一个bank。

Parallel access是最通常的模式，这个模式一般暗示，一些（也可能是全部）地址请求能够被一次传输解决。理想情况是，获取无conflict的shared memory的时，每个地址都在落在不同的bank中。

Serial access是最坏的模式，如果warp中的32个thread都访问了同一个bank中的不同位置，那就是32次单独的请求，而不是同时访问了。

Broadcast access也是只执行一次传输，然后传输结果会广播给所有发出请求的thread。这样的话就会导致带宽利用率低。

![image-20220111212925629](笔记.assets/image-20220111212925629.png)

> 上图是没有bank conflict，为同一个warp中的不同线程分配不同bank中的内存。
>
> 此时，为每个线程的分配内存地址为**\[threadIdx.y][threadIdx.x]**。之所以这样，是因为同一个warp中线程是连续的，后一个线程相比前一个线程，他们的row不变，col加1(row对应着y，col对应着x)。一个技巧\[x][y]，x变得慢，y变得快，谁在左边谁变得慢。

如果我们写作\[threadIdx.x][threadIdx.y]，就是下面这种情况：

<img src="笔记.assets/image-20220111214029530.png" alt="image-20220111214029530" style="zoom:50%;" />

如果出现这种情况（比如矩阵转置），如何解决这个问题呢？

我们改为sData\[BLOCKSIZE][BLOCKSIZE+1]即可，这样的效果就是在上图的右边再加一列：因为内存要按顺序分为32组，那么新加的第一个就是bank0的，以此类推，原来bank0的1号空间就分给了bank1。而warp中的线程不会错位，他们还是原来的位置，这样他们就分到了不同的bank里面了。

![image-20220111220024380](笔记.assets/image-20220111220024380.png)

> 其实我们可以发现，这样新加的一列内存空间，是不会被使用到的，也就是每一个bank都浪费了一个空间。

## 5.5 优化矩阵乘法

首先，矩阵乘法这个问题什么地方多次读取相同数据了？

很明显，矩阵M的每一行要和矩阵N的所有列进行计算，行上面的数据被多次连续使用。

下面是一个例子：

![image-20220111224140550](笔记.assets/image-20220111224140550.png)

> 右边是我们划分block后，每个block负责的区域。我们现在以求取紫色区域为例。
>
> 这一片紫色区域是由，M的下部分和N的右部分计算来的。
>
> | <img src="笔记.assets/image-20220111224601768.png" alt="image-20220111224601768" style="zoom:67%;" /> | <img src="笔记.assets/image-20220111224634325.png" alt="image-20220111224634325" style="zoom:67%;" /> |
> | ------------------------------------------------------------ | ------------------------------------------------------------ |

现在因为shared memory内存比较小，我们无法将MN中紫色区域全部放进来。所有我们现在把紫色区域再分成两部分，一次计算一块区域，然后将计算的两个结果相加。

<img src="笔记.assets/image-20220111224945451.png" alt="image-20220111224945451" style="zoom:50%;" />

我们的步骤就是：

每一个线程要计算的是P中的一个点。

```c
__global__ void gpu_matrix_mult_shared(int *d_a, int *d_b, int *d_result, int m, int n, int k) 
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) #sub表示两部分的哪一个部分
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        tile_a[threadIdx.y][threadIdx.x] = row<n && (sub * BLOCK_SIZE + threadIdx.x)<n? d_a[idx]:0;
        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        tile_b[threadIdx.y][threadIdx.x] = col<n && (sub * BLOCK_SIZE + threadIdx.y)<n? d_b[idx]:0;

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}
```

> 对于上面的代码，你可能存在下面的疑惑，为什么for里面syncthreads()前面的代码只赋予了一个值为什么，后面是循环k次。要明白该函数是计算P中紫色区域的一个元素，一个block中8x8（block中的所有线程，其他block在计算P的其他区域）个线程在执行该函数的，他们每一个线程都给tile_a和tile_b赋予了一个值，在syncthreads()里面的代码，每一个线程使用到了其他线程赋予的值。
>
> sub表示M中紫色区域被划分的份数。

# 六、代码注意事项

1. 在调用cuda核函数(\__global__声明的)时，一定要在后面一行加上`cudaDeviceSynchronize();`，否则会报`Bus error (core dumped)`错误。每调用一次核函数就要在其后一行加上这段代码。

2. 在核函数内部使用shared内存时，无论是读取还是写入数据，都需要在该代码段后面加上`__syncthreads();`，这个不要求后一行。另外读数据和写数据要分开进行，像下面这段代码：

   ```c
   for(int length=BLOCK_SIZE/2;length>=1;length/=2){
       int f=0;
       if(threadIdx.x<length){
           f=shared_block[threadIdx.x]+shared_block[threadIdx.x+length];
       }
       __syncthreads();
       if(threadIdx.x<length){
           shared_block[threadIdx.x]=f;
       }
       __syncthreads();
   }
   ```

3. 要注意越界问题，因为每一个BLOCK中线程的个数都是warp的整数倍（即使我们申请的不是warp整数倍），所有最后一组数据可能没有占满BLOCK。要注意加入判断条件。
4. C语言中的无穷大`INT_MAX`，无穷小`INT_MIN`。

# 七、例题

## 7.1 向量元素求和

题目：有一个一维数组A，求取该数组所有元素的和。

```c
#include<stdio.h>
#include<math.h>

#define N 100
#define BLOCK_SIZE 32

__managed__ int A[N];
__managed__ int result[1];

__global__ void sum_shared(int *input,int *output)
{
    __shared__ int shared_block[BLOCK_SIZE];
    int idx = BLOCK_SIZE*blockIdx.x+threadIdx.x;
    int temp=0;
    // 这段for循环的作用是因为有这种情况：我们的数据量大于我们申请的线程个数，
    // 我们需要每一个线程除了处理当前的数据外，还要处理下一轮数据，下下一轮，直到所有数据都处理完。
    // 步长：BLOCK_SIZE*gridDim.x，跨过已分配线程的数据。
    for(int i=idx;i<N;i+=BLOCK_SIZE*gridDim.x){
        temp+=input[i];
    }
    __syncthreads();
    if(idx<N){
        shared_block[threadIdx.x]=temp;
    }
    __syncthreads();
    // 每一个BLOCK内部线程分成两组，每一个线程计算“threadIdx.x和threadIdx.x+length”位置的和。
    // 这种方式每一轮都会减少1/2的活跃线程。
    for(int length=BLOCK_SIZE/2;length>=1;length/=2){
        int f=0;
        if(threadIdx.x<length){
            f=shared_block[threadIdx.x]+shared_block[threadIdx.x+length];
        }
        __syncthreads();
        if(threadIdx.x<length){
            shared_block[threadIdx.x]=f;
        }
        __syncthreads();
    }
    
    if(threadIdx.x==0){
        atomicAdd(result,shared_block[0]);
    }
    
}

int main()
{
    for(int i=0;i<N;i++){
        A[i]=i;
    }
    int grid_size=100;
    sum_shared<<<grid_size,BLOCK_SIZE>>>(A,result);
    cudaDeviceSynchronize();
    printf("%d\n",result[0]);
    return 0;
}
```

![image-20220114164014336](笔记.assets/image-20220114164014336.png)

## 7.2 矩阵转置

题目：有一个矩阵A\[M][N]，利用shared memory求出它的转置矩阵，放到B\[N][M]中。

```c
#include<stdio.h>
#include<math.h>

#define BLOCK_SIZE 32
#define M 1000
#define N 100

__managed__ int A[M][N];
__managed__ int B[N][M];

__global__ void matrix_trans_shared(int input[M][N],int output[N][M])
{
    __shared__ int rafa[BLOCK_SIZE][BLOCK_SIZE+1];
    int row = blockIdx.y*BLOCK_SIZE+threadIdx.y;
    int col = blockIdx.x*BLOCK_SIZE+threadIdx.x;
    if(row<M&&col<N){
        rafa[threadIdx.y][threadIdx.x]=input[row][col];
    }
    __syncthreads();
    int row1 = blockIdx.x*BLOCK_SIZE+threadIdx.y;
    int col1 = blockIdx.y*BLOCK_SIZE+threadIdx.x;
    if(row1<N&&col1<M){
        output[row1][col1]=rafa[threadIdx.x][threadIdx.y];
    }
    __syncthreads();
}

int main()
{
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            A[i][j]=rand();
        }
    }
    int grid_row= (M+BLOCK_SIZE-1)/BLOCK_SIZE;
    int grid_col= (N+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 dimGrid(grid_col,grid_row);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    matrix_trans_shared<<<dimGrid,dimBlock>>>(A,B);
    cudaDeviceSynchronize();
    //panduan
    int flag=0;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            if(A[i][j]!=B[j][i]){
                flag=1;
                break;
            }
        }
        if(flag==1){
            break;
        }
    }
    
    if(flag==1){
        printf("Fail\n");
    }else{
        printf("Pass\n");
    }
    
    return 0;
}
```

我们重点来看上面的写法和下面的写法：

```c
__global__ void matrix_trans_shared(int input[M][N],int output[N][M])
{
    int row = blockIdx.y*BLOCK_SIZE+threadIdx.y; // 线程的位置也是分配给它的数据的位置
    int col = blockIdx.x*BLOCK_SIZE+threadIdx.x;
    if(row<M&&col<N){
        output[col][row]=input[row][col];
    }
}
```

我们可能会有疑问为什么上面的写法这么复杂，直接传给output不就可以了，感觉完全不需要使用shared memory，使用shared memory之后还多走一步。

其实我们要明白，无论是global memory还是shared memory连续的线程访问连续地址访问速度大于跳跃访问，我们引入shared memory的目的就是为了，使我们从global memory中读取和写入数据时都是按照连续地址进行的。

接下来，说一下什么是连续地址，GPU地址是按行进行的，无论是一维还是二维数据，都是按照一维存储的。

我们来看从input中读数据时，显然，连续的线程访问连续的地址，block中的每一个线程访问与其位置对应的地址（位置[row，col]的线程要从位置[row，col]读取数据）。

然后我们再看向output中写数据时，位置[row，col]的线程要向位置[col，row]写入数据。我们很容易想到，现在两个连续的线程(比如[row,col+1]的线程要向位置[col+1，row]写入数据)要写入的地址是确实不连续的。

其实我们抛弃前面讲的，也可以这样想到：矩阵转置本来就是要`output[col][row]=input[row][col]`，如果读的时候是连续的，那么写的时候肯定不是连续的。如果我们这样`output[row][col]=input[col][row]`，虽然写连续，但读就不连续了。

我们现在引入shared memory也无法解决这个问题，但是从shared memory中不连续读取数据也很快，我们只需要保证在global memory 中是连续读取和连续写入即可。

`rafa[threadIdx.y][threadIdx.x]=input[row][col];` 连续读取，连续写入。

 `output[row1][col1]=rafa[threadIdx.x][threadIdx.y];` 不连续读取，连续写入。

下面解释一下`int row1 = blockIdx.x*BLOCK_SIZE+threadIdx.y; int col1 = blockIdx.y*BLOCK_SIZE+threadIdx.x;`

前面的乘法很好理解，block的位置肯定是要互换的，当前block中的数据要写的与我相对的block中，后面为什么block内部的相对位置和input一致，这是因为rafa是按列读取的，这样保证线程从global memory中无论读还是写都是连续的。

画个图：

<img src="笔记.assets/A39406BC1146C9DA00AEE6E3A7379CD5.png" alt="A39406BC1146C9DA00AEE6E3A7379CD5" style="zoom:50%;" />

