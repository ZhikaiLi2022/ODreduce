# ODreduce

#### 观测者同测光程序的一款友好交互python包。

Odreduce 提供便捷的使用方法去分析测光像素文件，搜寻亮星并绘制光变曲线。

![image](https://github.com/ZhikaiLi2022/ODreduce/blob/main/odreduce.gif)

程序演示动画：
https://raw.githubusercontent.com/ZhikaiLi2022/ODreduce/main/odreduce.gif
#### 在运行ODreduce之前，需要做一些准备。

##### **1**.在ODreduce路径找到setup.py可执行文件，尝试终端运行以下命令：

```
python setup.py install
```

测试程序是否可行，你可以在终端窗口使用帮助命令，应该呈现以下输出：

```
$ odreduce --help

usage: odreduce [-h] [-version] {load,run} ...

ODreduce: Observation and Data Reduction

optional arguments:
  -h, --help           show this help message and exit
  -version, --version  Print version number and exit.

ODreduce modes:
  {load,run}
    load               Load in data for a given target
    run                Run the main ODreduce pipeline
```

##### **2**.在你认为方便的任意位置创建一个文件夹，用于存放odreduce相关数据和输出结果：

```
mkdir path_to_put_odreduce_stuff
cd path_to_put_odreduce_stuff
mkdir raw
```

##### **3**.将原始数据放在raw中，现在，你可以使用相似的命令在终端运行ODreduce，结果将自动保存到red文件夹：

```
odreduce run --star YZ_Boo -wave B -fv
```

逐个分析每段的作用

`odreduce`

​          setup文件定义的`odreduce`接口，你可以通过`pysyd.cli.main`程序看到所有的解析与命令。

`run`

​          使得`ODreduce`处在在run模式下，我们的测光数据将会按照流程被处理。它被保存到`args`解析器`NameSpace`作为mode，该模式将通过调用`pysyd.pipeline.run`来运行管道。 

`--star YZ_Boo`

​          ODreduce在run下将要运行的文件名字，与raw目录下文件名字相同。

`-wave B`

​            观测波段作为一个变量应该被给出。

`-fv`

​            我们把它们叫做booleans，这些booleans可以被组合在一起，这里的`-f`和`-v`分别是find和verbose的缩写，用于寻找绘制光变的明亮恒星和将流程详细输出。我们完全可以定制更多的选项。



​           

