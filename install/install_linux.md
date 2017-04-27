# 在 Ubuntu 上安装 TensorFlow

这篇指南描述了如何在 Ubuntu 上安装 TensorFlow。这些实例也可能在其他 Linux 版本生效，但我们只在 Ubuntu 14.04 及更高的版本上测试过。

## 确定 TensorFlow 版本

如下之中选择一种来安装:

  * **只支持 CPU 的 TensorFlow**。如果你的系统不支持 NVIDIA® GPU, 你必须安装这个版本。这个版本的 TensorFlow 通常安装起来比较简单（一般 5 到 10分钟），所以即使你拥有 NVIDIA GPU，我们也推荐首先安装这个版本。
  * **支持 GPU 的 TensorFlow**. TensorFlow 在 GPU 上通常比在 CPU 上的执行的更快。所以如果你有符合如下要求的 NVIDIA® GPU 并且需要注重性能，可以随后安装这个版本。

<a name="NVIDIARequirements"></a>
### GPU support TensorFlow 的 NVIDIA 需求

需要事先安装如下 NVIDIA 软件。

  * CUDA® Toolkit 8.0. 详见 [NVIDIA's documentation](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A)。确保按照文档中描述的将 Cuda 相关路径加入到 `LD_LIBRARY_PATH` 环境变量中。
  * CUDA Toolkit 8.0 相关的 NVIDIA 驱动。
  * cuDNN v5.1。详见 [NVIDIA's documentation](https://developer.nvidia.com/cudnn)。确保创建了 `CUDA_HOME` 环境变量。
  * CUDA Compute Capability 3.0 或更高的 GPU 芯片。支持的 GPU 芯片详见 [NVIDIA documentation](https://developer.nvidia.com/cuda-gpus) 。
  * libcupti-dev 库, 该库提供了高级的性能支持，按如下命令安装：

    <pre>
    $ <b>sudo apt-get install libcupti-dev</b>
    </pre>

如果含有上述库但版本较老，先升级。如果不能升级，操作如下：

  * [从源码安装 TensorFlow](./install_linux.md)
  * 安装或升级至少如下版本:
    * CUDA toolkit 7.0 或更高
    * cuDNN v3 或更高
    * CUDA Compute Capability 3.0 或更高的 GPU 芯片。


## 确定如何安装 TensorFlow

有如下选择：

  * [virtualenv](#InstallingVirtualenv)
  * ["native" pip](#InstallingNativePip)
  * [Docker](#InstallingDocker)
  * [Anaconda](#InstallingAnaconda)

**推荐 virtualenv 安装**
*（略过四种方法的说明，自行查找）*


<a name="InstallingVirtualenv"></a>
## virtualenv 安装

步骤如下：

  1. 安装 pip 及 virtualenv：

     <pre>$ <b>sudo apt-get install python-pip python-dev python-virtualenv</b> </pre>

  2. 建立 virtualenv 环境：

     <pre>$ <b>virtualenv --system-site-packages</b> <i>targetDirectory</i> </pre>

     <code><em>targetDirectory</em></code> 指明了 virtualenv 的位置。

  3. 激活 virtualenv 环境：

     <pre>$ <b>source ~/tensorflow/bin/activate</b> # bash, sh, ksh, or zsh
     $ <b>source ~/tensorflow/bin/activate.csh</b>  # csh or tcsh </pre>

     如上操作会将提示符更改为如下：

     <pre> (tensorflow)$ </pre>

  4. 如下命令中选取一个安装 TensorFlow：

     <pre>(tensorflow)$ <b>pip install --upgrade tensorflow</b>      # for Python 2.7
     (tensorflow)$ <b>pip3 install --upgrade tensorflow</b>     # for Python 3.n
     (tensorflow)$ <b>pip install --upgrade tensorflow-gpu</b>  # for Python 2.7 and GPU
     (tensorflow)$ <b>pip3 install --upgrade tensorflow-gpu</b> # for Python 3.n and GPU</pre>
     上述命令成功则跳过步骤5，否则执行步骤5。

  5. (可选) 如果步骤4失败 (通常因为 pip 版本小于 8.1)：

     <pre>(tensorflow)$ <b>pip install --upgrade</b> <i>TF_PYTHON_URL</i>   # Python 2.7
     (tensorflow)$ <b>pip3 install --upgrade</b> <i>TF_PYTHON_URL</i>  # Python 3.N </pre>

     <code><em>TF_PYTHON_URL</em></code> 指定了 python tensorflow 的包的地址。<code><em>TF_PYTHON_URL</em></code> 依赖于操作系统、Python 版本、GPU 支持，从 [这里](#the_url_of_the_tensorflow_python_package) 找到合适的URL。如，安装 TensorFlow for Linux, Python 2.7、CPU-only，使用如下命令：

     <pre>(tensorflow)$ <b>pip3 install --upgrade \
     https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp34-cp34m-linux_x86_64.whl</b></pre>

如果遇到安装问题，详见 [常见安装问题](#CommonInstallationProblems).


### 下一步

安装完毕之后： [验证安装结果](#ValidateYourInstallation)。

注意，每次使用 TensorFlow 前需要先激活 virtualenv 环境，使用如下命令：

<pre>$ <b>source ~/tensorflow/bin/activate</b>      # bash, sh, ksh, or zsh 
$ <b>source ~/tensorflow/bin/activate.csh</b>  # csh or tcsh</pre>

激活之后可从当前 shell 执行命令，此时提示符变成如下：

<pre>(tensorflow)$ </pre>

使用完毕 TensorFlow 可用 `deactivate` 命令退出当前 virtualenv 环境：

<pre>(tensorflow)$ <b>deactivate</b> </pre>

此时提示符将会返回到默认提示符 （在 `PS1` 环境变量中定义）。


### 卸载 TensorFlow

卸载 TensorFlow，删除相关目录即可：

<pre>$ <b>rm -r</b> <i>targetDirectory</i> </pre>


<a name="InstallingNativePip"></a>
## 原生 pip 安装


**注意：** [setup.py 的 REQUIRED_PACKAGES 部分](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py) 列出了 pip 将会安装或者升级的 TensorFlow 包。


### 前提：Python 及 Pip

Python 已经在 Ubuntu 中自动安装了。花谢时间确定（`python -v`）你的操作系统中含有如下 Python 版本中的一个：

  * Python 2.7
  * Python 3.3+

Pip 或 pip3 *通常*已经在 Ubuntu 中安装。花些时间确定（`pip -V`或`pip3 -V`）已经安装。强烈建议使用 8.1 或更高的版本。如果 8.1 或更高的版本没有安装，使用如下命令安装或升级到最新 pip 版本：

<pre>
$ <b>sudo apt-get install python-pip python-dev</b>
</pre>


### 安装 TensorFlow

假定已经安装了如上必要软件，如下步骤安装 TensorFlow：

  1. 如下命令之一安装：

     <pre>$ <b>pip install tensorflow</b>      # Python 2.7; CPU support (no GPU support)
     $ <b>pip3 install tensorflow</b>     # Python 3.n; CPU support (no GPU support)
     $ <b>pip install tensorflow-gpu</b>  # Python 2.7;  GPU support
     $ <b>pip3 install tensorflow-gpu</b> # Python 3.n; GPU support </pre>

     如上命令执行完毕，可[验证安装结果](#ValidateYourInstallation).

  2. (可选) 如果步骤1失败，使用如下命令安装：

     <pre>$ <b>sudo pip  install --upgrade</b> <i>TF_PYTHON_URL</i>   # Python 2.7
     $ <b>sudo pip3 install --upgrade</b> <i>TF_PYTHON_URL</i>   # Python 3.N </pre>

     <code><em>TF_PYTHON_URL</em></code> 指定了 python tensorflow 的包的地址。<code><em>TF_PYTHON_URL</em></code> 依赖于操作系统、Python 版本、GPU 支持，从 [这里](#the_url_of_the_tensorflow_python_package) 找到合适的URL。
     例如，安装 TensorFlow for Linux, Python 3.4、CPU-only，使用如下命令：

     <pre>
     $ <b>sudo pip3 install --upgrade \
     https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp34-cp34m-linux_x86_64.whl</b>
     </pre>

     该步骤失败查询 [常见安装问题](#CommonInstallationProblems).


### 下一步

安装完毕之后： [验证安装结果](#ValidateYourInstallation)。


### 卸载 TensorFlow

如下命令卸载：

<pre>
$ <b>sudo pip uninstall tensorflow</b>  # for Python 2.7
$ <b>sudo pip3 uninstall tensorflow</b> # for Python 3.n
</pre>


<a name="InstallingDocker"></a>
## Docker 安装

如下步骤通过 Docker 安装 TensorFlow：

  1. 按描述在你的机器上安装 Docker
     [Docker documentation](http://docs.docker.com/engine/installation/).
  2. 可选，建立名为`docker`的用户组以便不通过 sudo 来登陆 container，
     [Docker documentation](https://docs.docker.com/engine/installation/linux/linux-postinstall/).
     （如果省略该步骤，每次启动 Docker 都需要使用 sudo。）
  3. 为安装支持 GPU 的 TensorFlow 版本，需要首先
     安装 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
  4. 启动含有
     [TensorFlow binary images](https://hub.docker.com/r/tensorflow/tensorflow/tags/).
     之一的 Docker 容器。

本章节的其余部分描述了如何启动一个 Docker 容器。


### CPU-only

使用如下命令启动一个 CPU-only Docker 容器：

<pre>
$ docker run -it <i>-p hostPort:containerPort TensorFlowCPUImage</i>
</pre>

其中：

  * <tt><i>-p hostPort:containerPort</i></tt> 可选。
    如果计划从 shell 执行 TensorFlow 程序，忽略该选项。
    如果计划作为`Jupyter notebooks`执行 TensorFlow 程序，设定
    <tt><i>hostPort</i></tt> 及 <tt><i>containerPort</i></tt>
    均为 <tt>8888</tt>。如果计划在容器中启动 TensorBoard，
    添加第二个 `-p` 参数, 设定 <i>hostPort</i> 及 <i>containerPort</i>
    均为 6006.
  * <tt><i>TensorFlowCPUImage</i></tt> 是必须的. 它指定了使用的容器，如下选项中选取一个：
    * <tt>gcr.io/tensorflow/tensorflow</tt>, TensorFlow CPU 镜像。
    * <tt>gcr.io/tensorflow/tensorflow:latest-devel</tt>, 最新的
      TensorFlow CPU 镜像外加源代码。
    * <tt>gcr.io/tensorflow/tensorflow:<i>version</i></tt>，指定版本（如1.0.1）。
    * <tt>gcr.io/tensorflow/tensorflow:<i>version</i>-devel</tt>，指定版本外加源代码。

    <tt>gcr.io</tt> 是 Google 的容器仓库。注意一些镜像同样可从
    [dockerhub](https://hub.docker.com/r/tensorflow/tensorflow/)获取。

例如，如下命令在一个容器中启动最新的 TensorFlow CPU 镜像，你可以在 shell 中执行 TensorFlow 程序：

<pre>
$ <b>docker run -it gcr.io/tensorflow/tensorflow bash</b>
</pre>

如下命令同样在一个容器中启动最新的 TensorFlow CPU 镜像。
但是在该容器中，你可以在`Jupyter notebook`中执行 TensorFlow 程序：

<pre>
$ <b>docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow</b>
</pre>

Docker 会在你第一次启动 TensorFlow 镜像时下载它。


### GPU support

安装支持 GPU 的 TensorFlow 之前，确保你的系统满足
[NVIDIA software requirements](#NVIDIARequirements)。To launch a Docker container
通过如下命令，启动一个支持 GPU 的 TensorFlow 的 Docker 容器

<pre>
$ <b>nvidia-docker run -it</b> <i>-p hostPort:containerPort TensorFlowGPUImage</i>
</pre>

其中：

  * <tt><i>-p hostPort:containerPort</i></tt> 可选。
    如果计划从 shell 执行 TensorFlow 程序，忽略该选项。
    如果计划作为`Jupyter notebooks`执行 TensorFlow 程序，设定
    <tt><i>hostPort</i></tt> 及 <tt><i>containerPort</i></tt>
    均为 <tt>8888</tt>。
  * <i>TensorFlowCPUImage</i> 是必须的. 它指定了使用的容器，如下选项中选取一个：
    * <tt>gcr.io/tensorflow/tensorflow:latest-gpu</tt>，最新 TensorFlow GPU 镜像。
    * <tt>gcr.io/tensorflow/tensorflow:latest-devel-gpu</tt>，最新 TensorFlow GPU 镜像外加源代码。
    * <tt>gcr.io/tensorflow/tensorflow:<i>version</i>-gpu</tt>，指定版本的 TensorFlow GPU 镜像。
    * <tt>gcr.io/tensorflow/tensorflow:<i>version</i>-devel-gpu</tt>，指定版本的 TensorFlow GPU 镜像外加源代码。

我们推荐安装一个`最新`版。如下命令可以在 Docker 容器中启动一个最新版本 TensorFlow GPU 镜像，
你可以在其 shell 中执行TensorFlow 程序：

<pre>
$ <b>nvidia-docker run -it gcr.io/tensorflow/tensorflow:latest-gpu bash</b>
</pre>

如下命令同样在一个容器中启动最新的 TensorFlow GPU 镜像。
但是在该容器中，你可以在`Jupyter notebook`中执行 TensorFlow 程序：

<pre>
$ <b>nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-gpu</b>
</pre>

如下命令启动一个老版本的 TensorFlow：

<pre>
$ <b>nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:0.12.1-gpu</b>
</pre>

Docker 会在你第一次启动 TensorFlow 镜像时下载它。详情
[TensorFlow docker readme](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker).


### 下一步

安装完毕之后： [验证安装结果](#ValidateYourInstallation)。


<a name="InstallingAnaconda"></a>
## Anaconda 安装

按照如下步骤在 Anaconda 环境中安装 TensorFlow：

  1. 按说明下载并安装 Anaconda：
     [Anaconda download site](https://www.continuum.io/downloads)

  2. 建立一个 conda 环境，命名为 <tt>tensorflow</tt>，以便运行某个 Python 版本：

     <pre>$ <b>conda create -n tensorflow</b></pre>

  3. 激活 anaconda 环境：

     <pre>$ <b>source activate tensorflow</b>
     (tensorflow)$  # 你的提示符应变化 </pre>

  4. 在你的 conda 环境中安装 TensorFlow：

     <pre>(tensorflow)$ <b>pip install --ignore-installed --upgrade</b> <i>TF_PYTHON_URL</i></pre>

     其中 <code><em>TF_PYTHON_URL</em></code> 是
     [TensorFlow Python 包地址](#the_url_of_the_tensorflow_python_package).
     比如：如下命令可以为 Python 3.4 安装 CPU-only 版本的 TensorFlow：

     <pre>
     (tensorflow)$ <b>pip install --ignore-installed --upgrade \
     https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp34-cp34m-linux_x86_64.whl</b></pre>


<a name="ValidateYourInstallation"></a>
## 验证安装结果

按如下操作验证 TensorFlow 安装结果：

  1. 确保准备环境完备
  2. 执行一个简短的 TensorFlow 程序


### 准备环境

If you installed on native pip, virtualenv, or Anaconda, then
do the following:
如果通过原生 pip、virtualenv、Anaconda 安装，做如下操作：

  1. 启动一个 terminal
  2. 如果通过 virtualenv 或 Anaconda 安装，激活容器
  3. 如果你安装了 TensorFlow 源码，定位到不含源码的任一目录中

如果通过 Docker 安装，启动一个可以通过 bash 操作的 Docker 容器：

<pre>
$ <b>docker run -it gcr.io/tensorflow/tensorflow bash</b>
</pre>


### 执行一个简短的 TensorFlow 程序

在 shell 中调用 Python：

<pre>$ <b>python</b></pre>

在 Python 交互式环境中输入如下命令：

```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

如果系统输出如下，则安装成功：

<pre>Hello, TensorFlow!</pre>

如果你新接触 TensorFlow，参考[初识 TensorFlow](../get_started)进行下一步学习。

如果系统输出错误信息而非欢迎信息，查看[常见安装问题](#common_installation_problems)。

## 常见安装问题

我们依靠 Stack Overflow 来编写 TensorFlow 安装问题及解决方案的文档。
如下表格包含了 Stack Overflow 上比较常见的安装问题的连接。
如果你遇到了不在列表中的新的错误信息或者其他安装问题，请在 Stack Overflow 上搜索。
如果搜索不到，请在 Stack Overflow 上提出一个新的问题，并打上 `tensorflow` 的标签。

<table>
<tr> <th>Stack Overflow Link</th> <th>Error Message</th> </tr>

<tr>
  <td><a href="https://stackoverflow.com/q/36159194">36159194</a></td>
  <td><pre>ImportError: libcudart.so.<i>Version</i>: cannot open shared object file:
  No such file or directory</pre></td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/41991101">41991101</a></td>
  <td><pre>ImportError: libcudnn.<i>Version</i>: cannot open shared object file:
  No such file or directory</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/36371137">36371137</a> and
  <a href="#Protobuf31">here</a></td>
  <td><pre>libprotobuf ERROR google/protobuf/src/google/protobuf/io/coded_stream.cc:207] A
  protocol message was rejected because it was too big (more than 67108864 bytes).
  To increase the limit (or to disable these warnings), see
  CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.</pre></td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/35252888">35252888</a></td>
  <td><pre>Error importing tensorflow. Unless you are using bazel, you should
  not try to import tensorflow from its source directory; please exit the
  tensorflow source tree, and relaunch your python interpreter from
  there.</pre></td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/33623453">33623453</a></td>
  <td><pre>IOError: [Errno 2] No such file or directory:
  '/tmp/pip-o6Tpui-build/setup.py'</tt></pre>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42006320">42006320</a></td>
  <td><pre>ImportError: Traceback (most recent call last):
  File ".../tensorflow/core/framework/graph_pb2.py", line 6, in <module>
  from google.protobuf import descriptor as _descriptor
  ImportError: cannot import name 'descriptor'</pre>
  </td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/questions/35190574">35190574</a> </td>
  <td><pre>SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
  failed</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42009190">42009190</a></td>
  <td><pre>
  Installing collected packages: setuptools, protobuf, wheel, numpy, tensorflow
  Found existing installation: setuptools 1.1.6
  Uninstalling setuptools-1.1.6:
  Exception:
  ...
  [Errno 1] Operation not permitted:
  '/tmp/pip-a1DXRT-uninstall/.../lib/python/_markerlib' </pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/questions/36933958">36933958</a></td>
  <td><pre>
  ...
  Installing collected packages: setuptools, protobuf, wheel, numpy, tensorflow
  Found existing installation: setuptools 1.1.6
  Uninstalling setuptools-1.1.6:
  Exception:
  ...
  [Errno 1] Operation not permitted:
  '/tmp/pip-a1DXRT-uninstall/System/Library/Frameworks/Python.framework/
   Versions/2.7/Extras/lib/python/_markerlib'</pre>
  </td>
</tr>

</table>


<a name="TF_PYTHON_URL"></a>
## TensorFlow Python 包地址（`TF_PYTHON_URL`）

一些安装方法需要 TensorFlow Python 包，它的地址依赖于几个方面：

  * 操作系统
  * Python 版本
  * CPU-only 还是 GPU support


### Python 2.7

CPU only:

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp27-none-linux_x86_64.whl
</pre>


GPU support:

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp27-none-linux_x86_64.whl
</pre>

注意 GPU 支持需要满足 NVIDIA 硬件需求以及在
[GPU support TensorFlow 的 NVIDIA 需求](#NVIDIARequirements)
中描述的软件。


### Python 3.4

CPU only:

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp34-cp34m-linux_x86_64.whl
</pre>


GPU support:

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp34-cp34m-linux_x86_64.whl
</pre>

注意 GPU 支持需要满足 NVIDIA 硬件需求以及在
[GPU support TensorFlow 的 NVIDIA 需求](#NVIDIARequirements)
中描述的软件。


### Python 3.5

CPU only:

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp35-cp35m-linux_x86_64.whl
</pre>


GPU support:

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl
</pre>


注意 GPU 支持需要满足 NVIDIA 硬件需求以及在
[GPU support TensorFlow 的 NVIDIA 需求](#NVIDIARequirements)
中描述的软件。

### Python 3.6

CPU only:

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp36-cp36m-linux_x86_64.whl
</pre>


GPU support:

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp36-cp36m-linux_x86_64.whl
</pre>


注意 GPU 支持需要满足 NVIDIA 硬件需求以及在
[GPU support TensorFlow 的 NVIDIA 需求](#NVIDIARequirements)
中描述的软件。

<a name="Protobuf31"></a>
## Protobuf pip 包 3.1

如果没有遇到和 protobuf pip 包相关的问题，你可以跳过这个部分。

**注意：** 如果你的 TensorFlow 运行缓慢，你可能遇到了一个 protobuf pip 包相关的问题。

TensorFlow pip 包依赖于 protobuf pip 包 3.1。
从 PyPI 下载的 protobuf 是 proto 序列化/反序列化的纯 Python 实现的库，
其速度比 C++ 实现**慢10-50倍**。Protobuf 支持二进制扩展，速度更快，基于 C++。
但该扩展无法在纯Python实现的 pip 包中获取。我们制作了包含该二进制扩展的protobuf pip 包。
如下命令可安装该自定义的 protobuf pip 包：

  * for Python 2.7:

  <pre>
  $ <b>pip install --upgrade \
  https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.1.0-cp27-none-linux_x86_64.whl</b></pre>

  * for Python 3.5:

  <pre>
  $ <b>pip3 install --upgrade \
  https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.1.0-cp35-none-linux_x86_64.whl</b></pre>

安装这个 protobuf 包会覆盖已经存在的 protobuf 包。
注意该二进制 pip 包已经支持大于 64MB 的问题，修复了如下问题：

<pre>[libprotobuf ERROR google/protobuf/src/google/protobuf/io/coded_stream.cc:207]
Protocol message 被拒绝，因为太大（大于 67108864 字节）。
为增大限制或禁用报警，
在 google/protobuf/io/coded_stream.h 中查看 CodedInputStream::SetTotalBytesLimit()</pre>
