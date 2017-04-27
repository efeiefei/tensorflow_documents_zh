# 在 Windows 上安装 TensorFlow

这篇指南描述了如何在 Windows 上安装 TensorFlow。

## 确定 TensorFlow 版本

如下之中选择一种来安装:

  * **只支持 CPU 的 TensorFlow**。如果你的系统不支持 NVIDIA® GPU, 你必须安装这个版本。这个版本的 TensorFlow 通常安装起来比较简单（一般 5 到 10分钟），所以即使你拥有 NVIDIA GPU，我们也推荐首先安装这个版本。
  * **支持 GPU 的 TensorFlow**. TensorFlow 在 GPU 上通常比在 CPU 上的执行的更快。所以如果你有符合如下要求的 NVIDIA® GPU 并且需要注重性能，可以随后安装这个版本。

### GPU support TensorFlow 的 NVIDIA 需求

需要事先安装如下软件：

  * CUDA® Toolkit 8.0。详见 [NVIDIA's documentation](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)。确保按照文档中描述的将 Cuda 相关路径加入到 `%PATH%` 环境变量中。
  * CUDA Toolkit 8.0 相关的 NVIDIA 驱动。
  * cuDNN v5.1。详见 [NVIDIA's documentation](https://developer.nvidia.com/cudnn)。注意：cuDNN 通常与其他 CUDA DLLs 安装的位置不同。确保将 cuDNN 库的安装目录加入到了`%PATH%`中。
  * CUDA Compute Capability 3.0 或更高的 GPU 芯片。支持的 GPU 芯片详见 [NVIDIA documentation](https://developer.nvidia.com/cuda-gpus) 。

如果上述软件版本较老，请将其升级到指定版本。


## 确定如何安装 TensorFlow

有如下选择：

  * "native" pip
  * Anaconda

原生 pip 直接在系统中安装 TensorFlow，而不使用虚拟环境。
因为原生 pip 安装没有使用独立的容器隔离开，所以可能干扰其他基于Python的安装。
不过，如果你理解 pip 和 Python 环境，原生 pip 安装通常只需要一个命令！
如果使用原生 pip 安装，用户可在任何目录中执行 TensorFlow 程序。

在 Anaconda 中，你可以通过 conda 创建一个虚拟环境。
然而，我们推荐使用 `pip install` 安装 TensorFlow，而非`conda install`。

**注意：**conda 包是社区支持而非官方支持。也就是说 TensorFlow 团队没有测试也没有管理过 conda 包。
使用这个包需要自行承担风险。


## 原生 pip 安装

如果如下版本的　Python 没有安装，先安装：

  * [Python 3.5.x from python.org](https://www.python.org/downloads/release/python-352/)

TensorFlow 在　Windows 上支持　Python 3.5.x。
注意 Python 3.5.x 使用 pip3，我们用 pip3 来安装 TensorFlow。

在 terminal 中输入如下命令安装只支持 CPU 的 TensorFlow：

<pre>C:\> <b>pip3 install --upgrade tensorflow</b></pre>

安装支持 GPU 的 TensorFlow，使用如下命令：

<pre>C:\> <b>pip3 install --upgrade tensorflow-gpu</b></pre>


## Anaconda 安装

**Anaconda 安装是社区支持，而非官方支持**

按照如下步骤在 Anaconda 环境中安装 TensorFlow：

  1. 按说明下载并安装 Anaconda：
     [Anaconda download site](https://www.continuum.io/downloads)

  2. 建立一个 conda 环境，命名为 <tt>tensorflow</tt>，以便运行某个 Python 版本：

     <pre>C:\> <b>conda create -n tensorflow</b> </pre>

  3. 激活 anaconda 环境：

     <pre>C:\> <b>activate tensorflow</b>
     (tensorflow)C:\>  # 你的提示符应该发生变化 </pre>

  4. 在你的 conda 环境中安装只支持 CPU 的 TensorFlow（写在一行）：

     <pre>(tensorflow)C:\> <b>pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.1.0-cp35-cp35m-win_amd64.whl</b> </pre>

     安装支持 GPU 的 TensorFlow（写在一行）：

     <pre>(tensorflow)C:\> <b>pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-win_amd64.whl</b> </pre>

## 验证安装结果

启动 terminal。

如果通过 Anaconda 安装，激活 Anaconda 环境。

启动 Python：

<pre>$ <b>python</b></pre>

在 Python 交互式环境中输入

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
  <td><a href="https://stackoverflow.com/q/41007279">41007279</a></td>
  <td>
  <pre>[...\stream_executor\dso_loader.cc] Couldn't open CUDA library nvcuda.dll</pre>
  </td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/41007279">41007279</a></td>
  <td>
  <pre>[...\stream_executor\cuda\cuda_dnn.cc] Unable to load cuDNN DSO</pre>
  </td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42006320">42006320</a></td>
  <td><pre>ImportError: Traceback (most recent call last):
File "...\tensorflow\core\framework\graph_pb2.py", line 6, in <module>
from google.protobuf import descriptor as _descriptor
ImportError: cannot import name 'descriptor'</pre>
  </td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/42011070">42011070</a></td>
  <td><pre>No module named "pywrap_tensorflow"</pre></td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/42217532">42217532</a></td>
  <td>
  <pre>OpKernel ('op: "BestSplits" device_type: "CPU"') for unknown op: BestSplits</pre>
  </td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/43134753">43134753</a></td>
  <td>
  <pre>The TensorFlow library wasn't compiled to use SSE instructions</pre>
  </td>
</tr>

</table>

