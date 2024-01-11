# LandscapeGAN Generator

## 简介
本项目是一个基于样式生成对抗网络（style generative adversarial network, StyleGAN）的风景园林设计自动生成系统，专注于中小尺度公园设计方案自动生成。  

## LandscapeGAN 实现了3个方面功能：  
1.**基于线稿的设计方案自动生成（sketch 模式）：**  
   仅需输入简单的手绘草图，数毫秒内即可实时生成相应的设计方案。可直接应用于设计工作中的前期概念设计阶段，帮助设计师快速表达设计内容。  
2.**基于布局的设计方案自动生成（rendering 模式）：**  
   输入由语义颜色组成的布局方案即可实时生成设计方案。  
3.**基于场地条件的布局方案自动设计（layout 模式）：**  
   在前者的生成逻辑中，其输入的布局方案还需要设计师精心制作。那么布局方案能不能通过算法自动生成？如果说基于线稿、基于布局的生成方式只是图像渲染的问题，那么布局生成才是更接近自动设计的一步。  
   在 layout 模式中，毫无内容的场地条件也可以生成完整的布局方案。  

## 使用方法
1. **安装依赖：**  
   在运行代码之前，请确保已经安装了以下依赖：  
   pip install torch torchvision Pillow numpy  
3. **准备图片：**  
   将待处理的图片放入指定的文件夹，并设置 your_path 变量为图片所在文件夹的路径。  
4. **准备模型权重：**  
   将预训练的模型权重文件（.pth）放入指定的文件夹，并设置 input_pth 变量为模型权重文件的路径。  
5. **运行代码：**  
   在终端运行命令：python landscapeGAN.py  
   生成的输出图片将保存在与输入图片相同的文件夹下。  

## 注意事项
本研究采用 StyleGAN2 模型，训练采用 512×512 分辨率，在双卡NVIDIA 3090 GPU上运行，显存为64 GB。  
如果您的电脑配置不够运行代码，可以采取以下方法：  
1. **云服务：**  
   使用云服务提供商（如 AutoDL、Google Cloud、Microsoft Azure）上的虚拟机实例，租用更强大的 GPU 来运行代码。  
2. **调整模型的大小和复杂度：**  
   model = ResnetGenerator(self, input_nc, output_nc, ngf=64, n_blocks=9)  
   可以调整 ngf（生成器的通道数）和 n_blocks（残差块的数量），减小这些值会减小模型的复杂度。  
   如： model = ResnetGenerator(self, input_nc, output_nc, ngf=32, n_blocks=6)  
4. **降低输入图像分辨率：**  
   if mode == 'sketch':  
       size = [512, 512]  
   在 'if __name__ == "__main__":' 的代码块中，通过调整 size 可以降低输入图像的分辨率：  
   如：size = [256,256]  
   
## 文件结构
**landscapeGAN.py**  
    主要的生成脚本  
**.pth**  
    模型权重文件   
**test文件夹**  
    输入图片与生成的输出图片范例  
**README.md**  
    项目说明文件  

## 版本历史
1.0.0 (2024.1.11): 第一个正式版本  
