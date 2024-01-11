import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        model = [nn.ReflectionPad2d(3),nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),nn.InstanceNorm2d(ngf),nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,stride=2, padding=1),nn.InstanceNorm2d(ngf * mult * 2),nn.ReLU(True)]
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),kernel_size=3, stride=2,padding=1, output_padding=1),nn.InstanceNorm2d(int(ngf * mult / 2)),nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, input):
        return self.model(input)
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)
    def build_conv_block(self, dim):
        conv_block = [nn.ReflectionPad2d(1),nn.Conv2d(dim, dim, kernel_size=3, padding=0),nn.InstanceNorm2d(dim),nn.ReLU(True),nn.ReflectionPad2d(1),nn.Conv2d(dim, dim, kernel_size=3, padding=0),nn.InstanceNorm2d(dim)]
        return nn.Sequential(*conv_block)
    def forward(self, x):
        out = x + self.conv_block(x)
        return out
def im2tensor(input_path,size):
    transform_list = [transforms.Resize(size=size, interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    image = Image.open(input_path).convert('RGB')
    input_tensor = transform(image)
    return input_tensor
def tensor2im(input_image, imtype=np.uint8):
    image_numpy = input_image.data[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  
    return image_numpy.astype(imtype)

if __name__ == "__main__":
    mode = 'sketch'  # 选择模式: sketch / layout / rendering
    num = 9          # 测试哪张图片

    if mode == 'sketch':
        size = [512, 512]
    elif mode == 'layout':
        size = [256, 256]
    elif mode == 'rendering':
        size = [2048, 2048]

    import os
    your_path = r""# 图片所在的文件夹
    files = os.listdir(your_path)

    for file in files:
        input_path = os.path.join(your_path, file)
        print("处理中:", os.path.join(your_path, file))
        
        if os.path.isdir(input_path):
            print(f"跳过处理目录: {input_path}")
            continue

        input_pth = r""# 模型权重所在的文件夹

        model = ResnetGenerator(3, 3)
        model.load_state_dict(torch.load(input_pth))

        input_tensor = im2tensor(input_path, size)
        output_tensor = model(input_tensor)
        output_tensor = tensor2im(output_tensor.unsqueeze(0))

        output_file_name = f"out_{file[:-4]}.png"
        out_path = os.path.join(your_path, output_file_name)
        out = Image.fromarray(output_tensor)
        out.save(out_path)
        print("输出已保存:", out_path)

