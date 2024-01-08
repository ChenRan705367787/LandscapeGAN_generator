
from PIL import Image
# 导入os库
import os
folder_path = "sd render 20" #文件夹路径
files = os.listdir(folder_path) #一个包含所有文件名的列表
files_sizes = [(f, os.path.getsize(os.path.join(folder_path, f))) for f in files] #这是一个包含所有文件名和大小的元组列表
files_sizes.sort(key=lambda x: x[1]) #按照第二个元素（即文件大小）对列表进行升序排序
for i, (f, s) in enumerate(files_sizes): #遍历列表中的每个元组，并且用i来记录索引
  old_name = os.path.join(folder_path, f) #原来的文件路径和名字
  ext = os.path.splitext(f)[1] #文件的扩展名，例如.jpg
  new_name = os.path.join(folder_path, f"{i+1:03d}{ext}") # 这是新的文件路径和名字，用三位数字和原来的扩展名
  os.rename(old_name, new_name) #重命名文件
# 导入random库
import random
# 定义间隔大小（单位为像素）
gap = 10
# 定义拼接后的图片大小（单位为像素）
width = 15 * 180 + 16 * gap
height = 15 * 180 + 16 * gap
# 创建一个空白的图片对象
canvas = Image.new('RGB', (width, height), (255, 255, 255))
# 定义图片路径
path = folder_path
# 创建一个列表，用来存储所有的照片编号（从1到194）
numbers = list(range(1, 195))
# 使用random.shuffle函数，来打乱这个列表的顺序
random.shuffle(numbers)
# 遍历15行15列的图片
for i in range(15):
  for j in range(15):
    # 打开图片文件（使用os.path.join来拼接路径）
    # 使用列表中的元素，来代替原来的i*15+j+1，作为照片文件名的编号
    if i*15+j+1 > 194:
      continue
    img = Image.open(os.path.join(path, f'{numbers[i*15+j]:03d}.png'))
    # 调整图片大小为180x180像素
    img = img.resize((180, 180))
    # 计算图片在拼接后的位置
    x = j * (180 + gap) + gap
    y = i * (180 + gap) + gap
    # 将图片粘贴到画布上
    canvas.paste(img, (x, y))
# 保存拼接后的图片
canvas.save(os.path.join('K:\\LandscapeGAN_generator_only\\PSoutput', 'collage7.png'))