import os, sys
import glob
from PIL import Image
import shutil
from scipy.io import loadmat
#img_Lists = glob.glob(src_img_dir + '\*.png')

# citypersons图像的标注位置
src_anno_dir = loadmat(r'F:\cityperson\annotations\anno_val.mat')

# cityscapes图像的存储位置
src_img_dir = r"F:\cityperson\leftImg8bit_trainvaltest\leftImg8bit\val\\"

#保存为VOC 数据集的原图和xml标注路径
new_img= r"F:\cityperson\cityscapes\val\JPEGImages"
new_xml=r"F:\cityperson\cityscapes\val\Annotations"

# if not os.path.isdir(new_img):
#     os.makedirs(new_img)
#
# if not os.path.isdir(new_txt):
#     os.makedirs(new_txt)

a=src_anno_dir['anno_val_aligned'][0]

    #处理标注文件
f = open("F:\cityperson\cityscapes\gtbox_validation_pedestrain.txt","w")


for i in range(len(a)):
    img_name=a[i][0][0][1][0]   #frankfurt_000000_000294_leftImg8bit.png
    dir_name=img_name.split('_')[0]
    img=src_img_dir+dir_name+"\\"+img_name

    shutil.copy(img, new_img+"\\"+img_name)
    img=Image.open(img)
    width, height = img.size

    position=a[i][0][0][2]
    # print(position)
    #sys.exit()

    # txt_name=img_name.split('.')[0]
    # txt_file = open((new_txt + '\\' + txt_name + '.txt'), 'w')

    # xml_file.write('<annotation>\n')
    # xml_file.write('    <folder>citysperson</folder>\n')
    # xml_file.write('    <filename>' + str(img_name)+ '</filename>\n')
    # xml_file.write('    <size>\n')
    # xml_file.write('        <width>' + str(width) + '</width>\n')
    # xml_file.write('        <height>' + str(height) + '</height>\n')
    # xml_file.write('        <depth>3</depth>\n')
    # xml_file.write('    </size>\n')
    # print('{}/{}'.format(i,length))
    for j in range(len(position)):
        category_location=position[j]  #[    1   947   406    17    40 24000   950   407    14    39]
        category=category_location[0]  # class_label =0: ignore regions 1: pedestrians 2: riders 3: sitting persons 4: other persons 5: group of people

        if category == 0:
            continue
#             if
            #if category == 1 or category ==2 or category ==3 category ==4 or category ==5:
        elif category==1:
            x=category_location[1]   #class_label==1 or 2: x1，y1，w，h是与全身对齐的边界框；
            y=category_location[2]
            w=category_location[3]
            h=category_location[4]
            f.write('{} {} {} {} {}\n'.format(img_name,x,y,w,h))
    # txt_file.close()
f.close()

