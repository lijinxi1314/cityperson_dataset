# cityperson_dataset
citypersontoVOC.py 文件，可以将下载的anno_train.mat文件变成xml格式


cityperson_to_txt.py文件，可以将将下载的anno_train.mat文件中ground truth中的坐标变成txt格式
根据category==1 ，可只提取pedestrain 类别的ground truth


function.py 是一个所有channel的函数集合


get_feature_to_gbdt_main.py 是在数据集上提取对应的特征，经过多个channel特征通道后，提取特征，并使用该特征进行gbdt模型训练



get_feature_to_gbdt_test.py 是在数据集上提取对应的特征，经过多个channel特征通道后，提取特征，使用*main.py中得到的model,测试
