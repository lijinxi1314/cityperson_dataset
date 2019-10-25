import numpy as np


from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.ensemble import GradientBoostingClassifier
from os import rename, listdir
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib

from sklearn import tree
import pydotplus

import function as f
import os
import cv2
import time

pos_data_path='F:/MIT/MIT/channel/positive/pos_one_person/'
neg_data_path='F:/MIT/MIT/channel/negative/neg_original/'
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'



def calculateIntegralFrom(image):
    # image=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    rows,cols=image.shape
    sum=np.zeros((rows,cols),np.float32)
    imageIntegral=cv2.integral(image,sum,cv2.CV_64F)
    # for i in range(66):
    #     print(imageIntegral[1][i])
    return imageIntegral

def calMaskfromIntegral(integral, x , y , w , h ):
    x=int(x)
    y=int(y)
    w=int(w)
    h=int(h)
    # print(x,y,w,h)
    if x+w-1>64:
        endX=63
    else:
        endX=x+w-1

    if y+h-1>128:
        endY=127
    else:
        endY=y+h-1

    sum1=integral[endY,endX]
    # print(sum1)
    if x>0:
        sum2=integral[endY,x-1]
    else:
        sum2=0
    # print(sum2)
    if y>0:
        sum3=integral[y-1,endX]
    else:
        sum3=0
    # print(sum3)
    if x>0 and y>0:
        sum4=integral[y-1,x-1]
    else:
        sum4=0
    # print(sum4)
    sum=sum1-sum2-sum3+sum4
    # print(sum)
    return sum

# input이 이미지 데이터
def extractFeature(integralImg,s_x,s_y,n_type,n_width,n_height):

    if n_type==0:
        sum1 = calMaskfromIntegral(integralImg, s_x, s_y, n_width / 2, n_height)
        sum2 = calMaskfromIntegral(integralImg, s_x + n_width / 2, s_y, n_width / 2, n_height)
        result = sum1-sum2
    elif n_type==1:
        sum1 = calMaskfromIntegral(integralImg, s_x, s_y, n_width / 3, n_height)
        sum2 = calMaskfromIntegral(integralImg, s_x + n_width / 3, s_y, n_width / 3, n_height)
        sum3 = calMaskfromIntegral(integralImg, s_x + (n_width * 2) / 3, s_y, n_width / 3, n_height)
        result=sum1-sum2+sum3
    elif n_type==2:
        sum1 = calMaskfromIntegral(integralImg, s_x, s_y, n_width, n_height / 2);
        sum2 = calMaskfromIntegral(integralImg, s_x, s_y + n_height / 2, n_width, n_height / 2)
        result = sum1 - sum2;
    elif n_type==3:
        sum1 = calMaskfromIntegral(integralImg, s_x, s_y, n_width, n_height / 3)
        sum2 = calMaskfromIntegral(integralImg, s_x, s_y + n_height / 3, n_width, n_height / 3)
        sum3 = calMaskfromIntegral(integralImg, s_x, s_y + (n_height * 2) / 3, n_width, n_height / 3)
        result = sum1 - sum2 + sum3
    elif n_type == 4:
        sum1 = calMaskfromIntegral(integralImg, s_x, s_y, n_width / 2, n_height / 2)
        sum2 = calMaskfromIntegral(integralImg, s_x + n_width / 2, s_y, n_width / 2, n_height / 2)
        sum3 = calMaskfromIntegral(integralImg, s_x, s_y + n_height / 2, n_width / 2, n_height / 2)
        sum4 = calMaskfromIntegral(integralImg, s_x + n_width / 2, s_y + n_height / 2, n_width / 2, n_height / 2)
        result = sum1 - sum2 - sum3 + sum4;

    return result




def read_channel(channel_name,image_path):
    ori_img=cv2.imread(image_path)
    gray_img=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    ############################################# LUV #############################################
    if(channel_name=="L"):
        cvrt_img=f.L(ori_img)
    if(channel_name=="U"):
        cvrt_img=f.U(ori_img)
    if(channel_name=="V"):
        cvrt_img=f.V(ori_img)
    ############################################ Garbor #############################################
    if(channel_name=="Gabor1"):
        cvrt_img=f.Garbor(ori_img,0)
    if(channel_name=="Gabor2"):
        cvrt_img=f.Garbor(ori_img,1)
    if(channel_name=="Gabor3"):
        cvrt_img=f.Garbor(ori_img,2)
    if(channel_name=="Gabor4"):
        cvrt_img=f.Garbor(ori_img,3)
    if(channel_name=="Gabor5"):
        cvrt_img=f.Garbor(ori_img,4)
    if(channel_name=="Gabor6"):
        cvrt_img=f.Garbor(ori_img,5)
    if(channel_name=="Gabor7"):
        cvrt_img=f.Garbor(ori_img,6)
    if(channel_name=="Gabor8"):
        cvrt_img=f.Garbor(ori_img,7)
    ############################################# Robinson #############################################
    if(channel_name=="Robinson_East"):
        cvrt_img=f.Robinson(ori_img,"robinson_east")
    if(channel_name=="Robinson_North"):
        cvrt_img=f.Robinson(ori_img,"robinson_north")
    if(channel_name=="Robinson_NE"):
        cvrt_img=f.Robinson(ori_img,"robinson_northeast")

    if(channel_name=="Robinson_NW"):
        cvrt_img=f.Robinson(ori_img,"robinson_northwest")
    if(channel_name=="Robinson_South"):
        cvrt_img=f.Robinson(ori_img,"robinson_south")
    if(channel_name=="Robinson_SE"):
        cvrt_img=f.Robinson(ori_img,"robinson_southeast")
    if(channel_name=="Robinson_SW"):
        cvrt_img=f.Robinson(ori_img,"robinson_southwest")
    if(channel_name=="Robinson_West"):
        cvrt_img=f.Robinson(ori_img,"robinson_west")
    ############################################# Grayscalse #############################################
    if(channel_name=="gray"):
        # cvrt_img=f.Gray(ori_img)
        cvrt_img=gray_img
    ############################################# Canny #############################################
    if(channel_name=="Canny"):
        cvrt_img=f.Canny(ori_img)
    ############################################# Threshold #############################################
    if(channel_name=="Threshold"):
        cvrt_img=f.Threshold(gray_img)
    ############################################# DoG #############################################
    if(channel_name=="DoG1"):
        cvrt_img=f.DoG(gray_img,1)
    if(channel_name=="DoG2"):
        cvrt_img=f.DoG(gray_img,2)
    if(channel_name=="DoG3"):
        cvrt_img=f.DoG(gray_img,3)
    ############################################# Gradient Magnitude #############################################
    if(channel_name=="gm"):
        cvrt_img=f.Gm(gray_img,1)
    if(channel_name=="Robinson_gm"):
        cvrt_img=f.Gm(gray_img,2)

    return cvrt_img

    # plt.imshow(cvrt_img)
    # cv2.imwrite('./result_channel/'+channel_name+'.png', cvrt_img)
    # plt.show()




def extract_POS_SHF_feature():
    result_pos=[]
    result_neg=[]
    channel_names= ["DoG1","DoG2", "DoG3","Gabor1","Gabor2","Gabor3","Gabor4","Gabor5","Gabor6","Gabor7","Gabor8",
		"Threshold","L","U","V","gray","Canny","Robinson_East", "Robinson_North","Robinson_NE","Robinson_NW","Robinson_South","Robinson_SE",
		"Robinson_SW","Robinson_West","gm","Robinson_gm" ]


# ----------------positive------------------------------
    for i in range(925):
        label=1
        img_path=pos_data_path+str(i)+'.bmp '

        img = cv2.imread(img_path)
        if img is None:
           continue
        else:
            txt_path='F:/MIT/weakclassifier/upper1_lower2_overlap0.3.txt'
            m=0
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    vec=line.strip('\n').split('\t')
                    type1=vec[0]
                    start_y1=vec[1]
                    start_x1=vec[2]
                    end_y1=vec[3]
                    end_x1=vec[4]
                    confVal1=vec[5]
                    h1=vec[6]
                    channel1=vec[7]

                    type2=vec[8]
                    start_y2=vec[9]
                    start_x2=vec[10]
                    end_y2=vec[11]
                    end_x2=vec[12]
                    confVal2=vec[13]
                    h2=vec[14]
                    channel2=vec[15]


                    type3=vec[16]
                    start_y3=vec[17]
                    start_x3=vec[18]
                    end_y3=vec[19]
                    end_x3=vec[20]
                    confVal3=vec[21]
                    h3=vec[22]
                    channel3=vec[23]

                    a=vec[24]
                    b=vec[25]
                    c=vec[26]
                    # print(channel3)
                    for k in range(27):
                        # print(channel_names[k])
                        channelImg=read_channel(channel_names[k],img_path)
                        IntegralImg=calculateIntegralFrom(channelImg)
                        haarVal1=extractFeature(IntegralImg,int(start_x1 )+ 1, int(start_y1) + 1, int(type1), int(end_x1) - int(start_x1), int(end_y1) - int(start_y1))
                        haarVal2 = extractFeature(IntegralImg, int(start_x2) + 1, int(start_y2) + 1, int(type2), int(end_x2) - int(start_x2), int(end_y2) - int(start_y2))
                        haarVal3 = extractFeature(IntegralImg, int(start_x3) + 1, int(start_y3) + 1, int(type3), int(end_x3) - int(start_x3), int(end_y3) - int(start_y3))
                        shfVal_new = int(haarVal1)*float(confVal1)*int(a) + int(haarVal2)*float(confVal2)*int(b) + int(haarVal3)*float(confVal3)*int(c);
                        result_pos.append(shfVal_new)
                        m=m+1
                        print(img_path,channel_names[k],m,label,shfVal_new)



    # result_pos=np.array([result_pos])
    # print(result_pos.shape)
    return result_pos

def exract_NEG_SHF_feature():
    channel_names= ["DoG1","DoG2", "DoG3","Gabor1","Gabor2","Gabor3","Gabor4","Gabor5","Gabor6","Gabor7","Gabor8",
    "Threshold","L","U","V","gray","Canny","Robinson_East", "Robinson_North","Robinson_NE","Robinson_NW","Robinson_South","Robinson_SE",
    "Robinson_SW","Robinson_West","gm","Robinson_gm" ]

    result_neg=[]
    for j in range(1500):
        label=-1
        img_path=neg_data_path+str(j)+'.bmp '
        img = cv2.imread(img_path)
        if img is None:
           continue
        else:
            txt_path='F:/MIT/weakclassifier/upper1_lower2_overlap0.3.txt'
            m=0
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    vec=line.strip('\n').split('\t')
                    type1=vec[0]
                    start_y1=vec[1]
                    start_x1=vec[2]
                    end_y1=vec[3]
                    end_x1=vec[4]
                    confVal1=vec[5]
                    h1=vec[6]
                    channel1=vec[7]

                    type2=vec[8]
                    start_y2=vec[9]
                    start_x2=vec[10]
                    end_y2=vec[11]
                    end_x2=vec[12]
                    confVal2=vec[13]
                    h2=vec[14]
                    channel2=vec[15]


                    type3=vec[16]
                    start_y3=vec[17]
                    start_x3=vec[18]
                    end_y3=vec[19]
                    end_x3=vec[20]
                    confVal3=vec[21]
                    h3=vec[22]
                    channel3=vec[23]

                    a=vec[24]
                    b=vec[25]
                    c=vec[26]

                    for k in range(27):
                        # print(channel_names[k])
                        channelImg=read_channel(channel_names[k],img_path)
                        IntegralImg=calculateIntegralFrom(channelImg)
                        haarVal1=extractFeature(IntegralImg,int(start_x1 )+ 1, int(start_y1) + 1, int(type1), int(end_x1) - int(start_x1), int(end_y1) - int(start_y1))
                        haarVal2 = extractFeature(IntegralImg, int(start_x2) + 1, int(start_y2) + 1, int(type2), int(end_x2) - int(start_x2), int(end_y2) - int(start_y2))
                        haarVal3 = extractFeature(IntegralImg, int(start_x3) + 1, int(start_y3) + 1, int(type3), int(end_x3) - int(start_x3), int(end_y3) - int(start_y3))
                        shfVal_new = int(haarVal1)*float(confVal1)*int(a) + int(haarVal2)*float(confVal2)*int(b) + int(haarVal3)*float(confVal3)*int(c);
                        result_neg.append(shfVal_new)
                        m=m+1
                        # print(img_path,channel_names[k],m,label,shfVal_new)

    result_neg=np.array([result_neg])
    # print(result_neg.shape)
    return result_neg

def load_data():

    feature_num=2295


    x_fail=exract_NEG_SHF_feature()
    x_fail=np.array([x_fail])
    print("neg")
    print(x_fail.shape)


    x_real=extract_POS_SHF_feature()
    x_real=np.array([x_real])
    print("pos")
    print(x_real.shape)



#shape 맞춰주기
    x_real=x_real.reshape(-1, feature_num) #pos data 수
    print("pos2")
    print("x_real.shape",x_real.shape)

    x_fail=x_fail.reshape(-1, feature_num) #neg data 수
    print("neg2")

    print("x_fail.shape",x_fail.shape)


    y_real=np.loadtxt('C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/three_patch_upper1_lower2/MIT/pos_label_origin.txt', dtype=np.int)
    #y_good=np.loadtxt('C:/Users/IVP Lab/source/repos/GAN_daseul/GAN_daseul/result/0528_finaldata/good_label.txt', dtype=np.int)
    y_fail=np.loadtxt('C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/three_patch_upper1_lower2/MIT/neg_label_origin.txt', dtype=np.int)
#################GBDT_classifier/pedestrian_feature
    # y_real=np.loadtxt('C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/upper1_lower2/pos_label_origin.txt', dtype=np.int)
    # #y_good=np.loadtxt('C:/Users/IVP Lab/source/repos/GAN_daseul/GAN_daseul/result/0528_finaldata/good_label.txt', dtype=np.int)
    # y_fail=np.loadtxt('C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/upper1_lower2/neg_label_origin.txt', dtype=np.int)


    print(y_real.shape)
    #print(y_good.shape)
    print(y_fail.shape)

    X=np.vstack([x_real,x_fail]) #X=np.vstack([x_real, x_good, x_fail])
    print("X.shape",X.shape)
    y=np.hstack([y_real, y_fail]) #y=np.hstack([y_real, y_good, y_fail])
    print("y.shape",y.shape)


    return X, y



def train():
    X,y=load_data()
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=0)#stratify=y
    #70%는 train셋, 30%는 테스트 셋
    #adaboost 분류기 생성하기

    gbm0 = GradientBoostingClassifier(random_state=10)
    gbm0.fit(X_train, y_train)

    y_predb=gbm0.predict(X_test)
    y_predprob = gbm0.predict_proba(X_test)[:,1]
    # print(y_predb)
    # print("Accuracy_b", metrics.accuracy_score(y_predb, y_test))
    sub_tree = gbm0.estimators_[69, 0]
    list_feature_name=[]
    for i in range (len(X_train[0])):
        list_feature_name.append(i)
    print("fit finsh")

    # tree.export_graphviz(gbrt_b, out_file=dot_data,feature_names=list_feature_name,class_names=np.unique(y)) # doctest: +SKIP
    dot_data = tree.export_graphviz(sub_tree,out_file=None, filled=True,rounded=True,
    special_characters=True, proportion=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf("best_depth_pdf/MIT_three_patch_upper1_lower2_nonono.pdf")
    # Image(graph.create_png())

    joblib.dump(gbm0, 'model/MIT_three_patch_upper1_lower2_nonono.pkl')

    print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_predb))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score( y_test, y_predprob))

# load_data()

# exract_NEG_SHF_feature()

# load_data()
train()
