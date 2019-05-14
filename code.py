import numpy as np
from sklearn import svm
import pandas as pd
import csv
import time
from datetime import datetime
from sklearn import tree
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
time1=datetime.now()
print(time1)
nm = 495 #miRNA个数
nd = 383 #疾病个数
nc = 5430 #miRNA-疾病关联对数

ConnectDate = np.loadtxt('knowndiseasemirnainteraction.txt',dtype=int)-1
# 读入两种疾病语义相似性数据
DS1 =np.loadtxt('疾病语义类似性矩阵1.txt') 
DS2 = np.loadtxt('疾病语义类似性矩阵2.txt')

# 读入miRNA功能类似性数据
FS = np.loadtxt(r'miRNA功能类似性矩阵.txt')

#print('s')
def Getgauss_miRNA(adjacentmatrix,nm):
       KM = np.zeros((nm,nm))
       gamaa=1
       sumnormm=0
       for i in range(nm):
           normm = np.linalg.norm(adjacentmatrix[:,i])**2   
           sumnormm = sumnormm + normm  
       gamam = gamaa/(sumnormm/nm)
       for i in range(nm):
              for j in range(nm):
                      KM[i,j]= np.exp (-gamam*(np.linalg.norm(adjacentmatrix[:,i]-adjacentmatrix[:,j])**2))
       return KM

# 疾病高斯类似性矩阵
def Getgauss_disease(adjacentmatrix,nd):
       KD = np.zeros((nd,nd))
       gamaa=1
       sumnormd=0
       for i in range(nd):
              normd = np.linalg.norm(adjacentmatrix[i])**2
              sumnormd = sumnormd + normd
       gamad=gamaa/(sumnormd/nd)
       for i in range(nd):
           for j in range(nd):
               KD[i,j]= np.exp(-(gamad*(np.linalg.norm(adjacentmatrix[i]-adjacentmatrix[j])**2)))
       return KD
def sample_partition(positive_sample_index,unknown_sample_index,FeatureD,FeatureM):    
    positive_sample_FetureD = FeatureD[positive_sample_index[:,0]] #这样用数组整体操作
    positive_sample_FetureM = FeatureM[positive_sample_index[:,1]]
    positive_sample_Feture = np.hstack((positive_sample_FetureD,positive_sample_FetureM))#特征拼接
    unknown_sample_FeatureD = FeatureD[unknown_sample_index[:,0]]
    unknown_sample_FeatureM = FeatureM[unknown_sample_index[:,1]]
    unknown_sample_Feature = np.hstack((unknown_sample_FeatureD,unknown_sample_FeatureM))#特征拼接
    mean=np.mean(positive_sample_Feture,0)
    
    T1=time.time()
    distance=[]
    for i in range(unknown_sample_Feature.shape[0]):
        dis=np.dot(unknown_sample_Feature[i],mean)/ np.linalg.norm(unknown_sample_Feature[i])/np.linalg.norm(mean)
        distance.append(dis)
    T2 = time.time()
    distance = np.array(distance)    
    arg_distance = np.argsort(distance)#此处应按降序排序
    negitive_sample_index = arg_distance[0:nc-1]
    negitive_sample_feature = unknown_sample_Feature[negitive_sample_index]
    test_sample_index = unknown_sample_index # 此处我把所有未知样本，包含那些已经被选作负样本的，作为测试样本，即给所有的0打分
    test_sample_feature = unknown_sample_Feature
    return positive_sample_Feture,negitive_sample_feature,test_sample_feature

def train_and_predict(train_sample_feature,test_sample_feature,X_train,X_test,Y_value,svmID,accList):
    ncomp = 6
    pca =PCA(n_components=ncomp, svd_solver='randomized',whiten=True ).fit(train_sample_feature)
    Z1 = pca.transform(train_sample_feature) 
    pca =PCA(n_components=ncomp, svd_solver='randomized',whiten=True ).fit(test_sample_feature)
    Z2 = pca.transform(test_sample_feature) 
    clf=svm.SVC()
    train_sample_lable = [1 for j in range(5430)]+[0 for j in range(train_sample_feature.shape[0]-5430)]
    clf.fit(Z1,train_sample_lable)
    
    #print (T7-T6)
# =============================================================================
#     #train_result=np.zeros((train_sample_feature.shape[0],2))
#     train_result[:,0]=clf.predict(train_sample_feature)
#     train_result=clf.decision_function(train_sample_feature)
#     T8=time.time()
# =============================================================================
    #test_result=np.zeros((test_sample_feature.shape[0],2))

    #test_result=clf.predict(test_sample_feature)
    test_result =clf.decision_function(Z2)

    #print(T8-T7)
    X_train[:,svmID] = clf.decision_function(Z1)
    #X_train[:,1] = train_result[:,4]
    Y_value = train_sample_lable
    X_test[:,svmID] = test_result
    
    ncomp = 2
    pca =PCA(n_components=ncomp, svd_solver='randomized',whiten=True ).fit(train_sample_feature)
    Z = pca.transform(train_sample_feature) 
    colors = train_sample_lable
    #plt.xlabel(u'Actual_y0',fontsize=14)
    #plt.ylabel(u'Predicted_y0',fontsize=14)
    plt.grid()
    plt.scatter(Z[:,0],Z[:,1],c = colors)
    plt.show()
    
    accList.append(clf.score(Z1,train_sample_lable))
    
    return(X_train,X_test,Y_value,test_result,accList)

# 生成邻接矩阵
A=np.zeros((nd,nm),dtype=float)  #生成一个383*495的矩阵,初值为0
for i in range(nc):
    A[ConnectDate[i,1], ConnectDate[i,0]] = 1
globalrank_pos = []
localrank_pos = []
predict_0_local = []
accList = []
for i_nc in range(5430):
    T3 = time.time()
    A[ConnectDate[i_nc,1],ConnectDate[i_nc,0]] = 0  # 将A中的一个1变成0
    KM = Getgauss_miRNA(A,nm)  #用邻接矩阵重新计算miRNA的高斯相似矩阵
    KD = Getgauss_disease(A,nd)  #用邻接矩阵重新计算disease的高斯相似矩阵
    positive_sample_index = np.argwhere(A == 1)#正样本，用数组操作，减少For循环的使用，可节省时间
    unknown_sample_index =  np.argwhere(A == 0)# 负样本
    for i in range(unknown_sample_index.shape[0]):
        if unknown_sample_index[i,0]== ConnectDate[i_nc,1] and unknown_sample_index[i,1]== ConnectDate[i_nc,0]:
            i_1_0 = i # 表示由1变0的实例在unknown_sample_index中位置          
            break
    
    #SVM1
    T4 = time.time()
    positive_sample_DS1_FS,negative_sample_DS1_FS,test_sample_DS1_FS = sample_partition(positive_sample_index,unknown_sample_index,DS1,FS) 
    train_sample_DS1_FS = np.vstack((positive_sample_DS1_FS,negative_sample_DS1_FS)) 
    #test_result_DS1_FS = train_and_predict (train_sample_DS1_FS,test_sample_DS1_FS)
    X_train = np.zeros((len(train_sample_DS1_FS),6))
    X_test = np.zeros((len(test_sample_DS1_FS),6))
    Y_value = np.zeros((len(train_sample_DS1_FS),6))
    


    # TODO


    (X_train,X_test,Y_value,test_result_DS1_FS,accList) = train_and_predict (train_sample_DS1_FS,test_sample_DS1_FS,X_train,X_test,Y_value,0,accList)
    #(X_train,X_test,Y_value,test_result_DS1_FS) = train_and_predict (Z1,Z2,X_train,X_test,Y_value,0)
    T5 = time.time()
    
     #SVM2
    
# =============================================================================
    positive_sample_DS2_FS,negative_sample_DS2_FS,test_sample_DS2_FS = sample_partition(positive_sample_index,unknown_sample_index,DS2,FS)
    train_sample_DS2_FS = np.vstack((positive_sample_DS2_FS,negative_sample_DS2_FS)) 
    #train_result_DS2_FS,test_result_DS2_FS = train_and_predict (train_sample_DS2_FS,test_sample_DS2_FS)
    (X_train,X_test,Y_value,test_result_DS2_FS,accList) = train_and_predict (train_sample_DS2_FS,test_sample_DS2_FS,X_train,X_test,Y_value,1,accList)

      #SVM3
    positive_sample_KD_FS,negative_sample_KD_FS,test_sample_KD_FS = sample_partition(positive_sample_index,unknown_sample_index,KD,FS)
    train_sample_KD_FS = np.vstack((positive_sample_DS1_FS,negative_sample_KD_FS)) 
    #train_result_KD_FS,test_result_KD_FS = train_and_predict (train_sample_KD_FS,test_sample_KD_FS)
    (X_train,X_test,Y_value,test_result_KD_FS,accList) = train_and_predict (train_sample_KD_FS,test_sample_KD_FS,X_train,X_test,Y_value,2,accList)

      #SVM4
    positive_sample_DS1_KM,negative_sample_DS1_KM,test_sample_DS1_KM = sample_partition(positive_sample_index,unknown_sample_index,DS1,KM) 
    train_sample_DS1_KM = np.vstack((positive_sample_DS1_KM,negative_sample_DS1_KM)) 
    #train_result_DS1_KM,test_result_DS1_KM = train_and_predict (train_sample_DS1_KM,test_sample_DS1_KM)
    (X_train,X_test,Y_value,test_result_DS1_KM,accList) = train_and_predict (train_sample_DS1_KM,test_sample_DS1_KM,X_train,X_test,Y_value,3,accList)

      #SVM5
    positive_sample_DS2_KM,negative_sample_DS2_KM,test_sample_DS2_KM = sample_partition(positive_sample_index,unknown_sample_index,DS2,KM) 
    train_sample_DS2_KM = np.vstack((positive_sample_DS2_KM,negative_sample_DS2_KM)) 
    #train_result_DS2_KM,test_result_DS2_KM = train_and_predict (train_sample_DS2_KM,test_sample_DS2_KM)
    (X_train,X_test,Y_value,test_result_DS2_KM,accList) = train_and_predict (train_sample_DS2_KM,test_sample_DS2_KM,X_train,X_test,Y_value,4,accList)

      #SVM6
    positive_sample_KD_KM,negative_sample_KD_KM,test_sample_KD_KM = sample_partition(positive_sample_index,unknown_sample_index,KD,KM) 
    train_sample_KD_KM = np.vstack((positive_sample_KD_KM,negative_sample_KD_KM)) 
    #train_result_KD_KM,test_result_KD_KM = train_and_predict (train_sample_KD_KM,test_sample_KD_KM)
    (X_train,X_test,Y_value,test_result_KD_KM,accList) = train_and_predict (train_sample_KD_KM,test_sample_KD_KM,X_train,X_test,Y_value,5,accList)
    clf = tree.DecisionTreeRegressor(splitter='random',min_samples_split=3,min_samples_leaf = 2)
    Y_value = np.zeros((len(train_sample_DS1_FS)))
    for i in range(6):
        Y_value +=X_train[:,i]
    Y_value = Y_value/6
    clf = clf.fit(X_train, Y_value)
    predict_0 = clf.predict(X_test)
    predict_0_globalrank = pd.Series(predict_0).rank(ascending=False) #降序，为各个值分配平均排名
    time2=datetime.now()
    print(time2)
    #print(time2-time1)
    globalrank_pos.append(predict_0_globalrank[i_1_0])#global
    j=0
    for i in range(unknown_sample_index.shape[0]):
        if unknown_sample_index[i,0] == ConnectDate[i_nc,1]:
            predict_0_local.append(predict_0[i]) # 本地排名序列
            j=j+1
        if i==i_1_0:
            i_local=j
    predict_0_localrank=pd.Series(predict_0_local).rank(ascending=False) 
    localrank_pos.append(predict_0_localrank[i_local-1]) #找出 找出A中1变0对应实例的local排名


    A[ConnectDate[i_nc,0], ConnectDate[i_nc,1]] = 1  # 将A中变0元素变回1 
    
    globalrank_posTemp = np.array(globalrank_pos)
    
    localrank_posTemp = np.array(localrank_pos)
    
    #print(len(globalrank_pos))
    
    np.savetxt('result.dat',globalrank_posTemp)
    np.savetxt('localResult.dat',localrank_posTemp)
    
    plt.plot(accList)
    plt.show()

globalrank_posTemp = np.array(globalrank_pos)
localrank_posTemp = np.array(localrank_pos)

np.savetxt('result.dat',globalrank_posTemp)
np.savetxt('localResult.dat',localrank_posTemp)
