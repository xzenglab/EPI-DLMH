import model
from keras.models import Model
import numpy as np
model = model.get_model_max() # create the original model
model.load_weights("./model/our_model_max/NHEKModel89.h5")
layer_name = 'concatenate_1'
intermediate_layer_model = Model(input=model.input,
                                 output=model.get_layer(layer_name).output)

#Data_dir = '/home/ycm/data/NHEK/'
name = 'NHEK'
#test = np.load(Data_dir+'%s_test.npz'%name)
#X_en_tes,X_pr_tes,y_tes=test['X_en_tes'],test['X_pr_tes'],test['y_tes']

Data_dir='/home/ycm/data/%s/'%name
train=np.load(Data_dir+'%s_train.npz'%name)
#test=np.load(Data_dir+'%s_test.npz'%name)
X_en_tra,X_pr_tra,y_tra=train['X_en_tra'],train['X_pr_tra'],train['y_tra']


print("****************Testing %s cell line specific model on %s cell line****************"%(name,name))
y_pred = intermediate_layer_model.predict([X_en_tra,X_pr_tra])
print(y_pred.shape)


import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd

plt.figure(figsize=(12,12), dpi=300,facecolor=(1, 1, 1))#平铺画布，设置dpi300,注意，发表文章dpi不能低于300

#adjust distrances of subplots
plt.subplots_adjust(wspace =0.3, hspace =0.35)#调节子图之间的宽度
model=TSNE(n_components=2,random_state=0)
ax2=plt.subplot(321)
# data1=pd.read_csv(r"C:\Users\Administrator\Desktop\Peptides_optim2\Anti-cancer_Peptides\fastadata\Tse\Bit20(NT=2).csv",header=None,index_col=None)
# labels1=pd.read_csv(r"C:\Users\Administrator\Desktop\Peptides_optim2\Anti-cancer_Peptides\fastadata\Tse\Bit20(NT=2)2.csv",header=None,index_col=None)
# data1=pd.read_csv(r"C:\Users\dell\Desktop\FT4\GAPCSV.csv",header=None,index_col=None)
# labels1=pd.read_csv(r"C:\Users\dell\Desktop\FT4\GAPlabel.csv",header=None,index_col=None)
tsne_data1=model.fit_transform(y_pred)
tsne_data1=np.vstack((tsne_data1.T,y_tra.T)).T
tsne_df=pd.DataFrame(data=tsne_data1,columns=("Dimension1","Dimension2","label"))
p1=tsne_df[(tsne_df.label==1) ]
p2=tsne_df[(tsne_df.label==0) ]
x1=p1.values[:,0]
y1=p1.values[:,1]
x2=p2.values[:,0]
y2=p2.values[:,1]
##plt.plot(x, y1)
plt.plot(x1, y1,'o', color='#EE7621',label = 'Positive',markersize='4')
#画曲线2
plt.plot(x2, y2,'o', color='#36648B',label = 'Negative',markersize='4')
# ax.scatter(x1,y1,c = 'r',marker = 'o')
# ax.scatter(x2,y2,c = 'b',marker = 'o')
plt.xlabel('Dimension1',fontsize=9)
plt.ylabel('Dimension2',fontsize=9)
plt.title('GAP (g=1)',fontsize=12)
plt.legend(loc="lower right")
plt.text(-52,26,'A',fontsize='13')
# plt.xlim((-38, 38))
# plt.ylim((-38, 38))

plt.show()