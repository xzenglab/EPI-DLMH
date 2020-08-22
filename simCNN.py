
# In[ ]:
import os
from model import get_simCNN
import numpy as np
import keras
from keras.callbacks import Callback
from datetime import datetime
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import train_test_split
from keras.utils.training_utils import multi_gpu_model

class roc_callback_speid(Callback):
    def __init__(self,name):
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights("./model/simCNN/%sModel%d.h5" % (self.name,epoch))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')


#names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK','all','all-NHEK']
#name=names[0]
#The data used here is the sequence processed by data_processing.py.
names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
for name in names:
    Data_dir='/home/ycm/data/%s/'%name
    train=np.load(Data_dir+'%s_train_speid.npz'%name)
#test=np.load(Data_dir+'%s_test.npz'%name)
    X_en_tra,X_pr_tra,y_tra=train['X_en_tra_speid'],train['X_pr_tra_speid'],train['y_tra_speid']
#X_en_tes,X_pr_tes,y_tes=test['X_en_tes'],test['X_pr_tes'],test['y_tes']

  #  X_en_tra, X_en_val,X_pr_tra,X_pr_val, y_tra, y_val=train_test_split(
     # X_en_tra,X_pr_tra,y_tra,test_size=0.05,stratify=y_tra,random_state=250)

    model=None
    model=get_simCNN()
    model.summary()
    print ('Traing %s cell line specific model ...'%name)


    back_1= roc_callback_speid(name=name)
    history=model.fit([X_en_tra, X_pr_tra], y_tra, epochs=90, batch_size=32,
                  callbacks=[back_1])

    t2 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print("开始时间:"+t1+"结束时间："+t2)




