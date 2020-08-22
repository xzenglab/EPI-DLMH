import numpy as np


#embedding_matrix = np.load('/home/ycm/data/GM12878/GM12878_train_speid.npz')
#print(embedding_matrix)

train=np.load('/home/ycm/data/GM12878/GM12878_train.npz')
X_en_tra, X_pr_tra, y_tra = train['X_en_tra'], train['X_pr_tra'], train['y_tra']
print((X_en_tra[0, :9]))