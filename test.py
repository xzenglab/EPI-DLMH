
from model import get_model, get_model_C_sub, get_model_C_mul, get_model_max
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
for name in names:
    for epoch in [89]:
        model = get_model_max()
        model.load_weights("./model/our_model/%sModel%s.h5" % (name, epoch))
        Data_dir = '/home/ycm/data/%s/' % name
        test = np.load(Data_dir+'%s_test.npz' % name)
        X_en_tes, X_pr_tes, y_tes = test['X_en_tes'], test['X_pr_tes'], test['y_tes']

        print("****************Testing %s cell line specific model on %s cell line****************" % (name, name))
        y_pred = model.predict([X_en_tes, X_pr_tes])
        auc = roc_auc_score(y_tes, y_pred)
        aupr = average_precision_score(y_tes, y_pred)
        f1 = f1_score(y_tes, np.round(y_pred.reshape(-1)))
        print("AUC : ", auc)
        print("AUPR : ", aupr)
        print("f1_score", f1)
