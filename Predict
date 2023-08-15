import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score, accuracy_score, \
    precision_recall_curve
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.svm import SVC
import matplotlib.pyplot as plt

MPRA_feature = np.load('MPRA_feature.npy')

train_label = np.load('MPRA_data_label.npy')
Train_label = to_categorical(train_label)
MPRA_feature = np.tanh(MPRA_feature)
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score, accuracy_score
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
def eval_model(y_prob, y_test):
    y_test_prob = y_prob
    y_test_classes = np.argmax(y_test_prob, axis=-1)
    fpr, tpr, thresholds = roc_curve(y_test[:, 0], y_test_prob[:, 0])
    auc_test = auc(fpr, tpr)
    acc_test = accuracy_score(y_test_classes, np.argmax(y_test, axis=-1))
    f1_test = f1_score(y_test_classes, np.argmax(y_test, axis=-1), average='binary')
    recall_test = recall_score(y_test_classes, np.argmax(y_test, axis=-1), average='binary')
    precision_test = precision_score(y_test_classes, np.argmax(y_test, axis=-1), average='binary')
    R_test = pearsonr(y_test[:, 0], y_test_prob[:, 0])[0]
    acc_test = round(acc_test, 3)
    auc_test = round(auc_test, 3)
    f1_test = round(f1_test, 3)
    precision_test = round(precision_test, 3)
    recall_test = round(recall_test, 3)
    R_test = round(R_test, 3)
    return [acc_test, auc_test, f1_test, precision_test, recall_test, R_test]
def plot_auc(fpr,tpr,roc_auc):
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='SVM (AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

shape = (1, 5)
test_auc = np.zeros(shape)
test_F1 = np.zeros(shape)
test_acc = np.zeros(shape)
test_mcc = np.zeros(shape)
fpr = dict()
tpr = dict()
roc_auc = dict()
auprc = np.zeros(shape)
if __name__ == '__main__':
    for i in range(5):
        frac_train = 1
        np.random.seed(i)
        x_train, x_test, y_train, y_test = train_test_split(MPRA_feature, train_label, stratify=train_label,
                                                            test_size=0.2, random_state=i)
        if frac_train != 1:
            x_train, _, y_train, _ = train_test_split(x_train, y_train, stratify=y_train, test_size=1 - frac_train,
                                                        random_state=i)
        y_test = to_categorical(y_test)
        sv = SVC(C=2, kernel='rbf', probability=True)
        sv.fit(x_train,y_train)
        y_pro = sv.predict_proba(x_test)
        y_test_prob = y_pro
        y_test_classes = np.argmax(y_test_prob, axis=-1)
        fpr[i], tpr[i], thresholds = roc_curve(y_test[:, 0], y_test_prob[:, 0])
        auc_test = auc(fpr[i], tpr[i])

        roc_auc[i] = auc_test
        test = eval_model(y_pro, y_test)
        test_auc[0, i] = test[1]
        test_F1[0, i] = test[2]
        test_mcc[0, i] = test[5]
        precision, recall, _ = precision_recall_curve(y_test[:, 0], y_test_prob[:, 0])
        auprc[0,i] = auc(recall, precision)
        print('auprc = %.3f'%np.mean(auprc,axis=1))
    print("auc_test = %.3f test_SVM_F1 = %.3f test_mcc=%.3f" % (
        np.mean(test_auc, axis=1), np.mean(test_F1, axis=1), np.mean(test_mcc, axis=1)))
# mean_fpr = np.linspace(0, 1, 100)
# mean_tpr = np.zeros_like(mean_fpr)
# for i in range(5):
#     mean_tpr += np.interp(mean_fpr, fpr[i], tpr[i])
# mean_tpr /= 5
# mean_auc = auc(mean_fpr, mean_tpr)
# plot_auc(mean_fpr, mean_tpr, mean_auc)
mean_fpr_SVM = np.linspace(0, 1, 100)
mean_tpr_SVM = np.zeros_like(mean_fpr_SVM)
for i in range(5):
    mean_tpr_SVM += np.interp(mean_fpr_SVM, fpr[i], tpr[i])
mean_tpr_SVM /= 5
mean_auc_SVM = auc(mean_fpr_SVM, mean_tpr_SVM)
plt.figure()
plt.plot(mean_fpr_SVM, mean_tpr_SVM, color='darkorange', lw=2, label='SVM (AUC = %0.3f)' % mean_auc_SVM)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
