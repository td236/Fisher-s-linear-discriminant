from math import sqrt


def lda(learning_set_path, test_set_path):
    import numpy as np
    import pandas as pd
    from math import log
    df = pd.read_csv(learning_set_path)

    authentic = df[df['class'] == 0]  # class 0
    authentic = authentic.drop(columns=["class"])
    no_authentic = len(authentic)

    fake = df[df['class'] == 1]  # class 1
    fake = fake.drop(columns=["class"])
    no_fake = len(fake)

    def getMu(dataset: list):
        mu = []
        for i in range(len(fake.columns)):
            column = dataset.iloc[:, [i]]
            column_mean = column.sum() / len(column)
            mu.append(column_mean)
        return mu

    mu_authentic = getMu(authentic)
    mu_authentic = np.asarray(mu_authentic, dtype=np.float64)

    mu_fake = getMu(fake)
    mu_fake = np.asarray(mu_fake, dtype=np.float64)

    Sw = no_authentic*np.cov(authentic.to_numpy()
                             [1:].T) + no_fake*np.cov(fake.to_numpy()[1:].T)
    Sw /= (no_authentic + no_fake)
    inv_S = np.linalg.inv(Sw)

    res = inv_S.dot(mu_authentic-mu_fake).T  # vector of coefficients

    mahalanobis_distance = sqrt(res.dot(mu_authentic - mu_fake))

    mean_vector_dim = (mu_authentic * no_authentic + mu_fake *
                       no_fake) / (2*(no_authentic + no_fake))
    mean_vec = [mean_vector_dim[i][0] for i in range(len(mean_vector_dim))]

    ratio = log(no_fake/no_authentic)

    df = pd.read_csv(test_set_path)
    authentic_test = df[df['class'] == 0]
    authentic_test = authentic_test.drop(columns=["class"])

    roc_auth = []
    tp = 0
    fn = 0
    for i in range(len(authentic_test)):
        roc_auth.append(float(res.dot(np.asarray(authentic_test)[i]-mean_vec)))
        if(res.dot(np.asarray(authentic_test)[i]-mean_vec) > ratio):
            tp += 1
        else:
            fn += 1

    fake_test = df[df['class'] == 1]
    fake_test = fake_test.drop(columns=["class"])

    roc_fake = []
    tn = 0
    fp = 0
    for i in range(len(fake_test)):
        roc_fake.append(float(res.dot(np.asarray(authentic_test)[i]-mean_vec)))
        if(res.dot(np.asarray(fake_test)[i]-mean_vec) <= ratio):
            tn += 1
        else:
            fp += 1

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "dist": mahalanobis_distance}, roc_auth, roc_fake


k = 5
acc_sum = 0
ppv_sum = 0
npv_sum = 0
recall_sum = 0
distance_sum = 0
f1_sum = 0

roc_auth = []
roc_fake = []

for i in range(k):
    result, roc_auth, roc_fake = lda("data_learning_set_" + str(i) + ".csv",
                                     "data_test_set_" + str(i) + ".csv")
    print("Mahalanobis distance: %0.3f" % result["dist"])
    print("FP: ", result["tp"])
    print("FN: ", result["fn"])
    print("TN: ", result["tn"])
    print("FT: ", result["fp"])
    acc = (result["tp"]+result["tn"]) / \
        (result["tp"]+result["tn"]+result["fp"]+result["fn"])
    ppv = (result["tp"])/(result["tp"]+result["fp"])
    npv = (result["tn"])/(result["fn"]+result["tn"])
    recall = result["tp"]/(result["tp"]+result["fn"])
    f1 = 2 * ppv * recall / (ppv + recall)
    print("Accuracy: %0.3f" % acc)  # Dokladnosc
    print("PPV: %0.3f" % ppv)  # Precyzja
    print("NPV: %0.3f" % npv)
    print("Recall: %0.3f" % recall)
    print("F1: %0.3f" % f1)
    print()
    distance_sum += result["dist"]
    acc_sum += acc
    ppv_sum += ppv
    npv_sum += npv
    recall_sum += recall
    f1_sum += f1

print("Mahalanobis distance mean: %0.3f" % (distance_sum/k))
print("Accuracy mean: %0.3f" % (acc_sum/k))  # Dokladnosc
print("PPV mean: %0.3f" % (ppv_sum/k))  # Precyzja
print("NPV mean: %0.3f" % (npv_sum/k))
print("Recall mean: %0.3f" % (recall_sum/k))
print("F1 mean: %0.3f" % (f1_sum/k))