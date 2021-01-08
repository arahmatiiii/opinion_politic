from fastai.text import *
from matplotlib import pyplot
import numpy
import time
import pandas

train_dataset = pandas.read_csv('../data/Processed/train_data_normed.csv')
train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)

test_dataset = pandas.read_csv('../data/Processed/train_data_normed.csv')
test_dataset = test_dataset.sample(frac=1).reset_index(drop=True)

# valid_dataset = pandas.read_csv('')
# valid_dataset = valid_dataset.sample(frac=1).reset_index(drop=True)
print('--------dataset loaded----------')

data_lm = (TextList.from_df(train_dataset, cols='text').split_by_rand_pct(0.1).label_for_lm().databunch(bs=48))
data_lm.save('data_lm.pkl')
data_lm = load_data('.', 'data_lm.pkl', bs=48)
print('--------data lm created and saved----------')

print('--------train started----------')
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
learn.fit_one_cycle(4, 1e-2, moms=(0.8,0.7))
learn.unfreeze()
learn.fit_one_cycle(4,1e-2,moms=(0.8,0.7))
learn.save_encoder('fine_tuned_enc')
print('--------train finished and saved----------')


data_clas= (TextList.from_df(train_dataset,cols='text',vocab=data_lm.vocab).split_none().label_from_df(cols='label').databunch(bs=48))
data_clas.save('data_clas.pkl')
data_clas = load_data('.','data_clas.pkl',bs=48)
print('--------data cls created and saved----------')

print('--------train cls started----------')
learn_cls = text_classifier_learner(data_clas,AWD_LSTM,drop_mult=0.5)
learn_cls.load_encoder('fine_tuned_enc')
learn_cls.fit_one_cycle(4,1e-3,moms=(0.8, 0.7))
print('--------train csl finished and saved----------')

true_label = []
pred_label = []
for i, item in enumerate(list(test_dataset.text)):
    res = learn_cls.predict(item)

    pred_label.append(int(res[1]))
    true_label.append(test_dataset.label[i])
    if i % 500 == 0:
        print(i)

from sklearn.metrics import classification_report
final_res = classification_report(y_true=true_label, y_pred=pred_label)
print(final_res)
