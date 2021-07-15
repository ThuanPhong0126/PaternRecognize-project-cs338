import pandas as pd

import tensorflow as tf

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist
from sklearn.metrics import f1_score, accuracy_score

# File paths
TEST_CSV = './data/test.csv'

# Load training set
test_df = pd.read_csv(TEST_CSV)
gold_label = test_df['is_duplicate']
for q in ['question1', 'question2']:
    test_df[q + '_n'] = test_df[q]

embedding_dim = 300
max_seq_length = 20
test_df, embeddings = make_w2v_embeddings(test_df, embedding_dim=embedding_dim, empty_w2v=False)

X_test = split_and_zero_padding(test_df, max_seq_length)

assert X_test['left'].shape == X_test['right'].shape


model = tf.keras.models.load_model('./data/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})
model.summary()

prediction = model.predict([X_test['left'], X_test['right']])
predict = []
for i in prediction:
  if i[0]>=0.5: predict.append(1)
  else: predict.append(0)
print(accuracy_score(gold_label, predict))
print(f1_score(gold_label, predict))
print(f1_score(gold_label, predict, average='weighted'))
