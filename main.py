import re 
import os
import gensim
import numpy as np
from gensim.models import Word2Vec, FastText
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

###### SECTION 1 DATA PREPROCESSING ######

lastreadchar = ''

with open("all_linkedin_skill_data",mode='r') as i, open('out.txt','w') as o:
    while True:
        x = i.read(1)

        if x == '': # end of file has been reached
            break 
        elif x==' ':
            pass
        elif x==']':
            pass
        elif x=='[':
            if lastreadchar == '[': 
                # at the beginning of the file, don't do anything
                pass
            elif lastreadchar == '\n': # a new line
                pass   
            elif lastreadchar == ',': # a new line
                pass
        elif x==',':
            if lastreadchar == ']': # at the beginning of the file
                
                o.write('\n')
            else:
                
                o.write(x)
        else:
                   
            o.write(x)

        lastreadchar = x

###### SECTION 2 TRAINING #######

#CBOW model
'''if not os.path.exists('model_out'):
    model1 = gensim.models.Word2Vec(l, min_count = 1, size = 100, window = 5)
    model1.save('model_out')'''
# a = input("Enter first skill:").lower()
# b = input("Enter second skill:").lower()
# model.similarity(a,b)

# Skip Gram Model
if not os.path.exists('model_out'):
    model2 = gensim.models.Word2Vec(lastreadchar, min_count = 1, size = 100, window = 5, sg = 1) 
    model2.save('model_out')
c = input("Enter first skill:").lower()
d = input("Enter second skill:").lower()
model_new = Word2Vec.load('model_out')
print(model_new.similarity(c,d))

#(OPTIONAL) SMARTER LEARNING TASK THAN A SIMPLE WORD2VEC

# if not os.path.exists('model_fast_out'):
#     model3 = FastText(lastreadchar, size =100, window=5, min_count=5, workers=4, sg=1)
#     model3.save('model_fast_out')
# model_fast_new = FastText.load('model_fast_out')
# e = input("Enter first skill:").lower()
# f = input("Enter second skill:").lower()
# print(model_new.similarity(e,f))

###### SECTION 3 VISUALIZATION USING TENSORBOARD ######

model2 = gensim.models.keyedvectors.KeyedVectors.load('model_out')
max_size = len(model2.wv.vocab)-1
w2v = np.zeros((max_size,model2.layer1_size))
if not os.path.exists('projections'):
    os.makedirs('projections')
with open("projections/metadata.tsv","w+") as file_metadata:
    for i, word in enumerate(model2.wv.index2word[:max_size]):
        w2v[i] = model2.wv[word]
        file_metadata.write(word + '\n')
sess = tf.InteractiveSession()
with tf.device("/cpu:0"):
    embedding = tf.Variable(w2v, trainable=False, name='embedding')
tf.global_variables_initializer().run()
saver = tf.train.Saver()
writer = tf.summary.FileWriter('projections',sess.graph)
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = 'embedding'
embed.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(writer, config)
saver.save(sess, 'projections/model.ckpt', global_step=max_size)

