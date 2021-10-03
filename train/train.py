# -*- coding: utf-8 -*-
"""
Author: Peter Yu
Dataset: COCO 2014
Tip: Use deafult parameters (batch size: 256) needs 32G memory of GPU
Code Reference: https://www.tensorflow.org/tutorials
"""

import os, time, re, cv2, random, json, collections, pickle
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from model import *

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: pass
tqdm.pandas()

SEED = 2021
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Download caption annotation files
annotation_folder = '/annotations/'
if not os.path.exists(os.path.abspath('.') + annotation_folder):
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                           cache_subdir=os.path.abspath('.'),
                                           origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                           extract=True)
    annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
    os.remove(annotation_zip)

# Download image files
image_folder = '/train2014/'
if not os.path.exists(os.path.abspath('.') + image_folder):
    image_zip = tf.keras.utils.get_file('train2014.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin='http://images.cocodataset.org/zips/train2014.zip',
                                      extract=True)
    PATH = os.path.dirname(image_zip) + image_folder
    os.remove(image_zip)
else:
    PATH = os.path.abspath('.') + image_folder

with open('./annotations/captions_train2014.json', 'r') as f:
    annotations = json.load(f)

image_path_to_caption = collections.defaultdict(list)
for val in annotations['annotations']:
    caption = f"<start> {val['caption']} <end>"
    image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
    image_path_to_caption[image_path].append(caption)
image_paths = list(image_path_to_caption.keys())
random.shuffle(image_paths)

train_image_paths = image_paths
print(len(train_image_paths))
train_captions = []
img_name_vector = []

for image_path in train_image_paths:
    caption_list = image_path_to_caption[image_path]
    train_captions.extend(caption_list)
    img_name_vector.extend([image_path] * len(caption_list))

def calc_max_length(tensor):
    return max(len(t) for t in tensor)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~')
tokenizer.fit_on_texts(train_captions)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
train_seqs = tokenizer.texts_to_sequences(train_captions)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
max_length = calc_max_length(train_seqs)

img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(img_name_vector, cap_vector):
    img_to_cap_vector[img].append(cap)

img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)

slice_index = int(len(img_keys)*0.9)
img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

img_name_train = []
cap_train = []
for imgt in img_name_train_keys:
    capt_len = len(img_to_cap_vector[imgt])
    img_name_train.extend([imgt] * capt_len)
    cap_train.extend(img_to_cap_vector[imgt])

img_name_val = []
cap_val = []
for imgv in img_name_val_keys:
    capv_len = len(img_to_cap_vector[imgv])
    img_name_val.extend([imgv] * capv_len)
    cap_val.extend(img_to_cap_vector[imgv])

class DataGenerator():
    def __init__(self,x,y,batch_size,image_size,*args):
        self.x = x
        self.y = y
        self.batch = batch_size
        self.img_size = image_size

    def data_generator(self):
        for x, label in tqdm(zip(self.x,self.y )):
            image = self.load_image(x)
            yield image, label

    def load_image(self,image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img,(self.img_size, self.img_size))
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img

    def get(self):
        dataset = tf.data.Dataset.from_generator(self.data_generator,(tf.float32,tf.int64))
        dataset = dataset.map(lambda x,y: (x,y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,row_size,col_size,
                target_vocab_size,max_pos_encoding, img_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, row_size, col_size, img_size)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_pos_encoding)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def __call__(self, inp, tar, training, look_ahead_mask=None):
        if training:
            enc_output = self.encoder(inp, training)
        else:
            enc_output = inp
        dec_output = self.decoder(tar, enc_output, training, look_ahead_mask)
        final_output = self.final_layer(dec_output)
        return final_output
    
    @tf.function(input_signature=[tf.TensorSpec([None, 320, 320, 3], tf.float32)])
    def predict(self,img):
        encoder_out = transformer.encoder(img,False)
        output = tf.constant([tokenizer.word_index['<start>']], tf.int64)
        output = tf.expand_dims(output, 0)
        # print(img,encoder_out)
        for i in tf.range(1, max_length):
            tf.autograph.experimental.set_loop_options(shape_invariants=[(output, tf.TensorShape([None,None]))])
            dec_mask = create_masks_decoder(output)
            predictions = transformer(encoder_out, output, False, dec_mask)
            predictions = predictions[: ,-1:, :]
            predicted_id = tf.cast(tf.argmax(predictions,-1),tf.int64)
            # print(predictions)
            if int(predicted_id[0][0]) == 3:
                break
            output = tf.concat([output, predicted_id], axis=-1)
        return output

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=5000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        extra = step//5000*0.0001
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.6)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

d_model = 1024
dff = 2048
num_heads = 16
row_size = 10
col_size = 10
target_vocab_size = tokenizer.num_words
img_size = 320
image_channel = 3
num_layers = 6

learning_rate = CustomSchedule(d_model)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-7)

def create_masks_decoder(tar):
    look_ahead = 1 - tf.linalg.band_part(tf.ones((tf.shape(tar)[1], tf.shape(tar)[1])), -1, 0)
    dec_target = tf.cast(tf.math.equal(tar, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
    combined = tf.maximum(look_ahead, dec_target)
    return combined

@tf.function
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

@tf.function
def train_step(img_tensor, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    dec_mask = create_masks_decoder(tar_inp)
    with tf.GradientTape() as tape:
        predictions = transformer(img_tensor, tar_inp, True, dec_mask)
        loss = loss_function(tar_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)   
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
transformer = Transformer(num_layers,d_model,num_heads,dff,row_size,col_size,target_vocab_size, max_pos_encoding=max_length-1,img_size=img_size)

checkpoint_path = "./image-caption-save"
ckpt = tf.train.Checkpoint(transformer=transformer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)

if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)

batch_size = 256
step_to_show = 10
loss_plot = []
train_loss.reset_states()
train_accuracy.reset_states()

def randomSelect():
    n = random.randint(0,len(cap_val))
    dataset = DataGenerator([img_name_val[n]],[cap_val[n]],1,320).get()
    img, label = next(iter(dataset))
    encoder_out = transformer.encoder(img,False)
    output = tf.constant([tokenizer.word_index['<start>']], tf.int64)
    output = tf.expand_dims(output, 0)
    for i in tf.range(1,max_length):
        dec_mask = create_masks_decoder(output)
        predictions = transformer(encoder_out, output, False, dec_mask)
        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions,-1),tf.int64)
        if tokenizer.index_word[int(predicted_id[0][0])] == '<end>':
            break
        output = tf.concat([output, predicted_id], axis=-1)
    text = tokenizer.sequences_to_texts([output[0].numpy()[1:]])
    label = tokenizer.sequences_to_texts([label[0].numpy()[1:-1]])
    return text[0]+'\n'+label[0]


    
train_array = [0,0,0,0,0,0,0,0,0,0]
start_time = time.time()
print('Start training')
for e, batch in enumerate(train_array):
    dataset = DataGenerator(img_name_train,cap_train,batch_size,320,3).get()
    for img_tensor, target in dataset:
        train_step(img_tensor, target)
        if batch % step_to_show == 0:
            print(randomSelect())
            print(f'Epoch {e+1} Batch {batch} Loss {train_loss.result():.4f} Acc {train_accuracy.result():.4f} LR {optimizer._decayed_lr(tf.float32)} Time Used {time.time()-start_time:.2f} sec')
        if batch % (step_to_show*10) == 0 and batch != 0:
            ckpt_manager.save()
            print('model save')
        if batch % (step_to_show *50) == 0 and batch != 0:
            tf.saved_model.save(transformer, "savedmodel/", signatures={"serving_default": transformer.predict})
        batch+=1