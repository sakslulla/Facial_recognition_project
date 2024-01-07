import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
import numpy as np
import matplotlib as plt
from keras import Input
# Import tensorflow dependencies -functional API
from tensorflow.keras.models import Model
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from tensorflow.python.layers.base import Layer
from tensorflow.keras.metrics import Precision, Recall

# Avoid OOM errors by setting GPU MEMORY consumption growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Make the directory
#os.makedirs(ANC_PATH)
#os.makedirs(POS_PATH)
#os.makedirs(NEG_PATH)


# http://vis-www.cs.umass.edu/lfw/
# Uncompress Tar GZ Labelled faces in the wild dataset

# move LFW Images to the following repository data/negative
#for directory in os.listdir('lfw'):
#   for file in os.listdir(os.path.join('lfw',directory)):
#      EX_PATH = os.path.join('lfw',directory,file)
#      NEW_PATH =os.path.join(NEG_PATH,file)
#      os.replace(EX_PATH,NEW_PATH)

import os.path
import cv2
# Import uuid library to generate unique names
import uuid
import matplotlib as plt

#cap = cv2.VideoCapture(0)
#while cap.isOpened():
#    ret, frame = cap.read()
    # cut down frame to 250x250 px
 #   frame = frame[120:120 + 250, 200:200 + 250]
    # collect anchors
 #   if cv2.waitKey(1) & 0XFF == ord('a'):
        # create the unique file path
 #       imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
  #      cv2.imwrite(imgname, frame)
    # collect positives
 #   if cv2.waitKey(1) & 0XFF == ord('p'):
 #       imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
 #       cv2.imwrite(imgname, frame)
    # show image back to screen
 #   cv2.imshow('Image Collection', frame)

    # Breaking gracefully
 #   if cv2.waitKey(1) & 0XFF == ord('q'):
 #       break

# Release the webcam
#cap.release()
#cv2.destroyAllWindows()


anchor = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(100)
positive = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(100)
negative = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(100)
dir_test = anchor.as_numpy_iterator()

#prepocessing -scale and resize
def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # load in the image
    img = tf.io.decode_jpeg(byte_img)
    # preprocessing step -resizing the image to be 100 x 100 x3
    img = tf.image.resize(img, (100,100))
    # Scaling image to be between 0 and 1
    img = img/255.0
    # Return image
    return img
    dataset.map(preprocess)

#(anchor,positive) => 1,1,1,1,1
# (anchor, negative) = > 0,0,0,0,0
#class_labels= tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))
#iterator_labs = class_labels.as_numpy_iterator()
#iterator_labs.next()

positives = tf.data.Dataset.zip((anchor, positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)
samples = data.as_numpy_iterator()
example = samples.next()


# Build Train and Test partition

def prepocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)
res = prepocess_twin(*example)




# Build dataloader pipeline
data = data.map(prepocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)


# Training partition
train_data = data.take(round(len(data)*7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

train_sample = train_data.as_numpy_iterator()
train_sample = train_sample.next()


# Testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# Build Embedding layer

inp = Input(shape=(100, 100, 3), name='input_image')
# first block
c1 = Conv2D(64, (10, 10), activation='relu')(inp)
m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

# second block
c2 = Conv2D(128, (7, 7), activation='relu')(m1)
m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

# Third block
c3 = Conv2D(128, (4, 4), activation='relu')(m2)
m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

# final embedding  layer
c4 = Conv2D(128, (4, 4), activation='relu')(m3)
f1 = Flatten()(c4)
d1 = Dense(4096, activation='sigmoid')(f1)

embedding = Model(inputs=[inp], outputs=[d1], name='embedding')
#embedding.summary()


# Build Distance Layer siamese network
class L1Dist(Layer):
    # Init method - inheritance
    def __int__(self,**kwargs):
        super().__init__()

    # Magic Happens here-similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

l1 = L1Dist()


def make_siamese_model():
    # Handle inputs
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))
    # validation image in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))
    # Combine siamese distance layer components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model=make_siamese_model()
#siamese_model.summary()


# Training
# setup Loss and Optimizer
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001
# Establish checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)
# Build Train step Function
test_batch = train_data.as_numpy_iterator()
batch_1 = test_batch.next()
X = batch_1[:2]
y = batch_1[2]
@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        # Get anchor and positive /negative images
        X = batch[:2]
        #get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training =True)
        # calculate loss
        loss = binary_cross_loss(y,yhat)
    #print(loss)
    #calculate   gradients
    grad = tape.gradient(loss,siamese_model.trainable_variables)
    # calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad,siamese_model.trainable_variables))
    return loss

def train(data, EPOCHS):
    # loop through epochs

    for epoch in range(1, EPOCHS+1):
        print('\n Epoch{}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        # loop through each batch
        for idx, batch in enumerate(data):
            # Run  train step here
            train_step(batch)
            progbar.update(idx+1)
        # save checkpoints
        if epoch % 10 ==0:
            checkpoint.save(file_prefix=checkpoint_prefix)

# Train the Model
EPOCHS = 50
#train(train_data, EPOCHS)


# Evaluate Model
# Import metrics
# Get  a batch test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

# Make Predictions
y_hat = siamese_model.predict([test_input, test_val])
print(y_hat)
# Post Processing the result
[1 if prediction>0.5 else 0 for prediction in y_hat]
print(y_true)
# Creating a metrics object
m = Precision()
m.update_state(y_true, y_hat)
# Return Recall Result
m.result().numpy()

r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = siamese_model.predict([test_input, test_val])
    r.update_state(y_true, yhat)
    p.update_state(y_true,yhat)

print(r.result().numpy(), p.result().numpy())
import matplotlib.pyplot as plt
import numpy as np

# 6.4 Viz Results
#set plot size
#plt.figure(figsize=(10, 8))
#set first subplot
#plt.subplot(1,2,1)

#plt.imshow(test_input[1])
# set second subplot
#plt.subplot(1,2,2)
#plt.imshow(test_val[3])
# Renders clearly
#plt.show()


# Save Model
# save weights
siamese_model.save('siamesemodelv2.h5')

# Reload model
model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy' : tf.losses.BinaryCrossentropy})
# Make predictions with reloaded model
model.predict([test_input, test_val])
#model.summary()

# Real time Test
# 8.1 Verification Functon

def verify(model, detection_threshold, verification_threshold):
    # Build result array
    results = []
    for image in os.listdir(os.path.join('application_data','verification_images')):
        input_img = preprocess(os.path.join('application_data','input_image','input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data','verification_images', image))
        # Make Predictions
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    # Detection Threshold: Metrics above which a prediction is considered positive
    detection =np.sum(np.array(results) >detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(os.path.join('application_data','verification_images')))
    verified = verification > verification_threshold
    return results, verified

# Detection Threshold: Metric above which a prediction is considered positive
# verification Threshold: proportion of positive predictions/ total positive sample
# opencv verification function
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120 + 250, 200:200 + 250]

    cv2.imshow('Verification', frame)

    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):

        # save input image to application_data/input_image folder
        cv2.imwrite(os.path.join('application_data', 'input_image','input_image.jpg'),frame)
        # Run Verification
        results, verified = verify(siamese_model, 0.5, 0.5)
        print(verified)
    # Breaking gracefully
    if cv2.waitKey(10) & 0XFF == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()

