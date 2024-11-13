import tensorflow as tf
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from model import Unet
from util import load_image_test, load_image_train

dataset, info = tfds.load(data_link, with_info=True)


train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)


BATCH_SIZE = 50
BUFFER_SIZE = 1000

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

class_names = ['pet', 'background', 'outline']

model = Unet()

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
TRAIN_LENGTH = info.splits['train'].num_examples
EPOCHS = 10
VAL_SUBSPLITS = 5
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset)

model.save("unet.h5")


def get_test_image_and_annotation_arrays():
  ds = test_dataset.unbatch()
  ds = ds.batch(info.splits['test'].num_examples)
  
  images = []
  y_true_segments = []

  for image, annotation in ds.take(1):
    y_true_segments = annotation.numpy()
    images = image.numpy()
  
  y_true_segments = y_true_segments[:(info.splits['test'].num_examples - (info.splits['test'].num_examples % BATCH_SIZE))]
  
  return images[:(info.splits['test'].num_examples - (info.splits['test'].num_examples % BATCH_SIZE))], y_true_segments


def create_mask(pred_mask):

  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0].numpy()


def make_predictions(image, mask, num=1):

  image = np.reshape(image,(1, image.shape[0], image.shape[1], image.shape[2]))
  pred_mask = model.predict(image)
  pred_mask = create_mask(pred_mask)
  return pred_mask

"""### Compute IOU score

"""

def class_iou(y_true, y_pred):
  class_wise_iou = [] 
  smoothening_factor = 0.00001
  for i in range(3):    
    intersection = np.sum((y_pred == i) * (y_true == i))
    y_true_area = np.sum((y_true == i))
    y_pred_area = np.sum((y_pred == i))
    
    combined_area = y_true_area + y_pred_area
    
    iou = (intersection + smoothening_factor) / (combined_area - intersection + smoothening_factor)
    class_wise_iou.append(iou*100)   
  return class_wise_iou

"""With all the utilities defined, you can now proceed to showing the metrics and feeding test images.

"""

# get the ground truth from the test set
y_true_images, y_true_segments = get_test_image_and_annotation_arrays()
# feed the test set to th emodel to get the predicted masks
results = model.predict(test_dataset, steps=info.splits['test'].num_examples//BATCH_SIZE)
results = np.argmax(results, axis=3)
results = results[..., tf.newaxis]

# compute the class wise metrics
cls_wise_iou= class_iou(y_true_segments, results)

# show the IOU for each class
for idx, iou in enumerate(cls_wise_iou):
  spaces = ' ' * (10-len(class_names[idx]) + 2)
  print("{}{}{} ".format(class_names[idx], spaces, iou))

"""### Show Predictions"""

# Please input a number between 0 to 3647 to pick an image from the dataset
ind = 346

# Get the prediction mask
y_pred_mask = make_predictions(y_true_images[ind], y_true_segments[ind])

# Compute the class wise metrics
iou= class_iou(y_true_segments[ind], y_pred_mask)  

# Overlay the metrics with the images
display_with_metrics([y_true_images[ind], y_pred_mask, y_true_segments[ind]], iou )

import cv2
import matplotlib.pyplot as plt
im1=cv2.imread('/content/pet.jpeg')
print(im1.shape)
im2=cv2.imread('/content/pet1.jpg')
print(im2.shape)
frame = cv2.resize(im1, (128, 128))
frame1 = cv2.resize(im2, (128, 128))
plt.imshow(frame)

frame = np.expand_dims(frame, axis=0)
frame = frame / 255.0
mask = model.predict(frame)[0]
plt.imshow(mask)

plt.imshow(frame1)

frame1 = np.expand_dims(frame1, axis=0)
frame1 = frame1 / 255.0
mask1 = model.predict(frame1)[0]
plt.imshow(mask1)

frame=cv2.imread('/content/smile.jpg')
  H, W, _ = frame.shape
  
  ori_frame = frame
  frame = cv2.resize(frame, (128, 128))
  frame = np.expand_dims(frame, axis=0)
  frame = frame / 255.0

  mask = model.predict(frame)[0] 
  mask = mask.astype(np.float32)  
  mask = cv2.resize(mask, (W, H))  
  combine_frame = ori_frame * mask
  combine_frame = combine_frame.astype(np.uint8)
  plt.imshow(combine_frame)

frame=cv2.imread('/content/pet.jpeg')
  H, W, _ = frame.shape
  
  ori_frame = frame
  frame = cv2.resize(frame, (128, 128))
  frame = np.expand_dims(frame, axis=0)
  frame = frame / 255.0

  mask = model.predict(frame)[0] 
  mask = mask.astype(np.float32)  
  mask = cv2.resize(mask, (W, H))  
  combine_frame = ori_frame * mask
  combine_frame = combine_frame.astype(np.uint8)
  plt.imshow(combine_frame)