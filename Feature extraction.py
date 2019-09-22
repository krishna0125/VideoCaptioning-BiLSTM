from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.preprocessing import image
import numpy as np 
import cv2
import os 

#extracting frames from the video

folder = 'test'
os.mkdir(folder)
vidcap = cv2.VideoCapture("C:/Users/Admin/Desktop/resnet/161212_031_Vietnam_1080p96fps.mp4")
count = 0
while True:
    success,image = vidcap.read()
    if not success:
        break
    cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)),image)
    count += 1
print("{} images are extracted in {}.".format(count,folder))

#Storing all the images in one Variable

folder_path = 'C:/Users/Admin/test/'
images_1 = []
for img in os.listdir(folder_path):
    img = os.path.join(folder_path, img)
    img = image.load_img(img,target_size = (224,224,3))
    img =  np.ones((1,224,224,3))
    images_1.append(img)
      
batch_size = 1
height = 224
width = 224
channels = 3

#Feature Extraction for the Images

inputs = tf.placeholder(tf.float32, shape=[batch_size, height, width, channels])
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    net, end_points = resnet_v1.resnet_v1_50(inputs, is_training=False, reuse=tf.AUTO_REUSE)

saver = tf.train.Saver()    

with tf.Session() as sess:
    saver.restore(sess, 'C:/Users/Admin/resnet_v1_50.ckpt')
    representation_tensor = sess.graph.get_tensor_by_name('resnet_v1_50/pool5:0')
    Features = []
    for i in range(len(images_1)):
        features = sess.run(representation_tensor, {'Placeholder:0': images_1[i]})
        Features.append(features)