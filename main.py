''' Part of code from https://github.com/taki0112/Densenet-Tensorflow'''

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import os
import random
import imageio
import glob
from input_pipeline import *
from ops import *

'''parameters:
     f_dim:
     nb_blocks: number of dense block
     init_learning_rate:
     epsilon: epsilon for AdamOptimizer
     class_num: number of the catogories. (We simplify the number of catogories (of ISIC2019) from 9 to 2.)
     batch_size: batch size
     total_epochs: total number of training epoch
     input_size: size of input image
     crop_window_size: window size of center cropping
     continue_from: Continues from the given run, None does start training from scratch [None]
     continue_from_iteration: Continues from the given iteration (of the given run), 
                            'None does restore the most current iteration [None]
     train_dir: Directory name to store the input images

'''

f_dim = 16
nb_block = 4 
init_learning_rate = 0.2*1e-4
epsilon = 1e-8 
dropout_rate = 0.2

class_num = 2
batch_size = 16
total_epochs = 100
input_size = 256
crop_window_size = 10

train_dir = 'ISIC/ISIC_2019_Training_Input/'

continue_from = None
continue_from_iteration = None


''' train mode can be 'random' or 'dark'
    - 'dark': separate the images into training and testing according to the reference image.
             if the average intensity of one image is greater than the reference image, then the image is 
             considered as training set, otherwise, it's considered as test set.
    - 'random': randomly separate the images into training and testing set. 
                The amount of images of training and testing set are the same with the 'dark' mode.
    The image selection can be achieved with 'labelmaker.py'
'''

train_mode = 'dark' # or 'random'
train_filename = train_mode+'_train.txt'
test_filename = train_mode+'_test.txt'
imgslst, labels = get_image_label(train_dir,train_filename)
test_imgslst, test_label = get_image_label(train_dir,train_filename)

model_name = 'NET.model' 

# create checkpoint directory
checkpoint_dir = 'model_'+train_mode
if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
runs = sorted(map(int, next(os.walk(checkpoint_dir))[1]))
if len(runs) == 0:
    run_nr = 0
else:
    run_nr = runs[-1] + 1
run_folder = str(run_nr).zfill(3)

checkpoint_dir = os.path.join(checkpoint_dir, run_folder)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


x = tf.compat.v1.placeholder(tf.float32, shape=[None, input_size,input_size,3])
batch_images = tf.reshape(x, [-1, input_size, input_size, 3])

label = tf.compat.v1.placeholder(tf.float32, shape=[None, class_num])

learning_rate = tf.compat.v1.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=batch_images, 
                nb_blocks=nb_block, 
                filters=f_dim,
                class_num=class_num,
                dropout_rate=dropout_rate).model

train_loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(label), logits=logits))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
train = optimizer.minimize(train_loss)

correct_prediction = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=label, axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())


with tf.compat.v1.Session() as sess:

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and tf.compat.v1.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.compat.v1.global_variables_initializer())

    if continue_from:
        checkpoint_dir = os.path.join(os.path.dirname(checkpoint_dir), continue_from)
        print('Loading variables from ' + checkpoint_dir)
        load_checkpoint(sess,checkpoint_dir, continue_from_iteration,model_name)
    if continue_from_iteration:
        epoch_start = continue_from_iteration
    else:
        epoch_start = 0

    tf.compat.v1.summary.scalar('loss', train_loss)
    tf.compat.v1.summary.scalar('accuracy', accuracy)

    summary_all = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter('./summary', sess.graph)

    global_step = 0
    epoch_learning_rate = init_learning_rate
    h = input_size
    w = input_size

    for epoch in range(epoch_start,total_epochs):
        if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        total_batch = int(len(imgslst) / batch_size)
        
        # generate random index in one epoch
        idx = random.sample(range(0,len(imgslst)),len(imgslst))

        for step in range(total_batch):
            batch_idx = idx[step*batch_size:(step+1)*(batch_size)]
            batch_x = []
            batch_y = []
            tmp = random.randint(2,6)
            for ii in range(batch_size):
                if ii % tmp ==0:
                    batch_img = get_gray_img(imgslst,batch_idx[ii],input_size,crop_window_size)
                else:
                    batch_img = get_img(imgslst,batch_idx[ii],input_size,crop_window_size)
                batch_x.append(batch_img)
                batch_y.append(labels[batch_idx[ii]])

            batch_img = np.array(batch_x)
            batch_label = np.asarray(batch_y)

            train_feed_dict = {
                x: batch_img,
                label: batch_label,
                learning_rate: epoch_learning_rate
            }

            _, loss, train_accuracy = sess.run([train, train_loss,accuracy], feed_dict=train_feed_dict)
            print("Step:", str(global_step), "Loss:", loss, "Training accuracy:", train_accuracy)
                
            if global_step % 200 == 0:
                train_summary, train_accuracy = sess.run([summary_all, accuracy], feed_dict=train_feed_dict)
                print("Step:", str(global_step), "Loss:", loss, "Training accuracy:", train_accuracy)
                
                writer.add_summary(train_summary, global_step=global_step)

            
            global_step += 1

        saver.save(sess, save_path=os.path.join(checkpoint_dir,model_name),
                            global_step=epoch)


        total_test_batch = int(len(test_imgslst) / batch_size)
        accuracy_all = 0

        for step in range(total_test_batch):
            test_x = []
            test_y = []
            for i in range(batch_size):
                test_x_tmp = get_img(test_imgslst,step*batch_size+i,input_size,crop_window_size)
                test_x.append(test_x_tmp)
                test_y.append(test_label[step*batch_size+i])

            test_feed_dict = {x: test_x,label: test_y,learning_rate: epoch_learning_rate}
            accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)
            accuracy_all = accuracy_all+accuracy_rates

        print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy_all/len(test_imgslst))
            

