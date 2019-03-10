"""
deep learning-aided message passing detection

Editors: Yuhao and Lahiru
"""

from __future__ import print_function
import tensorflow as tf
from Indicate_SMP import Indic
import numpy as np
import os
# In[2]:
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#####################  training SNR ####################
times=2#from 1e-3 to 3e-6
startLR=3.165e-5
training_epochs = 20
total_data =2000000
batch_size = 2000
total_batch = total_data/batch_size

total_test =500000
test_batch_size=2000
test_batch = total_test/test_batch_size


# SMP parameters
Max=60 #for clipping
Nc = 4
Ns = 20
Np = 4
M = 16
Pa= 0.1
snr = 10
Sigma_x=1
Sigma_n=Sigma_x*np.power(10,(-0.1) * snr)/Nc
lsp=tf.log((Pa/Np)/(1-(Pa/Np)))
C2decEdge=Ns*Np
feature_size=M*Np
label_size=Ns*Np
H_size=M*Ns*Np

Iter=8        ##################################################   Attention  HERE   Only 8 Iterations for convenience     ################################################################
tol_layers=4*Iter+2
Nlayers=4*Iter

# Define hyperparameters
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("input_file_format", "tfrecord", "Input file format")
# 文件路径设置
flags.DEFINE_string("train_file_feature", "./data/trainY.csv.tfrecords",
                    "The glob pattern of train_feature TFRecords files")

flags.DEFINE_string("train_file_labels", "./data/trainS.csv.tfrecords",
                    "The glob pattern of train_labels TFRecords files")

flags.DEFINE_string("H_file", "./data/trainH.csv.tfrecords",
                    "The glob pattern of H_file TFRecords files")
###########################################################
# 文件路径设置
flags.DEFINE_string("test_file_feature", "./test/testY.csv.tfrecords",
                    "The glob pattern of train_feature TFRecords files")

flags.DEFINE_string("test_file_labels", "./test/testS.csv.tfrecords",
                    "The glob pattern of train_labels TFRecords files")

flags.DEFINE_string("test_H_file", "./test/testH.csv.tfrecords",
                    "The glob pattern of H_file TFRecords files")
############################################
def read_and_decode_tfrecord(filename_queue, file_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "label": tf.FixedLenFeature([file_size], tf.float32),
        })
    label = features["label"]
    return label
#####################################
# In[4]:
EPOCH_NUMBER = None
BATCH_THREAD_NUMBER = 1
MIN_AFTER_DEQUEUE = 5
BATCH_CAPACITY = BATCH_THREAD_NUMBER * batch_size + MIN_AFTER_DEQUEUE
INPUT_FILE_FORMAT = 'tfrecord'
# Read TFRecords files for training
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(FLAGS.train_file_feature), num_epochs=EPOCH_NUMBER)

if INPUT_FILE_FORMAT == "tfrecord":
    label = read_and_decode_tfrecord(filename_queue, feature_size)
    batch_features = tf.train.batch(
        [label],
        batch_size=batch_size,
        num_threads=BATCH_THREAD_NUMBER,
        capacity=BATCH_CAPACITY
    )
################################################################
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(FLAGS.train_file_labels), num_epochs=EPOCH_NUMBER)
if INPUT_FILE_FORMAT == "tfrecord":
    label2 = read_and_decode_tfrecord(filename_queue, label_size)
    batch_labels = tf.train.batch(
        [label2],
        batch_size=batch_size,
        num_threads=BATCH_THREAD_NUMBER,
        capacity=BATCH_CAPACITY
    )
    ###########################
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(FLAGS.H_file), num_epochs=EPOCH_NUMBER)
if INPUT_FILE_FORMAT == "tfrecord":
    label3 = read_and_decode_tfrecord(filename_queue, H_size)
    train_H = tf.train.batch(
        [label3],
        batch_size=batch_size,
        num_threads=BATCH_THREAD_NUMBER,
        capacity=BATCH_CAPACITY
    )
    ##############################

# In[5]:

# Read TFRecords files for testing
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(FLAGS.test_file_feature), num_epochs=EPOCH_NUMBER)
if INPUT_FILE_FORMAT == "tfrecord":
    test_label1 = read_and_decode_tfrecord(filename_queue, feature_size)
    test_features = tf.train.batch(
        [test_label1],
        batch_size=test_batch_size,
        num_threads=BATCH_THREAD_NUMBER,
        capacity=BATCH_CAPACITY,
        # min_after_dequeue=MIN_AFTER_DEQUEUE
    )
################################################################
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(FLAGS.test_file_labels), num_epochs=EPOCH_NUMBER)
if INPUT_FILE_FORMAT == "tfrecord":
    test_label2 = read_and_decode_tfrecord(filename_queue, label_size)
    test_labels = tf.train.batch(
        [test_label2],
        batch_size=test_batch_size,
        num_threads=BATCH_THREAD_NUMBER,
        capacity=BATCH_CAPACITY
    )
    ###########################
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(FLAGS.test_H_file), num_epochs=EPOCH_NUMBER)
if INPUT_FILE_FORMAT == "tfrecord":
    test_label3 = read_and_decode_tfrecord(filename_queue, H_size)
    test_H = tf.train.batch(
        [test_label3],
        batch_size=test_batch_size,
        num_threads=BATCH_THREAD_NUMBER,
        capacity=BATCH_CAPACITY
    )
    ##############################


#########  Initial Input signals:  y=Hs+n      ###############

Onetem=tf.constant(1.0, dtype=tf.float32, shape=[batch_size,1], name='Onetem')

############### indicate matrix between layers ################
mask_Y_np,mask_A2B_np,mask_B2C_np,mask_C2D_np,mask_A2D_np,mask_D2A_np,mask_A2dec_np,mask_sigma2V_np,mask_H22A_np,mask_lsp2B_np,mask_logPa2C_np,mask_lsp2D_np,mask_lsp2dec_np,mask_C2dec_np= Indic(M,Ns,Np)

mask_Y  =tf.constant(mask_Y_np.copy(),dtype=np.float32)
mask_A2B=tf.constant(mask_A2B_np.copy(),dtype=np.float32)
mask_B2C=tf.constant(mask_B2C_np.copy(),dtype=np.float32)
mask_C2D=tf.constant(mask_C2D_np.copy(),dtype=np.float32)
mask_A2D=tf.constant(mask_A2D_np.copy(),dtype=np.float32)
mask_D2A=tf.constant(mask_D2A_np.copy(),dtype=np.float32)
mask_A2dec=tf.constant(mask_A2dec_np.copy(),dtype=np.float32)

# mask_sigma2V=tf.constant(mask_sigma2V_np.copy(),dtype=np.float32)
# mask_lsp2B=tf.constant(mask_lsp2B_np.copy(),dtype=np.float32)
# mask_logPa2C=tf.constant(mask_logPa2C_np.copy(),dtype=np.float32)
# mask_lsp2D=tf.constant(mask_lsp2D_np.copy(),dtype=np.float32)
# mask_lsp2dec=tf.constant(mask_lsp2dec_np.copy(),dtype=np.float32)

mask_C2dec=tf.constant(mask_C2dec_np.copy(),dtype=np.float32)
###################
# s = batch_labels
# y = batch_features
# H = train_H
s = test_labels
y = test_features
H = test_H
H2=tf.square(H)
################### feedforward network ##############################
D_ini=tf.constant(0.0, dtype=tf.float32, shape=[batch_size,M*Ns*Np], name='D_ini')
with tf.device("/gpu:0"):
        layers = {}
        for h1 in range(0, Nlayers - 1):  # Nlayers-1 is layer D so the hidden is terminated when layer C is finished
            if h1 == 0:          #Initial A
                layer_tem = {}
                layer_tem['W_U_D2A']=tf.Variable(mask_D2A_np.copy(), dtype=np.float32, name='W_U_ini_A')
                layer_tem['W_V_D2A']=tf.Variable(mask_D2A_np.copy(), dtype=np.float32, name='W_V_ini_A')
                layer_tem['W_V_sigma2V']=tf.Variable(mask_sigma2V_np.copy(), dtype=np.float32, name='W_V_sigma2V')
                layer_tem['W_A_H22A']=tf.Variable(mask_H22A_np.copy(), dtype=np.float32, name='W_A_H22A')
                layer_tem['W_Y'] = tf.Variable(mask_Y_np.copy(), dtype=np.float32, name='W_in_Y') #for initialization: weight_Y=tf.variable(mask_Y)

                layer_tem['U'] = tf.matmul( H * tf.sigmoid( D_ini ), mask_D2A * layer_tem['W_U_D2A'])
                layer_tem['V'] = tf.matmul( H2 *tf.sigmoid( D_ini ) * tf.sigmoid(- D_ini ), mask_D2A* layer_tem['W_V_D2A']) + Sigma_n*tf.matmul(Onetem, layer_tem['W_V_sigma2V'])

                layer_tem['A'] = (2 * (tf.matmul(y, layer_tem['W_Y']*mask_Y) - layer_tem['U']) * H - H2*tf.matmul(Onetem, layer_tem['W_A_H22A'])) / (2 * layer_tem['V'])
                layers[h1] = layer_tem
                del layer_tem

            elif h1 % 4 == 1:    #B
                layer_tem = {}
                layer_tem['W_A2B']=tf.Variable(mask_A2B_np.copy(), dtype=np.float32, name='W_hid_A2B') #for initialization: weight_A2B=tf.variable(mask_A2B)
                layer_tem['W_lsp2B']=tf.Variable(mask_lsp2B_np.copy(), dtype=np.float32, name='W_hid_lsp2B')
                layer_tem['B']=tf.matmul(layers[h1-1]['A'], mask_A2B * layer_tem['W_A2B']) + lsp*tf.matmul(Onetem, layer_tem['W_lsp2B'])
                layers[h1] = layer_tem
                del layer_tem

            elif h1 % 4 == 2:    #C
                layer_tem = {}
                layer_tem['W_B2C'] = tf.Variable(mask_B2C_np.copy(), dtype=np.float32, name='W_hid_B2C')
                layer_tem['W_logPa2C'] = tf.Variable(mask_logPa2C_np.copy(), dtype=np.float32, name='W_hid_logPa2C')
                B = tf.clip_by_value(layers[h1-1]['B'], tf.reduce_min(layers[h1-1]['B']), Max)         #no weight for this layer due to the complex calculation
                BB = -tf.log_sigmoid(-B)
                BBB = tf.log(Pa)*tf.matmul(Onetem, layer_tem['W_logPa2C']) - tf.matmul(BB, mask_B2C*layer_tem['W_B2C'])
                BBB = tf.clip_by_value(BBB, -Max, tf.reduce_max(BBB))
                layer_tem['C']=-tf.log(tf.exp(-BBB) - 1)
                layers[h1] = layer_tem
                del layer_tem

            elif h1 % 4 == 3:   #D
                layer_tem = {}
                layer_tem['W_C2D'] = tf.Variable(mask_C2D_np.copy(), dtype=np.float32, name='W_hid_C2D') #for initialization: weight_C2D=tf.variable(mask_C2D)
                layer_tem['W_A2D'] = tf.Variable(mask_A2D_np.copy(), dtype=np.float32, name='W_hid_A2D') # for initialization: weight_A2D=tf.variable(mask_A2D)
                layer_tem['W_lsp2D'] = tf.Variable(mask_lsp2D_np.copy(), dtype=np.float32, name='W_hid_lsp2D')
                layer_tem['D'] = lsp*tf.matmul(Onetem, layer_tem['W_lsp2D']) + tf.matmul(layers[h1-1]['C'], mask_C2D * layer_tem['W_C2D']) + tf.matmul(layers[h1-3]['A'], mask_A2D * layer_tem['W_A2D'])
                layers[h1] = layer_tem
                del layer_tem

            elif h1 % 4 ==0:    #A
                layer_tem = {}
                layer_tem['W_U_D2A']=tf.Variable(mask_D2A_np.copy(), dtype=np.float32, name='W_U_hid_D2A')
                layer_tem['W_V_D2A']=tf.Variable(mask_D2A_np.copy(), dtype=np.float32, name='W_V_hid_D2A')
                layer_tem['W_hid_A_H22A']=tf.Variable(mask_H22A_np.copy(), dtype=np.float32, name='W_hid_A_H22A')
                layer_tem['W_V_sigma2V'] = tf.Variable(mask_sigma2V_np.copy(), dtype=np.float32, name='W_V_hid_sigma2V')
                layer_tem['W_Y'] = tf.Variable(mask_Y_np.copy(), dtype=np.float32, name='W_hid_Y')
                layer_tem['U'] = tf.matmul( H * tf.sigmoid(layers[h1-1]['D']), mask_D2A* layer_tem['W_U_D2A'] )
                layer_tem['V'] = tf.matmul( H2 *tf.sigmoid(layers[h1-1]['D']) *tf.sigmoid(-layers[h1-1]['D']), mask_D2A*layer_tem['W_V_D2A']) + Sigma_n*tf.matmul(Onetem, layer_tem['W_V_sigma2V'])
                layer_tem['A'] = (2 * (tf.matmul(y,layer_tem['W_Y']*mask_Y) - layer_tem['U']) * H - H2*tf.matmul(Onetem, layer_tem['W_hid_A_H22A']) ) / (2 * layer_tem['V'])
                layers[h1] = layer_tem
                del layer_tem
        output_layer = {}
        output_layer['W_A2dec']=tf.Variable(mask_A2dec_np.copy(), dtype=np.float32, name='W_out_A2dec')
        output_layer['W_C2dec']=tf.Variable(mask_C2dec_np.copy(), dtype=np.float32, name='W_out_C2dec')
        output_layer['W_lsp2dec'] = tf.Variable(mask_lsp2dec_np.copy(), dtype=np.float32, name='W_out_lsp2dec')
        output_layer['Dec'] = tf.matmul(layers[h1-2]['A'], mask_A2dec*output_layer['W_A2dec']) + layers[h1]['C'] * tf.matmul(Onetem, output_layer['W_C2dec']) + lsp*tf.matmul(Onetem, output_layer['W_lsp2dec'])

########## cost function ############################
        s_pred=output_layer['Dec']
        s_true=s
        sum_entropy = 0
        for i in range(0, Iter - 1):
            tem = tf.matmul(layers[4 * i]['A'], mask_A2dec) + layers[4 * i + 2]['C'] + lsp
            sum_entropy = sum_entropy + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=s_true, logits=tem))  ## middle hidden layer entropy
        cro_entrp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=s_true, logits=s_pred))  ## output layer entropy
        loss = cro_entrp + sum_entropy
        learning_base_rate=tf.placeholder(dtype=tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_base_rate, name='opt').minimize(loss)
        err_num = tf.reduce_sum(tf.abs(tf.cast(s_pred>0,tf.float32) - s_true))
        num_err = tf.Variable(0.0, tf.float32)
        new_num = tf.add(num_err, err_num)
        Test_pe = tf.assign(num_err, new_num)
######### training process #############################
        init = [tf.global_variables_initializer(),tf.local_variables_initializer()]
        saver = tf.train.Saver()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    session.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=session)
    # saver = tf.train.import_meta_graph('NGMPK3M4SNR30.meta')

    saver.restore(session, "./Netvf500/GMP50v.ckpt")
    
    #load_path=saver.restore(session, "./SNR=40dB/Netv/GMP40v")
    #training the NN


################# Train #############################
    # lr=startLR
    # for time in range(0,times):
    #     for epoch in range(0, training_epochs):
    #         for i in range(int(total_batch)):
    #             _o,cr_ent,out_ent,lr = session.run([optimizer, loss,cro_entrp,learning_base_rate],feed_dict={learning_base_rate:lr})
    #             if i%100 == 0:
    #                 print("time:", time, "epoch:", epoch, "batch:",i,"Cross_Entropy:",cr_ent,"Current LR:",lr,"Output_Entropy:", out_ent)
    #     lr=lr/3.165
    # print("Optimization Finished!")
    # save_path = saver.save(session, "./Netvf500/GMP50v.ckpt")

############# test the trained NN #############################
    for j in range(int(test_batch)):
        Pe_sum = session.run(Test_pe)
        print(j, Pe_sum)
    print("Avg_Pe:", Pe_sum/(total_test*Np*Ns))

    # for j in range(int(total_batch)):
    #     Pe_sum = session.run(Test_pe)
    #     print(j, Pe_sum)
    # print("Avg_Pe:", Pe_sum/(total_data*Np*Ns))
