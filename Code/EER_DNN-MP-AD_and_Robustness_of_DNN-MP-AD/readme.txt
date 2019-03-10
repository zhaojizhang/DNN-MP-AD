This is the code for the simulations in Fig. 7 and Fig. 9. Use the codes as follows.

1. run new_generate.m with desired parameters to generate training data and test data.
2. run generate_csv_tfrecords.py to transfer .csv files into .csv_tfrecords files.
3. place .csv_tfrecords files for training into "data" folder and  place .csv_tfrecords files for testing into "test" folder.
4. run tensorflow_DNN_SMP_fullweight.py to do the training.
5. add additional random error to H in step 1 to generate the training and testing samples for Fig. 9.