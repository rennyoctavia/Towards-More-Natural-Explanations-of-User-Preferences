import gpt_2_simple as gpt2
import os
import requests
import tensorflow as tf
import pickle

import time



# Choose GPU
os.environ['CUDA_VISIBLE_DEVICES']='0'

# Limit memory usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

model_name = "355M"
file_name_list = ["train_1_50k_355M.npz"]
optimizer_learningrate_list = [('adam',0.00002)]#,('sgd',0.006)]#('adam',0.00002)
batch_size_list = [1]
steps_list = [1,3,5,10,15,20,25,50,75,100,150,200,250,500,750,1000]#50,75,100,,500,750,1000] #number of max 200,500, 
#steps_list = [250,500,750,1000]

sample_every = 200
sample_length=500
sample_num = 1


if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/
    
    
sess = gpt2.start_tf_sess()

#run_parameters = {}

def dump_pickle(filename,obj):
    with open(filename,'wb') as f:
        pickle.dump(obj,f)

def load_pickle(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    f.close()
    return obj


try:
    run_parameters = load_pickle("param_note/"+"final_50k_train1_2_diff_steps_model_355M.pkl")

except:
    run_parameters = {}

for filename in file_name_list:
    #run_parameters = {} # reset when filename changes
    #run_parameters = load_pickle("param_note/"+filename.split(".")[0]+"_"+model_name+".pkl")
    #run_number = 1 #run number for different dataset_and model_name
    for item in optimizer_learningrate_list:
        optimizer = item[0]
        learning_rate = item[1]
        for batch_size in batch_size_list:
            for steps in steps_list:
                run_name = filename.split(".")[0]+"_"+str(steps)+"_final"
                print("============================ Fine Tuning {} =================================".format(run_name))
                sess = gpt2.reset_session(sess)
                sess = gpt2.start_tf_sess()
                start_time = time.time()
                gpt2.finetune(sess,
                              dataset = "train_data/"+filename,
                              steps = steps,
                              model_name=model_name,
                              batch_size = batch_size,
                              learning_rate = learning_rate,
                              run_name = run_name,
                              sample_every = sample_every,
                              sample_length = sample_length,
                              sample_num = sample_num,
                              save_every = steps,
                              optimizer = optimizer  
                              )# steps is max number of training steps
                e = int(time.time() - start_time)
                end_time = '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)
                run_parameters[run_name] = {"dataset":filename,
                                            "steps":steps,
                                            "model_name":model_name,
                                            "batch_size":batch_size,
                                            "learning_rate": learning_rate,
                                            "optimizer":optimizer,
                                            "sample_every":sample_every,
                                            "sample_length":sample_length,
                                            "sample_num":sample_num,
                                            "save_every":steps,
                                            #"run_number":run_number,
                                            "time_elapsed" : e, 
                                            "time_elapsed_str": end_time
                                           }
                #run_number = run_number + 1
                
                dump_pickle("param_note/"+"final_50k_train1_2_diff_steps_model_355M.pkl",run_parameters)
                
#dataset,
#             steps=-1,
#             model_name='124M',
#             model_dir='models',
#             combine=50000,
#             batch_size=1,
#             learning_rate=0.0001,
#             accumulate_gradients=5,
#             restore_from='latest',
#             run_name='run1',
#             checkpoint_dir='checkpoint',
#             sample_every=100,
##             sample_length=1023,
#             sample_num=1,
##             multi_gpu=False,
#             save_every=1000,
#             print_every=1,
#             max_checkpoints=1,
#             use_memory_saving_gradients=False,
#             only_train_transformer_layers=False,
#             optimizer='adam',
#             overwrite=False):
