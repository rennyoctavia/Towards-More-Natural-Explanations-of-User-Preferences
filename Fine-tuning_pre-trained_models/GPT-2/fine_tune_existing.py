import gpt_2_simple as gpt2
import os
import requests
import tensorflow as tf
import pickle

import time



# Choose GPU
os.environ['CUDA_VISIBLE_DEVICES']='7'

# Limit memory usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


#function
def dump_pickle(filename,obj):
    with open(filename,'wb') as f:
        pickle.dump(obj,f)

def load_pickle(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    f.close()
    return obj


#parameters
model_name = "355M"
#file_name_list = ["train_2_10k_ver1.txt","train_2_10k_ver2.txt"]
dataset_50 = "train_data/mscoco_train_50k.npz"
dataset_10 = "train_data/mscoco_train_10k.npz"

optimizer = "adam"
learning_rate = 0.00002
batch_size = 2
#run_name_list = ["train_2_50k_ver2_model3","train_2_50k_ver2_model4","train_2_100k_ver2_model3","train_2_100k_ver2_model4"]
run_name_list = os.listdir("checkpoint/")
run_name_list = [x for x in run_name_list if x.endswith("sequence")]
print(run_name_list)


steps_list = [int(x.split('_')[-2]) for x in run_name_list] #number of max 
print(steps_list)
sample_every = 100
sample_length=500
sample_num = 1


if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/
    
    
sess = gpt2.start_tf_sess()
run_parameters = load_pickle("param_note/sequence_training_mscoco"+"_"+model_name+"_10k_50k_"+".pkl")

for run_name,steps in zip(run_name_list,steps_list):
    print("============================ Fine Tuning {} =================================".format(run_name))
    sess = gpt2.reset_session(sess)
    sess = gpt2.start_tf_sess()
    start_time = time.time()
    if run_name.split('_')[2] == "50k":
        dataset = dataset_50
    else:
        dataset = dataset_10
   
    gpt2.finetune(sess,
                  dataset = dataset,
                  steps = steps,
                  model_name=model_name,
                  batch_size = batch_size,
                  learning_rate = learning_rate,
                  run_name = run_name,
                  sample_every = sample_every,
                  sample_length = sample_length,
                  sample_num = sample_num,
                  save_every = steps,
                  optimizer = optimizer,
                  overwrite = True #to continue fine tuning from latest state
                 )# steps is max number of training steps
    e = int(time.time() - start_time)
    end_time = '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)
    
    run_parameters[run_name] = {"dataset": dataset,
            "steps":steps,
            "model_name":model_name,
            "batch_size":batch_size,
            "learning_rate": learning_rate,
            "optimizer":optimizer,
            "sample_every":sample_every,
            "sample_length":sample_length,
            "sample_num":sample_num,
            "save_every":steps,
            "time_elapsed" : e, 
            "time_elapsed_str": end_time
    }
                
                
dump_pickle("param_note/sequence_training_mscoco"+"_"+model_name+"_10k_50k_"+".pkl",run_parameters)
                
