import gpt_2_simple as gpt2
import os
import requests
import tensorflow as tf
import pickle
import pandas as pd
import time


# Choose GPU
os.environ['CUDA_VISIBLE_DEVICES']='0'

# Limit memory usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def load_pickle(filename):
    with open(filename,'rb') as f:
        obj = pickle.load(f)
    f.close()
    return obj

def dump_pickle(filename,obj):
    with open(filename,'wb') as f:
        pickle.dump(obj,f)

sess = gpt2.start_tf_sess()
#sess = gpt2.reset_session(sess)

test_data = "test_data/test_20_final.txt"
run_name_list = os.listdir("checkpoint/")
run_name_list = [x for x in run_name_list if x.startswith('train_2')]
run_name_list = [x for x in run_name_list if x.split('_')[-3]=='355M']
print(run_name_list)
print(len(run_name_list))
output_file = "generated_para_test20_final_355_train2"

try:
    generated_paraphrase = load_pickle("generated_paraphrase/"+output_file+".pkl")
except:
    generated_paraphrase = {}

run_name_list = [x for x in run_name_list if x not in generated_paraphrase]

num_samples= 1
file = open(test_data,'r')
test_input_list = file.readlines()
model_num = len(run_name_list)
print("number of model in pickle = ",str(len(generated_paraphrase)))
print("number of model to run = ", str(model_num))


for i,run_name in enumerate(run_name_list) :
    #generated_paraphrase[run_name] = {}
    print("=====================================")
    print(run_name)
    print("Model "+str(i+1)+" from "+ str(model_num))
    print("=====================================")
    sess = gpt2.reset_session(sess)
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name= run_name)

    generated_paraphrase[run_name] = {}
    for test_input in test_input_list:
        test_input= test_input.replace('\n','')
        gen_para = gpt2.generate(sess,
              run_name=run_name,
              length=50,
              temperature=1,
              prefix=test_input,
              nsamples=num_samples,
              batch_size=1,#,
              include_prefix = True,
              truncate='<|end of text|>',
              return_as_list=True
              )
        #results will include the prefix since if include_prefix == False, the result sometimes not consistent, so more difficult to preprocess
        generated_paraphrase[run_name][test_input] = {}
        for i,x in enumerate(gen_para):
            generated_paraphrase [run_name][test_input][i] = x.replace(test_input,'').replace('/n','')
        
    
        
    dump_pickle("generated_paraphrase/"+output_file+'.pkl',generated_paraphrase)

    pd.concat({
    v: pd.DataFrame.from_dict(k, 'index') for v, k in generated_paraphrase.items()},
    axis=1).transpose().to_excel("generated_paraphrase/"+output_file+'.xlsx')

