import numpy as np
import pandas as pd
import os
import subprocess

# Only in case of Informer, comment out the following code
subprocess.run(["python","./Informer/main_informer.py",
"--model", "informer",
"--data", "Part2",
"--features", "S",
"--attn", "prob",
"--do_predict"])

def vanilla_informer(train_data: str,test_data: str):
    for j in range(0,5,1):
        overall_prediction= []
        for i in range(0,157,1):
            subprocess.run(["python","./Informer/main_informer.py",
            "--model", "informer",
            "--data", "Part2",
            "--features", "S",
            "--attn", "prob",
            "--train_data_path", train_data,
            "--test_data_path", test_data,
            "--label_len",24,
            "--train_epochs",10,
            "--do_predict"])
            block_pred= np.loadtxt(r"./pred_results.csv")
            testing= np.loadtxt(r"./test_data")
            testing=testing[:24]
            testing1= np.append(testing,block_pred)
            np.savetxt(r"./test_data",testing1)
            overall_prediction.append(block_pred)
        overall_prediction= np.array(overall_prediction)
        prediction_loc= f"{train_data}_trained_{test_data}_{j}.csv"
        np.savetxt(f"./prediction_loc")
        
        
train_data= ["L1MAG_train","L2MAG_train","L3MAG_train","LIC_train","gold_price_train","bike_rentals_train","ETTh1_train"]
test_data= ["L1MAG_test","L2MAG_test","L3MAG_test","LIC_test","gold_price_test","bike_rentals_test","ETTh1_test"]


for k in range(len(train_data)):
    if train_data[k] == ETTh1_train:
        for l in range(test_data):
            print(train_data[k],test_data[l])
            vanilla_informer(train_data[k],test_data[l])
    else:
        print(train_data[k],test_data[k])
        vanilla_informer(train_data[k],test_data[k])