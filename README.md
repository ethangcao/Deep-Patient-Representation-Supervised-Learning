# Goal
1. Learning robust patient representations from multi-modal electronic health records
2. Experimenting different multi-modal concatenation architectures
3. Implementing learning algorithm for multi-modal similarity and between-sample variance

# Result
Tested on Bi-LSTM and Transformer encoder for LFpool (benchmark) and SDPRL, outperform by 13.4% in AUROC.

# Dependencies
Please refer to environment.yml

# Download instruction of data
To access data, which is the output of MIMIC extract, you will need to be credentialed for MIMIC-III GCP access through physionet. GCP repo is https://console.cloud.google.com/storage/browser/mimic_extract.

# Code Instruction
1. download all_hourly_data.h5 from the MIMIC extract GCP repo, put under /data.

2. run /code/etl_data.py to preprocess data, output will be under /data/processed.

3. change the experiment setup in /code/main.py and run for different fusion methods, encoders and tasks.

4. change the experiment setup in /code/main_SDPRL.py and run for different encoders and tasks.

5. /code/mydatasets.py contains the dataset constructor and collate function

6. /code/mymodels.py contains all model architectures

7. /code/utils.py & utils_SDPRL.py contains utiliy function for training and evaluation.

8. /code/plots.py generates training curve.

9. Output will be under /output
