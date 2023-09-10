# This is the source code for paper: A Data-centric Framework to Endow Graph Neural Networks with Out-Of-Distribution Detection Ability. If you have any questions about the paper or code, please contact <u>gyxx@bupt.edu.cn</u> .


### Requirements:

- torch-geometric==2.0.4
- torch-scatter==2.0.93
- torch-sparse==0.6.15
- numpy==1.21.2
- pandas==1.3.0
- python==3.9.15
- scikit-learn=1.0.2
- scipy==1.9.3
- torch==1.11.0
- torchvision==0.12.0

### Training:

Run our model with SSD scoring fuction in TU datasets:

```
python clmd.py -DS_pair xxxx 
```


We also provide the code to search hyper-parameters, you can use the following command (for ENZYMES dataset) to run it:
```
bash eme.sh 
```
