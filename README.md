This project is Sensational News Detection. We have built a Jupyter Notebook using which you can run the code. Following files are used:
1. ObjectiveSensationalist_Baseline.py - This file is used to demonstrate the baseline model i.e. Bernoulli's Naive Bayes model.
2. SensationalismNews_Classification.ipynb - This is a jupyter notebook built on Google Colaboratory tool.

In order for the model to train,  please download the following files in the /data/ folder:
1. glove.6B.100d.txt from https://drive.google.com/open?id=1-FLa3Sfmn5XWyidMs-155tLGAMCWXDnV

In order to test the pre-trained models, please follow the following steps:
1. Download the following file in the /data/ folder:

https://drive.google.com/open?id=1-2ex_uj839Cfd0X1h03DMkKXRnFl09fm

2. In the existing jupyter notebook run the following code in the last cell of the jupyter notebook:

load_model = pickle.load(open('data/finalized_model_300D.sav'))
x_test = feature_test('data/sample_test_data.csv')
pred_result = load_model(x_test)
pred_res_final = finalized_result(pred_result)

