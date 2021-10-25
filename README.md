# Winner at FIRE 2021 Shared Task: HASOC - Abusive and Threatening language detection in Urdu

## Overview
This is official Github repository of team **hate-alert** which ranked 1st in the shared task on ```Abusive and Threatening language detection in Urdu``` at the [FIRE-2021(CICLing 2021 track @ FIRE 2021 co-hosted with ODS SoC 2021)](https://www.urduthreat2021.cicling.org/), part of the [FIRE 2021](http://fire.irsi.res.in/fire/2021/home/) conference. 

## Authors: Mithun Das, Somnath	Banerjee, Punyajoy Saha
Social media often acts as breeding grounds for different forms of abusive content. For low resource languages like Urdu the situation is more complex due to the poor performance of multilingual or language-specific models and lack of proper benchmark datasets. Based on this shared task ```HASOC - Abusive and Threatening language detection in Urdu``` at FIRE 2021, we present an exhaustive exploration of different machine learning models, Our models trained separately for each language secured the 1st position in both abusive and threat detection task in Urdu.



## Sections
1. [System description paper](#system-description-paper)
2. [Task Details](#task-details)
3. [Methodology](#reproducing-results) 
4. [Results](#results)
5. [Acknowledgements](#acknowledgements)

## System Description Paper  
Our paper can be found [here](https://arxiv.org/linkcommingsoon).    

## Task Details
The shared tasks present in this competition are divided into two parts. Where in one part participants have to focus on detecting Abusive language using twitter tweets in Urdu language [(Subtask A)](https://ods.ai/competitions/Urdu-hack-soc2021) and in other part mainly focusing on detecting Threatening language using Twitter tweets in Urdu language [(Subtask B)](https://ods.ai/competitions/Urdu-hack-soc2021-threat). To download the data, go to the following [link](https://www.urduthreat2021.cicling.org/).



## Methodology
In this section, we discuss the different parts of thepipeline that we followed to detect offensive posts in this dataset
1. [Machine Learning Models](#machine-learning-models)
2. [Transformer Models](#transformer-models)


### Machine Learning Models
As a part of our initial experiments, we used several machine learning models to establish a baseline per-formance. We employed XGBoost, LGBM and trained them with pre-trained Urdu laser embedding. The best results were obtained on XGBoost Classifier  with 0.760 and 0.247  F1-scores on abusive and threat detection respectively.

### Transformer Models
We fine-tuned state-of-the-art multilingual BERT model on the given datasets. The beauty of the mBERT is it is pretrained in unsupervised manner on multilingual corpus. Besides we have used another m-BERT based model which is previously fine-tuned on Arabic hate speech date set. The model has been referred as [```Hate-speech-CNERG/dehatebert-mono-arabic' model```](https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-arabic) The motivation of using the following model in Arabic language because it is [origin of Urdu](https://en.wikipedia.org/wiki/Urdu), so further fine-tuning the model with the Urdu dataset may yield better performance.




## Results  
Results of different models on private test dataset can be found here:
The results have been in terms of the **F1** scores.  

###  F1-score  comparison  for  differnt models: 

<h4 align="center">

|   Classifiers 			 |Abusive | Threat    |
|----------------------------|--------|-----------|
| XGBoost					 | 0.7602   | 0.2471  |
| LGBM   					 | 0.7666   | 0.2047  |
| mBERT    					 | 0.8400   | 0.4696  | 
| dehatebert-mono-arabic     | 0.8806   | 0.5457  | 




## Reproducing Results  

```bash
├── TransformerBasedModel/
├── README.md
└── LICENSE
```

### Citations
Please consider citing this project in your publications if it helps your research.
```
Will add soon.
```

## Acknowledgements    

Additionally, we would like to extend a big thanks to the makers and maintainers of the excellent [HuggingFace](https://github.com/huggingface/transformers) repository, without which most of our research would have been impossible.