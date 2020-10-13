# Japanese-BERT-Recommendation
This is a simple recommedation model example using Japanese BERT by AllenNLP.   
In terms of accuracy, directly using BERT embeddings is not recommended.


![model_image](https://github.com/onozeam/Japanese-BERT-Recommendation/blob/master/model_overview.png)

## Detail
We assume that we are going to apply recommendation model to our media site for articles recommendation.

An object of this model is expressing recommendation score between 0 to 1.0. In general applications, the article whicn has high recommendation score will be recommended.

### Dataset
There are all training datasets in `dataset.csv`. ***The training data is dubby dataset, so the model cannot be improved by the data.*** this dummy data was made by wikipedia.

We assume that `dataset.csv` has transition histories between our website articles which we gathered by GooleAnalytics.

```
# description about columns of dataset.csv

current_text: Article text being viewed
candidate_text: Article text which is a candaidate for recommendation
current_main_category_id: Attribute data 1 of article being viewed
current_sub_category_id: Attribute data 2 of article being viewed
candidate_main_category_id: Attribute data 1 of article which is candidate for recommendation
candidate_sub_category_id: Attribute data 2 of article which is candidate for recommendation
label: training label. we can decide freely the labeling logic. (For example, a transition has been happned even once under the combination of current and candidate article, label becomes 1.)
```
## Install
`pip install -r requiremnts.txt`


`pip install git+https://github.com/allenai/allennlp.git@v1.0.0rc6` 


(Downloading AllenNLP from github, because pre-trained japanese BERT model requires Allennlp >= 1.0. (at 2020/7/10))


## Using
### Training
`python3 main.py`
### Inferrence
`python3 main.py -infer -model_name best.th`
