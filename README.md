# Japanese-BERT-Recommendation
日本語BERTモデルとAllenNLPを用いた記事レコメンデーションモデルの雛形です。
[モデルの画像]

## Detail
メディアサイトで記事のレコメンデーションする際に、日本語BERTの事前学習モデル(東北大学 乾・鈴木研究室)を使用する際にベースとして使用できるモデル設計です。
Youtube社の論文 ([Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/45530.pdf))にインスパイアされています。

### Dataset
閲覧中の記事から推薦候補となる別の記事への遷移しやすさを、0~0.999...の間で表現することがこのモデルのゴールです。一般的には、この値の高い記事を優先して推薦することになるかと思います。

このスクリプトで学習を始める際は、google analyticsなどで、記事間の遷移履歴が事前に取得されており、`dataset.csv`の形式でまとめられていると仮定しています。

テキストデータの他に、記事のカテゴリIDといった属性データも、学習に使用しています。必要に応じてその他の属性値(例えば閲覧ユーザーの属性値など)を`dateset.csv`に追加することで、学習に利用できます。
```
# dataset.csvにおける各カラムの説明

current_text: 閲覧中の記事のテキスト
candidate_text: 推薦候補となる記事のテキスト
current_main_category_id: 閲覧中の記事の属性データその1
current_sub_category_id: 閲覧中の記事の属性データその2
candidate_main_category_id: 閲覧中の記事の属性データその1
candidate_sub_category_id: 閲覧中の記事の属性データその2
label: 閲覧中の記事から、推薦候補の記事への遷移が、これまで発生している場合は1、していない場合は0
```

テキストはwikipediaから引用し、属性データとラベルはランダムな値を用いたデータセットです。(ランダムなラベルを使用しているので、このデータセットでは学習は収束しません。)

## Install
`pip install -r requiremnts.txt`


`pip install git+https://github.com/allenai/allennlp.git@v1.0.0rc6` 


(事前学習済み日本語BERTを使用するためにはAllenNLP1.0以上が必要なため(2020/7/10現在)、github repositoryからダウンロードします。)


## Using
### Training
`python3 main.py`
### Inferrence
`python3 main.py -infer -model_name best.th`
