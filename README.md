# meta_learned_review_helpfulness_prediction

## Installation

1. 下载代码，创建环境

    ```bash
    $ git clone git@github.com:guzy0324/meta_learned_review_helpfulness_prediction.git
    $ cd meta_learned_review_helpfulness_prediction
    $ conda env create -f env.yml
    ```

2. 注册kaggle账户并[获得kaggle授权](https://www.kaggle.com/docs/api#authentication)

3. 激活环境，退出环境

    ```bash
    # 激活环境
    $ conda activate MeRH
    # 退出环境
    $ conda deactivate
    ```

## Demo

1. PRH-Net fit

    ```python
    python -m MeRH.PRH_Net fit -s 42 -bs 2 -d amazon-9_splited -da '{"count": 700}' -nw 128 -ss '["Grocery & Gourmet Food"]' -e glove -rr BiLSTM -rra '{"r": 3}' -o AdamW -as Identification -me 30 -a 1
    ```

2. MeRH fit

    ```python
    python -m MeRH.MeRH fit -s 42 -d amazon-9_splited -ss '["Grocery & Gourmet Food"]' -bs 8 -nw 128 -e glove -rr BiLSTM -rra '{"r": 3}' -a 1 -o AdamW -as Identification -me 20
    ```
