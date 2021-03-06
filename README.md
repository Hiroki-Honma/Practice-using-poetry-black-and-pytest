# ディレクトリ構成  

+ /submit
    + /__pycache__
        + __init__.cpython-38.pyc
        + compute.cpython-38.pyc
        + preprocessing.cpython-38.pyc
        + print_.cpython-38.pyc
    + /data
        + climate_precip.csv
        + climate_temp.csv
    + __init__.py
    + compute.py (Module to manage functions that perform calculations.)
    + main.py (Main file)
    + monthly_data.csv (The csv file that was requested to be output in the problem statement)
    + preprocessing.py (Module that manages the processing of data frames)
    + print_.py (Module to manage functions that output to the console)
    
+ /tests (Folder to run pytest)
    + /__pyache__
        + __init__.cpython-38.pyc
        + test_compute.cpython-38-pytest-6.2.5.pyc
        + test_preprocessing.cpython-38-pytest-6.2.5.pyc
        + test_submit.cpython-38-pytest-6.2.5.pyc
    + __init__.py
    + test.csv
    + test_compute.py (Module to test functions in compute.py.)
    + test_preprocessing.py (Module to test functions in preprocessing.py.)
    + test_submit.py
+ poetry.lock
+ pyproject.toml

# 課題にどう取り組んだか
## 取り組んだ手順
### 手順1. poetryでの仮想環境構築  
poetryによる仮想環境を構築
poetry.lockの内容を構築した環境にインストールする。  
### 手順2. 与えられたコードの読解  
pyproject.tomlの内容を見ると, blackの記載が見られたためblackを利用しコードを整形した。
その上でコードを読解した。  
### 手順3. リファクタリング  
1つ1つの動作を細かく関数に分け、コードが最後まで動くかを確認した。その上で各動作(関数)の性質を見て, 
各関数をcompute, preprocessin, print_の3つのモジュールに分けた。  
また, はじめのコードの中には名前をみても何を表しているかわかりにく変数(inner_joinなど)が存在した。
それ故, 変数名も何が格納されているのか分かりやすいように編集をした。  
### 手順4. pytestを用いてユニットテストを実施  
pytestを用いて, 各関数へ値を入力した際に, 想定される出力が正しいかを確認するテストを行った。
### 手順5. blackによるコードの整形  
最後に改めてblackを用いてコードの整形を行った。

# 苦労した点
## リファクタリングの方法
リファクタリング=コードを綺麗にすること, というざっくりした概念はネットで調べて捉えることができた。
しかしながら, 具体的にじゃあ何をどうすればよいのかということで頭を悩ませた。  
特に意識したことは, 以下である。
1. 1つ1つの動作を関数に分けること
2. 関数名と動作の整合性をとること
3. 関数の性質別にモジュールを分けること  

1については, 何か問題が発生した際に, コード修正が行いやすくなると考え実行した。
もし、多くの動作が1つになってしまっていれば, その分修正も大変になる。細かく分けていれば, 対応するところのみを修正すればよくて済む。また, 後にコードのテストが複雑にならないようにすることも考えた。関数によってはテストコードを書きながら, 関数の作成を行った。関数の作成を2については, 関数名と動作が異なっていると, コードの可読性が下がってしまうと
考え意識するようにした。特に, 関数名に含まれていない動作は行わないようにした。例えば, read_dataという
関数で「データをreadした上でmergeまでする」ということは敢えて行わなかった。それは, 上述した細かく分ける
という理由もあるが, 自分以外の人が読んだときに関数の動作が分からなくなってしまう事を想定したためでもある。
3については, 管理を行いやすくするために行った。作業をしていて, 今回作成した関数は目的別に「データ前処理を
行う関数」「計算を行う関数」「コンソールに出力する関数」の3つに分けられると感じた。これらの関数をすべて
1つのパッケージで管理していては段々と煩雑になってしまう。それ故, preprocessing, compute, print_の3つの
モジュールでそれぞれ管理を行った。少し余談になるが, computeで定義した関数はあくまで計算をすることが
目的である。それ故, 計算結果のみを返すように定義し, 結果をデータフレームに反映させるまでの処理は入れない
といった工夫も行った。
## 技術力とそれに付随する知識
poetry, black, pytestに関しては扱ったことが無かったため初めに与えられたファイルを見たときは何が何だか分からなかった。
しかし, ファイルの中身をよく見るとpandas, numpyなど知っている言葉も含まれていた。それ故, poetryやblackなども
何かしらのツールなのだろうと考え, ひたすらにgoogle検索を行った。特にpoetryについては, 情報が少なく英語の資料と動画
を見て仮想環境の構築を行った。pytestについては, 聞いたことはあったが実際に動かしたことがなかった。なので, 結局はコードの
書き方やテスト実行の仕方をgoogle検索した。大学で独学で数学を学んでいたときもそうであったが, 足りない知識やスキルを
キャッチアップしていくことの重要さを感じた。
