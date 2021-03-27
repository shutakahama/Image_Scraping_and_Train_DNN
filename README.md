# Image_Scraping_and_Train_DNN
Image scraping from google image search and finetune DNNs with saved images.

Google画像検索から画像をスクレイピングし，それを使ってCNNモデルを訓練するプログラムです．

## スクレイピング
入力されたクエリ（ワード）をGoogle画像検索で検索し，画像ファイルを保存します．  
ルートディレクトリ以下に，./img/{category_name} というフォルダを作って保存します．

### download_image_fast.py
検索画面にあるURLをダウンロードするので，速いですが画質は悪いです．

```
python download_image_fast.py
```

### download_image_fast.py
各画像を一度クリックしてからダウンロードするので，遅いですが画質はより良いです．

```
python download_image_high-quality.py
```

## モデル学習
./img 以下に保存された各クラスの画像を使ってCNNを訓練します．  
trainとtestはランダムに分割されます．  
モデルは[inception(googlenet), squeezenet, vgg, resnet, mobilenet]から選ぶことができます．  
学習後のモデルは./trained_models に保存されます．  

```
python train.py
```
