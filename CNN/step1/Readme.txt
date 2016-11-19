概要(STEP1)
所定サイズのペットボトル画像の種類を識別する.
- "-mdl": モデル名
　　学習したモデルを保存する場合, 保存したモデルを使って評価を
　行う場合に指定する.
- "-mode": L : 学習, T : 評価, LT : 学習->評価 
  　動作モードを指定する.
　　デフォルトは, LT.
- './log'ディレクトリにサマリーを出力する.
- 学習回数は, pet_mainWWxHH.py内のITELATION_NUMで指定する.
- 学習/評価データは, pet_dataWWxHH.py内のself.fnListで指定する.

使い方
 python pet_mainWWxHH.py <-mdl model> <-mode L or T> 
