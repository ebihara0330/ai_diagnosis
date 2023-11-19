## AI診断API

## API概要
* 入力データを法規情報として扱う必要があるかを確認する
  * 入力：法規原文、法規和訳、原文の言語情報
  * 出力：AI判断結果、AI出力値、AI確信度、キーワードハイライト(原文/和訳)
　　※入力パラメータエラーはstatus_code:400、その他エラーは500を返却する

## 処理概要　
* AI診断の対応可否（原文があるか、原文の言語にAIが対応しているか）をチェックする
* AI診断が可能なら、パラメータで指定されたAIモデルを使って原文を法規情報として扱う必要があるかを確認する
  * AIモデルは複数設定可（1つでも複数でもOK）複数の場合はパラメータのアンサンブル比率を元にを加重平均したものを最終的なAIの出力値（0～1）とする
  * AI出力値を算出したら、パラメータの閾値（出力値が～以上なら要、それ以外は不要）を元に確認結果を算出する
* AI確信度は、閾値・AIの最大/最小出力値を元に、AI診断の出力値が閾値からどれくらい離れているかで算出する
  * AI確信度は10%～90%で、AI診断の出力値が閾値から離れている程高くなる
* キーワードハイライトは、shapを使って原文/和訳の注目部分に色付けしたhtmlテキストを返却する

## 処理構成
* 事前に作成済のAIモデルとパラメータを使ってAI診断を行う
* 単体で利用するときはAI診断のテストクラスを実行する

```
■AI診断のプロジェクト構成
root
└models
| └ AI診断に使用するmbert/xlmroberta/setfitそれぞれの学習済モデル　
└ai_diagnosis_parameter.json：診断に利用するAI情報
└ai_diagnosis.py：AI診断処理
└tests
  └ ai_diagnosis_test.py：AI診断のテストクラス
  └ ai_diagnosis_parameter_textual.txt：テスト時の法規原文格納用
  └ ai_diagnosis_parameter_ja_translation.txt：テスト時の法規和訳格納用

```

## AI診断API（テストクラス）の利用方法
1. AI診断の実行パラメータ(ai_diagnosis_parameter.json)を設定
2. ai_diagnosis_testで引数に法規原文、法規和訳、原文の言語情報を設定してAI診断API（ai_diagnosis.runメソッド）を実行する

```
■参考：診断に利用するAI情報（ai_diagnosis_parameter.json）の設定内容
* threshold（閾値）： AI出力値を要/不要に分類するための値で0～1を設定する。AI出力値が閾値以上なら要として扱われる
* ai_min_output_value：学習時に検出した最小のAI出力値。AI確信度（不要判定）を算出するために使用する
* ai_max_output_value：学習時に検出した最大のAI出力値。AI確信度（要判定）を算出するために使用する
* input_models：アンサンブルで利用するAIモデルを格納するオブジェクト（複数設定可）
  * name：AIモデル名（mbert/setfit/xlmroberta）
  * auroc：AIモデルの精度（0～1で1に近い方が精度が高い）を設定する。一番精度の高いモデルをshapに使用する
  * directory：AIモデルに関連するファイル群の格納先
  * ensemble_ratio：AIが出力する結果のアンサンブル比率（高いほどそのモデルの出力値の利用度が高くなる）
  * supported_languages：モデルが対応する言語を記載する 入力の言語が非対応なら診断不可で返却する

------------------------------------------------------------
＜ai_diagnosis_parameter.jsonの設定例＞
{
  "threshold": 0.0054,
  "ai_min_output_value": 0.0001,
  "ai_max_output_value": 0.9,
  "input_models": [
    {
      "name": "mbert",
      "auroc": "1",
      "directory": "models/yamaguchi_bert-base-multilingual-cased_v3_english_hojoren_undersampled-4000",
      "ensemble_ratio": 80,
      "supported_languages": ["ja", "en", "fa" ]
    },
    {
      "name": "setfit",
      "auroc": "0.5",
      "directory": "models/ono_paraphrase-multilingual-mpnet-base-v2_v5_setfit-cabinetonly-undersampling4000",
      "ensemble_ratio": 1,
      "supported_languages": ["ja", "en"]
    },
    {
        "name": "xlmroberta",
        "auroc": "0.3",
        "directory": "models/ono_xlm-roberta-base_v5_partial-legalinfor_jap-undersampling4000_ep3_lr1e-5",
        "ensemble_ratio": 2,
        "supported_languages": ["ja", "en", "bn"]
    }
  ]
}
------------------------------------------------------------

```

