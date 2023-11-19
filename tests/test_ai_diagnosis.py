"""
overview:
・AI診断APIのテストクラス

contents:
・以下のパラメータを設定して本pythonクラスを実行する
  - 診断に利用するAIモデル情報：../ai_diagnosis_parameter.jsonに診断に利用するAIモデルの情報を設定
  - 法規原文：ai_diagnosis_parameter_textual.txtに原文テキストを設定
  - 法規和訳：ai_diagnosis_parameter_ja_translation.txtに和訳テキストを設定
  - 言語情報：ai_diagnosis.runの第三引数に言語文字列を設定
  ※ AI診断APIの動作概要および、AIモデル情報の設定方法は../ai_diagnosis_readme.md参照
"""
import sys
sys.path.append("/home/ubuntu/git_repo/hm_ai_api")
import ai_diagnosis

def test():
    """
    AI診断のテストメソッド
    """
    result = ai_diagnosis.run(get_textual(), get_ja_translation(), "ja")
    print(result)


def get_textual():
    """
    テスト用の原文テキスト取得処理

    Returns:
    原文テキスト
    """
    with open('/home/ubuntu/git_repo/hm_ai_api/tests/ai_diagnosis_parameter_textual.txt', 'r', encoding='utf-8') as file:
        return file.read().replace("\\n", "\n")


def get_ja_translation():
    """
    テスト用の和訳テキスト取得処理

    Returns:
    和訳テキスト
    """
    with open('/home/ubuntu/git_repo/hm_ai_api/tests/ai_diagnosis_parameter_ja_translation.txt', 'r', encoding='utf-8') as file:
        return file.read().replace("\\n", "\n")

if __name__ == '__main__':
    test()
