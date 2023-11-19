"""
overview:
・入力情報を法規情報として扱う必要があるかをAIが診断する

contents:
・法規原文/和訳、言語情報を元にAI診断を実行する
  - 原文の言語がAI非対応の場合、診断の対象外とする
  - AIが対応可能な場合、原文を元に法規情報の取り扱い要否を推論する
  - 推論後にshapで結果の詳細を取得して、結果および結果の詳細を返却する
"""

import re 
import torch
import json
import shap
import logging
import math
from bs4 import BeautifulSoup
from setfit import SetFitModel
from types import SimpleNamespace
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import pipeline

# GPUの指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ログの設定
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


class DiagnosticResult:
    """
    AI診断結果格納用
    """
    def __init__(self, 
                 ai_result: str = None,
                 ai_output_value: float = None,
                 ai_confidence: int = None,
                 keyword_textual: str = None,
                 keyword_ja_translation: str = None):
        
        """
        Initialize the DiagnosticResult class.
        Args:
        ai_result：AI判断結果
        ai_output_value：AI出力値
        ai_confidence：AI確信度
        keyword_textual：キーワードハイライト(原文)
        keyword_ja_translation：キーワードハイライト(和訳)
        """
        self.ai_result = ai_result
        self.ai_output_value = ai_output_value
        self.ai_confidence = ai_confidence
        self.keyword_textual = keyword_textual
        self.keyword_ja_translation = keyword_ja_translation


def run(textual, ja_translation, language):
    """
    AI診断処理（入力チェック～診断結果の返却）
    
    Args:
    textual：推論用の法規原文
    ja_translation：法規和訳
    language：言語情報
    
    Returns:
    DiagnosticResult：AI診断結果
    """
    try:
        # 設定ファイル読み込み
        params = load_params()
        # 原文・対応言語の入力チェック
        error = validate(params, textual, language)
        if error:
            return error
        # 推論
        ai_output_value = inference(params, textual)
        # AI判断結果取得
        ai_result = get_ai_result(params, ai_output_value)
        # AI確信度取得
        ai_confidence = get_ai_confidence(params, ai_output_value)
        # キーワードハイライト(原文)取得
        keyword_textual = get_keyword(params, textual)
        # キーワードハイライト(和訳)取得
        keyword_ja_translation = get_keyword(params, ja_translation)

        return DiagnosticResult(ai_result, ai_output_value, ai_confidence, keyword_textual, keyword_ja_translation)

    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        return {"status_code": 500, "message": str(e)}


def load_params():
    """
    設定ファイル読み込み
    
    Returns:
    診断で利用するAIモデル情報
    """
    with open('ai_diagnosis_parameter.json', 'r') as f:
        return json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))


def validate(params, textual, language):
    """
    入力チェック処理（原文の有無/対応言語の確認）
    
    Args:
    params：診断で利用するAIモデル情報
    textual：法規原文
    language：言語情報
    
    Returns:
    入力チェックエラー情報
    """
    error = None
    if textual is None or textual == "":
        error = {"status_code": 400, "message": "Textual not set"}
    for input_model in params.input_models:
        if language not in input_model.supported_languages:
            error = {"status_code": 400, "message": "Unsupported language"}
            break
    return error


def inference(params, textual):
    """
    推論処理
    
    Args:
    params：診断で利用するAIモデル情報
    textual：法規原文
    
    Returns:
    アンサンブル後のAI出力値
    """

    # アンサンブル（加重平均）したAIの推論結果
    ai_output_value = 0.0

    # AIモデルの合計アンサンブル比率　
    total_ratio = 0

    for input_model in params.input_models:
        # setfitモデルの推論
        if input_model.name == "setfit":
            setfit_model = SetFitModel.from_pretrained(input_model.directory)
            with torch.no_grad():
                outputs = setfit_model.predict_proba([textual])
            p = torch.nn.functional.softmax(outputs, dim=1).detach().numpy().copy()

        # mbert/xlmrobertaモデルの推論
        if input_model.name == "mbert" or input_model.name == "xlmroberta":
            model = AutoModelForSequenceClassification.from_pretrained(input_model.directory, local_files_only=True).to(device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(input_model.directory)
            inputs = tokenizer(textual, padding="max_length", truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs.to(device))
            logits = outputs.logits.to("cpu")
            p = torch.nn.functional.softmax(logits, dim=1).detach().numpy().copy()

        # 推論の結果を重みづけしてAI出力値に加算
        ai_output_value += p[0,1] * input_model.ensemble_ratio

        # モデルのアンサンブル比率を加算
        total_ratio += input_model.ensemble_ratio

    # アンサンブル結果 = 重みづけした推論結果の合計値 / 合計比率
    ai_output_value = ai_output_value / total_ratio

    return ai_output_value


def get_ai_result(params, ai_output_value) :
    """
    AI判断結果取得
    
    Args:
    params：診断で利用するAIモデル情報
    ai_output_value：アンサンブル後のAI出力値
    
    Returns:
    AI判断結果
    """
    return "〇" if ai_output_value > params.threshold else "×"


def get_ai_confidence(params, ai_output_value):
    """
    AI確信度取得
    
    Args:
    params：診断で利用するAIモデル情報
    ai_output_value：アンサンブル後のAI出力値
    
    Returns:
    AI確信度
    """

    # AI出力値が閾値よりも大きい場合
    if ai_output_value > params.threshold:
        # （AI出力値（最大）- 閾値）/ 9 でプラスの確信度10%あたりの値を算出する
        positive_confidence_value = (params.ai_max_output_value - params.threshold) / 9
        # 閾値とAI出力値の差分 / プラスの確信度10%あたりの値（少数点切り上げ） * 10で確信度（10%～90）を算出する
        # なお、プラスの場合は閾値と比較して、より大きいAI出力値の確信度が高くなる
        # ※閾値に近い場合は10%、最大出力値に近い場合は90%のイメージ
        ai_confidence = math.ceil((ai_output_value - params.threshold) / positive_confidence_value) * 10

    # AI出力値が閾値未満の場合
    else:
        # （閾値 - AI出力値（最小））/ 9 でマイナスの確信度10%あたりの値を算出する
        negative_confidence_value = (params.threshold - params.ai_min_output_value) / 9
        # 閾値とAI出力値の差分 / マイナスの確信度10%あたりの値（少数点切り上げ） * 10で確信度（10%～90）を算出する
        # なお、マイナスの場合は閾値と比較して、より小さいAI出力値の確信度が高くなる
        # ※閾値に近い場合は10%、最小出力値に近い場合は90%のイメージ
        ai_confidence = math.ceil((params.threshold - ai_output_value) / negative_confidence_value) * 10

    # 計算結果が0の場合（AI出力値=閾値）は10におきかえる、90より大きい場合（AI出力値<最小値 or 最大<AI出力値）は90に置き換える
    if ai_confidence == 0:
        ai_confidence = 10
    elif ai_confidence > 90:
        ai_confidence = 90

    return ai_confidence


def get_keyword(params, regulatory_text) :
    """
    キーワードハイライト取得
    
    Args:
    params：診断で利用するAIモデル情報
    regulatory_text：法規テキスト（原文 or 和訳）
    
    Returns:
    法規テキストのキーワードにハイライトを設定したhtmlデータ
    """

    # 入力の中で一番aurocが高いモデル（複数ある場合は先に取得したモデル）を取得する
    input_model = max(params.input_models, key=lambda x: float(x.auroc))
    model = pipeline('text-classification', model=input_model.directory, return_all_scores=True, device=torch.device(f'cuda:{0}'))
    
    # テキストを510トークン相当（shapに入力できる最大トークン数）にトリミング
    MAX_SEQ_LEN = 510
    tokenizer = AutoTokenizer.from_pretrained(input_model.directory)
    regulatory_text_trimmed = tokenizer.decode(tokenizer.encode(regulatory_text,truncation=True,max_length = MAX_SEQ_LEN), skip_special_tokens=True).replace('\\n', '')

    # shap（推論結果の解析）実行
    try:
        explainer = shap.Explainer(model, output_names=["不要", "要"]) 
        shap_values = explainer([regulatory_text_trimmed])
    except RuntimeError as e:
        # トークン超過の場合はエラーメッセージを元に超過分を除外して再実行
        matches = re.findall(r"\((.*?)\)", str(e))
        diff = int(matches[0])-int(matches[1])
        MAX_SEQ_LEN_new = 510-(diff+3)
        text_new = tokenizer.decode(tokenizer.encode(regulatory_text_trimmed,truncation=True,max_length = MAX_SEQ_LEN_new), skip_special_tokens=True)
        try:
            shap_values = explainer([text_new])
        # 再エラーで結果が取得できない場合はキーワード取得終了
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)
            return
    # その他エラーで結果が取得できない場合もキーワード取得終了
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        return

    # 要の出力に関連するShap値の影響度に応じて色付け（＋の影響なら赤－の影響なら青）されたhtmlを出力する
    html = shap.plots.text(shap_values[0, :, "要"],display=False)

    # BeautifulSoupでhtmlを要素分解して検索できるようにする
    soup = BeautifulSoup(html, 'html.parser')

    # ベクターグラフィック+原文で構成されているので不要なベクター要素を削除
    svg_element = soup.find('svg')
    if svg_element:
        svg_element.extract()

    # shapの結果が設定されたhtmlの色付けを調整する
    # @todo : ハイライトなしにする場合は全ての要素に色をクリアする処理を実行する
    for i,div in enumerate(soup.find_all('div', id=True)):
        background_value = next((item.strip() for item in div['style'].split(';') if 'background' in item), None)
        match = re.search(r'rgba\((\d+\.?\d*), (\d+\.?\d*), (\d+\.?\d*), (\d+\.?\d*)\)', background_value)
        r, g, b, a = match.groups()
        rgba = [float(val) for val in [r, g, b, a]]
        # 濃い色（影響が大きい要素）は色の透明度を高くする（1→0.25）
        if (rgba[3]==1):
            div['style'] = div['style'].replace(background_value, f'background: rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {0.25})')
        # それ以外（影響の小さい要素）は色をクリアする
        else:
            div['style'] = div['style'].replace(background_value, 'background: rgba(0.0, 0.0, 0.0, 0.0)')

    return soup.prettify()
