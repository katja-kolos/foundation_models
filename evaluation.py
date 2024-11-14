import pandas as pd
import json, ast, re
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from evaluations import caculate_bleu, caculate_rouge, caculate_similariry
from evaluate_acc import *


def eval(RES_DIR:str, MODEL_NAME:str):
    for i in range(1,5):
        df = pd.read_csv(f"{RES_DIR}/{MODEL_NAME}_val_output_setting_{i}.csv", sep="\t")
        df = pd.read_excel("../res/see.xlsx")
        answer_pred_col, solution_pred_col = "answer_output", "solution_output"
        df[[answer_pred_col, solution_pred_col]] = df["output"].apply(string2json).apply(pd.Series)
        df[answer_pred_col] = df[answer_pred_col].apply(lambda x: int(x))
        answer_preds = df[answer_pred_col].tolist()
        solution_preds =  df[solution_pred_col].tolist()
        scores = get_scores(df)
        print_scores(scores)
        metrics = calculate_metrics_solutions(solution_preds , df)
        scores.update(metrics)
        df_save = pd.DataFrame(scores)
        df_save.to_csv(f"{RES_DIR}/{MODEL_NAME}_val_metrics.csv", sep="\t", encoding="utf-8")
       

def string2json(text):
    # convert json-like string to "answer" and "solution"
    dict_json = ast.literal_eval(re.sub("```|\n|json", "", text))
    return dict_json["answer"], dict_json["solution"]


def calculate_metrics_solutions(results, data):
    ## BLEU
    bleu1 = caculate_bleu(results, data, gram=1)
    bleu4 = caculate_bleu(results, data, gram=4)
    print("BLEU-1: %.3f" % bleu1)
    print("BLEU-4: %.3f" % bleu4)

    ## Rouge-L
    rouge = caculate_rouge(results, data)
    print("ROUGE-L: %.3f" % rouge)

    ## Similarity
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    similarity = caculate_similariry(results, data, model)
    print("Similariry: %.3f" % similarity)

    return {"BLEU-1": [bleu1], "BLEU-4": [bleu4], "ROUGE": [rouge], "similarity": similarity}
