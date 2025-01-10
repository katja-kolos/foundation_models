import re
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import util
import evaluate
meteor = evaluate.load("meteor")
from sentence_transformers import SentenceTransformer

def extract_explanation(text):
    text = re.sub(r"The answer is [A-Z]. BECAUSE: ", "", text)
    return text


########################
## BLEU
########################
def tokenize(text):
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


def bleu_score(reference, hypothesis, gram):
    
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    if gram == 1:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1., ))  # BELU-1
    elif gram == 2:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 2., 1. / 2.))  # BELU-2
    elif gram == 3:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 3., 1. / 3., 1. / 3.))  # BELU-3
    elif gram == 4:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 4., 1. / 4., 1. / 4., 1. / 4.))  # BELU-4

    return bleu


def caculate_bleu(results, df, gram):
    bleus = []
    for qid, output in enumerate(results):
        # prediction = extract_explanation(output)
        prediction = output
        target = df["lecture"][qid] + " " + df["solution"][qid]
        target = target.strip()
        if target == "":
            continue
        bleu = bleu_score(target, prediction, gram)
        bleus.append(bleu)
    avg_bleu = sum(bleus) / len(bleus)

    return avg_bleu

def score_rouge(str1, str2):
    rouge = Rouge(metrics=["rouge-l"])
    try:
        scores = rouge.get_scores(str1, str2, avg=True)
        rouge_l = scores['rouge-l']['f']
        return rouge_l
    except:
        return 0


def caculate_rouge(results, df):
    rouges = []
    for qid, output in enumerate(results):
        prediction = extract_explanation(output)
        target = df["lecture"][qid] + " " + df["solution"][qid]
        target = target.strip()
        if prediction == "":
            continue
        if target == "":
            continue
        rouge = score_rouge(target, prediction)
        rouges.append(rouge)

    avg_rouge = sum(rouges) / len(rouges)
    return avg_rouge

def caculate_meteor(results, df):
    meteors = []
    for qid, output in enumerate(results):
        # prediction = extract_explanation(output)
        prediction = output
        target = df["lecture"][qid] + " " + df["solution"][qid]
        target = target.strip()
        if prediction == "":
            continue
        if target == "":
            continue
        meteor_score = meteor.compute(references=[target], predictions=[prediction])['meteor']
        meteors.append(meteor_score)
    avg_meteor = sum(meteors) / len(meteors)
    return avg_meteor


########################
## Sentence Similarity
########################
def similariry_score(str1, str2, model):
    # compute embedding for both lists
    embedding_1 = model.encode(str1, convert_to_tensor=True)
    embedding_2 = model.encode(str2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding_1, embedding_2).item()
    return score


def caculate_similariry(results, data, model):
    scores = []
    for qid, output in enumerate(results):
        prediction = extract_explanation(output)
        target = data["lecture"][qid] + " " + data["solution"][qid]
        target = target.strip()
        score = similariry_score(target, prediction, model)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    return avg_score

def calculate_metrics_solutions(results, data):
    ## BLEU
    bleu1 = caculate_bleu(results, data, gram=1)
    bleu4 = caculate_bleu(results, data, gram=4)
    print("BLEU-1: %.3f" % bleu1)
    print("BLEU-4: %.3f" % bleu4)

    ## Rouge-L
    rouge = caculate_rouge(results, data)
    print("ROUGE-L: %.3f" % rouge)

    meteor = caculate_meteor(results, data)
    print("METEOR: %.3f" % meteor)

    ## Similarity
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    similarity = caculate_similariry(results, data, model)
    print("Similarity: %.3f" % similarity)

    return {"BLEU-1": [bleu1], "BLEU-4": [bleu4], "ROUGE": [rouge], "METEOR": [meteor], "similarity": similarity}


