from transformers import BertTokenizer, BertForSequenceClassification
from rouge_score import rouge_scorer
from bert_score import score
from utils import load_csv
import numpy as np
import pandas as pd
import torch
import os


def get_query_lst(dataset_path, target_txt):
    train_dataset, dev_dataset, test_dataset, label_to_query = load_csv(dataset_path)
    query_lst= []
    for i, txt in enumerate(target_txt):
        query_lst.append(label_to_query[txt])
        

    return query_lst, target_txt

def get_stance_label(dataset_path, target_txt):
    stance_label = []
    df = pd.read_csv(dataset_path)
    for txt in target_txt:
        #print(df.loc[df['replaced_text'] == txt+'\n'])
        #print(txt)
        row = df.loc[df['replaced_text'] == txt+'\n']['Support']
        #print(row['Support'])
        if row.empty:
            row = df.loc[df['replaced_text'] == txt]['Support']
            

        stance_label.append(int(row))
        
    assert all(v == 0 or v==1 for v in stance_label)
    #print(len(stance_label), len(target_txt))
    return stance_label

def eval(generated_txt, target_txt, query_lst, stance_label, stance_path = None, rel_path = None, bertscore=False):
    if stance_path!= None:
        stance_model = BertForSequenceClassification.from_pretrained(stance_path)
        stance_tokenizer = BertTokenizer.from_pretrained(stance_path)
    if rel_path != None:
        rel_model = BertForSequenceClassification.from_pretrained(rel_path)
        rel_tokenizer = BertTokenizer.from_pretrained(rel_path)
        
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2','rougeL'], use_stemmer=True)
    rouge1_sum = []
    rouge2_sum = []
    rougeL_sum = []
    rel_lst=[]
    stance_lst = []
    for i in range(len(generated_txt)):
        rouge_scores = scorer.score(generated_txt[i],target_txt[i])
        rouge1_sum.append(rouge_scores['rouge1'][2])
        rouge2_sum.append(rouge_scores['rouge2'][2])
        rougeL_sum.append(rouge_scores['rougeL'][2])
        
        
        rel = rel_tokenizer.encode_plus(query_lst[i], generated_txt[i], return_tensors="pt")
        rel_logits = rel_model(**rel)[0]
        loss = torch.softmax(rel_logits, dim=1).tolist()[0]
        if loss.index(max(loss)) ==0:
            rel_lst.append(1)
        else:
            rel_lst.append(0)
        
        stance = stance_tokenizer.encode_plus(query_lst[i], generated_txt[i], return_tensors="pt")
        stance_logits = stance_model(**stance)[0]
        loss = torch.softmax(stance_logits, dim=1).tolist()[0]
        if loss.index(max(loss)) != int(stance_label[i]):
            stance_lst.append(1)
        else:
            stance_lst.append(0)
        

    if bertscore==True:
        P, R, F1 = score(generated_txt, target_txt, lang='en')
        score_bert = F1.mean()
    
    
    
    return round(np.mean(rouge1_sum)*100, 2), round(np.mean(rouge2_sum)*100, 2), round(np.mean(rougeL_sum)*100,2), round(float(score_bert)*100,2), round(np.mean(rel_lst)*100,2), round(np.mean(stance_lst)*100,2)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Evaluating the results using the generated perspective")
    parser.add_argument('--dataset_path', type=str, help="Path to the MultiOpEd dataset file (in csv format)")
    parser.add_argument('--generated_file', type=str, help="Path to a newline-separated file containing the generated perspective")
    parser.add_argument('--labels_file', type=str, help="Path to a newline-separated file containing the ground truth perspective")
    parser.add_argument('--relevance_classifier_path', type=str, help="Path to the BERT relevance classifier")
    parser.add_argument('--stance_classifier_path', type=str, help="Path to the BERT stance classifier")
    parser.add_argument('--bert_score', type=bool, default=True, help="use bert score or not")
    parser.add_argument('--result_path', type=str, help="the directory to save the results.txt")
    
    args = parser.parse_args()
    
    

    write_string = ""
    with open(args.generated_file, 'r') as file:
        generated_txt = [a.replace('\n','') for a in file.readlines()]
    with open(args.labels_file, 'r') as file:
        target_txt = [a.replace('\n','') for a in file.readlines()]
            
    query_lst, target_txt = get_query_lst(args.dataset_path, target_txt)
    stance_label = get_stance_label(args.dataset_path, target_txt)
            
            
    rouge1, rouge2, rougeL, b_score, rel_score, stance_score = eval(generated_txt,target_txt, query_lst, stance_label, stance_path = args.stance_classifier_path, rel_path = args.relevance_classifier_path, bertscore=args.bert_score)
            
    write_string += "rouge1: " + str(rouge1) + " rouge2: " + str(rouge2) + " rougeL: " + str(rougeL) + " bscore: " + str(b_score) + " rel_score: " + str(rel_score) + " stance_score: " + str(stance_score) + "\n"
    
    print(rouge1, rouge2, rougeL, b_score, rel_score, stance_score)
    #print(rouge1_sum, rouge2_sum, rougeL_sum, rel_lst, stance_lst)
            
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    with open (args.result_path+"/results.txt", 'a') as file:
        file.write(write_string)
