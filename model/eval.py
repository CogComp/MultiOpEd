from transformers import BertTokenizer, BertForSequenceClassification
from rouge_score import rouge_scorer
from bert_score import score
from utils import load_csv
import numpy as np
import pandas as pd
import torch


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

    for seed in [1,6,9]:
        write_string = ""
        for alpha in [1,5,50]:
            with open('results/seed_%d_stance=1+both/alpha=%f_dev_generated.txt' %(seed,alpha), 'r') as file:
                generated_txt = [a.replace('\n','') for a in file.readlines()]
            with open('results/seed_%d_stance=1+both/alpha=%f_dev_labels.txt' %(seed,alpha), 'r') as file:
                target_txt = [a.replace('\n','') for a in file.readlines()]
            
            query_lst, target_txt = get_query_lst('dataset.csv', target_txt)
            stance_label = get_stance_label('dataset.csv', target_txt)
            
            write_string += "Alpha: "+ str(alpha) + '\n'
            
            rouge1, rouge2, rougeL, b_score, rel_score, stance_score = eval(generated_txt,target_txt, query_lst, stance_label, stance_path = "/shared/siyiliu/transformers/examples/seq2seq/xander_stance_finetuned", rel_path = "/shared/siyiliu/transformers/examples/seq2seq/Bert_Xander_finetuned", bertscore=True)
            
            write_string += "rouge1: " + str(rouge1) + " rouge2: " + str(rouge2) + " rougeL: " + str(rougeL) + " bscore: " + str(b_score) + " rel_score: " + str(rel_score) + " stance_score: " + str(stance_score) + "\n\n"
            print(alpha)
            print(rouge1, rouge2, rougeL, b_score, rel_score, stance_score)
            #print(rouge1_sum, rouge2_sum, rougeL_sum, rel_lst, stance_lst)
            
        #with open ('eval_results/seed_%d_stance=1_both_test.txt' %seed, 'w') as file:
        #    file.write(write_string)
