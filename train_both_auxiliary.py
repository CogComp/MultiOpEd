import csv
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from rouge_score import rouge_scorer
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.nn import CrossEntropyLoss, MSELoss
import os
from model import MultiTaskBart
from utils import load_csv


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def main():
    tokenizer_rel = BertTokenizer.from_pretrained("/shared/siyiliu/transformers/examples/seq2seq/Bert_Xander_finetuned")
    model_rel = BertForSequenceClassification.from_pretrained("/shared/siyiliu/transformers/examples/seq2seq/Bert_Xander_finetuned")
    
    tokenizer_stance = BertTokenizer.from_pretrained("/shared/siyiliu/transformers/examples/seq2seq/xander_stance_finetuned")
    model_stance = BertForSequenceClassification.from_pretrained("/shared/siyiliu/transformers/examples/seq2seq/xander_stance_finetuned")
    
    train_dataset, dev_dataset, test_dataset, label_to_query = load_csv('dataset.csv')
    for seed in (1,6,9):
        set_seed(seed)
    
        train_loader= DataLoader(train_dataset, shuffle=True)
        dev_loader= DataLoader(dev_dataset, shuffle=False)
        test_loader= DataLoader(test_dataset, shuffle=False)
        
        results_path_prefix = "results/seed_%d_stance=1+both" %seed
        models_path_prefix = "trained_models/seed_%d_stance=1+both" %seed
        if not os.path.exists(results_path_prefix):
            os.mkdir(results_path_prefix)
        if not os.path.exists(models_path_prefix):
            os.mkdir(models_path_prefix)
        for alpha in (1,5,50):
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir="/shared/siyiliu/transformers/examples/seq2seq/cached_models")
            model = MultiTaskBart.from_pretrained('facebook/bart-base', cache_dir="/shared/siyiliu/transformers/examples/seq2seq/cached_models")
            
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            model.to(device)
            model.train()
            optim = AdamW(model.parameters(), lr=3e-5)
            
            for epoch in range(6):
                i=0
                avg_loss1 = []
                avg_loss2 = []
                for batch in train_loader:
                    optim.zero_grad()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs=model(input_ids, attention_mask=attention_mask, labels = labels, return_dict=True)
                    loss2 =0
                    batch_label = tokenizer.decode(batch['labels'][0])
                    batch_label = batch_label[batch_label.find('<s>')+len('<s>'):batch_label.find('</s>')]
                    batch_query = label_to_query[batch_label]
                    
                    if alpha >0.0:
                        summa_last_hidden_state = outputs.encoder_last_hidden_state # this is actually decoder's last hidden state, look at model.py for details
                        eos_mask = labels.eq(model.config.eos_token_id)
                        sum_sentence_embedding = summa_last_hidden_state[eos_mask, :].view(summa_last_hidden_state.size(0), -1, summa_last_hidden_state.size(-1))[:, -1, :][0]
                        
                        batch_query_encoding = tokenizer.encode_plus(batch_query, return_tensors = "pt")
                        batch_query_encoding=batch_query_encoding.to(device)
                        outputs_query = model(**batch_query_encoding, return_dict=True)
                        query_last_hidden_state = outputs_query.encoder_last_hidden_state #decoder's last hidden state
                        query_sentence_embedding = query_last_hidden_state[0][-1]
                        
                        concat_embedding = torch.cat((sum_sentence_embedding, query_sentence_embedding), 0)
                        
                        logits_classification_rel = model.classification_head(concat_embedding)
                        logits_classification_stance = model.classification_head(concat_embedding)
                        
                        summary_ids = model.generate(input_ids)
                        generated_txt = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
                        
                        
                        token_rel = tokenizer_rel.encode_plus(batch_query, generated_txt, return_tensors="pt")
                        rel_logits = model_rel(**token_rel)[0]
                        loss_rel_prob = torch.sigmoid(rel_logits[0]).to(device)
                        
                        loss_fct = MSELoss()
                        loss2 = loss_fct(torch.sigmoid(logits_classification_rel.view(-1))[0], loss_rel_prob[0])
                        
                        token_stance = tokenizer_stance.encode_plus(batch_query, generated_txt, return_tensors="pt")
                        stance_logits = model_stance(**token_stance)[0]
                        loss_stance_prob = torch.sigmoid(stance_logits[0]).to(device)
                        
                        loss_fct = MSELoss()
                        loss3 = loss_fct(torch.sigmoid(logits_classification_stance.view(-1))[0], loss_stance_prob[0])
                        
                        
                        
                        
                        
                    
                    loss = outputs.loss + alpha*loss2 + 1*loss3
                    avg_loss1.append(float(outputs.loss))
                    avg_loss2.append(float(alpha*loss2))
                    
                    loss.backward(retain_graph=True)
                    optim.step()
                    
                    if i<3 and epoch ==0:
                        print()
                        print("--------Start--------")
                        print("Alpha = ", alpha)
                        print('Source:', tokenizer.decode(batch['input_ids'][0]))
                        print('Target:', batch_label)
                        print('Query:', batch_query)
                        print('LM Loss:', outputs[0], 'Auxiliary Loss:', loss2)
                        print()
                    i+=1
                    
                with open(results_path_prefix +'/alpha=%f_log.txt' %alpha, 'a') as file:
                    file.write('\n')
                    file.write("Epoch ")
                    file.write(str(epoch))
                    file.write('\n')
                    file.writelines(['Source: ', tokenizer.decode(batch['input_ids'][0]), '\n', 'Target: ', tokenizer.decode(batch['labels'][0]), '\n', 'Query: ', batch_query, '\n',"Avg loss1 = " + str(np.mean(avg_loss1)), "Avg loss2 = "+str(np.mean(avg_loss2)), "Avg total loss= " +str(np.mean(avg_loss1) + np.mean(avg_loss2))])
                    
            path_model =models_path_prefix + "/alpha=%f_models" %alpha
            model.save_pretrained(path_model)
            tokenizer.save_pretrained(path_model)
                    
            model.eval()
            
            generated_results = []
            generated_results_labels = []
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                summary_ids = model.generate(input_ids)
                generated_txt = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
                
                batch_label = tokenizer.decode(batch['labels'][0])
                batch_label = batch_label[batch_label.find('<s>')+len('<s>'):batch_label.find('</s>')]
                
                generated_results.append(generated_txt)
                generated_results_labels.append(batch_label)
                
                
                
            with open(results_path_prefix +'/alpha=%f_test_generated.txt' %alpha, 'w') as file:
                for txt in generated_results:
                    file.write(txt+'\n')
                    
            with open(results_path_prefix +'/alpha=%f_test_labels.txt' %alpha, 'w') as file:
                for txt in generated_results_labels:
                    file.write(txt+'\n')
                    
            
            generated_results = []
            generated_results_labels = []
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                summary_ids = model.generate(input_ids)
                generated_txt = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
                
                batch_label = tokenizer.decode(batch['labels'][0])
                batch_label = batch_label[batch_label.find('<s>')+len('<s>'):batch_label.find('</s>')]
                
                generated_results.append(generated_txt)
                generated_results_labels.append(batch_label)
                
                
                
            with open(results_path_prefix +'/alpha=%f_dev_generated.txt' %alpha, 'w') as file:
                for txt in generated_results:
                    file.write(txt+'\n')
                    
            with open(results_path_prefix +'/alpha=%f_dev_labels.txt' %alpha, 'w') as file:
                for txt in generated_results_labels:
                    file.write(txt+'\n')
                    
                    
if __name__ == "__main__":
    main()
