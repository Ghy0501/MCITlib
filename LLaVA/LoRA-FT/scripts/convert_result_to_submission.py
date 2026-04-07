import json
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, default='./LLaVA/results/CoIN_slim_new_0.8/OCRVQA/Finetune/merge.jsonl')
    parser.add_argument('--output_file', type=str, default='./LLaVA/results/CoIN_slim_new_0.8/OCRVQA/Finetune/our_result_for_submission.tsv')
    args=parser.parse_args()
    
    tsv_file = 'llava_v1.5_7b_MMT-Bench_ALL_openai_submission.tsv'
    tsv_data = pd.read_csv(tsv_file, sep='\t')

    jsonl_file = args.result_file
    
    
    
    jsonl_data = {}

    with open(jsonl_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            question_id = entry['question_id'].replace("ALL_", "")
            text = entry['text'].strip().upper()  
            jsonl_data[int(question_id)] = {"question_id": question_id, 'text': text}

 
    for index, row in tsv_data.iterrows():
        
        matched_entry = jsonl_data.get(index, None)
        
        if matched_entry:
            # print('match')

            tsv_data.at[index, 'prediction'] = row['prediction']  
            tsv_data.at[index, 'opt'] = matched_entry['text'] 
        else:
            tsv_data.at[index, 'prediction'] = "0"
            tsv_data.at[index, 'opt'] = 'Z'


    tsv_data.to_csv(args.output_file, sep='\t', index=False)

    print("TSV file updated successfully!")