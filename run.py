import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit

import ucsf_gpt
import example_sorting

ucsf_models=['gpt-35-turbo','gpt-4']
drugs=['Hydroxychloroquine 200 mg','Prednisone']
temperatures=[0.5]
max_tokens=10
few_shot_examples=[0,1,5,10,20,40,60,75,100]
example_criterions=['word_count','uncommon_words','model_disagreement','random']

shuffle_examples=False
example_shuffling_random_state=1

text_file=os.path.join(os.getcwd(),'data', 'azure_api.txt')
file = open(text_file, "r")
azure_api=file.read()

for drug in drugs:
    for temperature in temperatures:
        for criterion in example_criterions:

            # Task description for prompting, load dataset and select examples for few-shot learning
            task_description='The following text describes the dosing pattern of '+drug+'. The task is to extract the daily average dose in number of miligrams per day. Reply only with a number with one decimal value. Do not include an explanation.'
            drug_file=os.path.join(os.getcwd(),'data', drug+'.csv')
            df=pd.read_csv(drug_file)
            
            # Few-shot
            for n_samples in few_shot_examples:
                examples=''
                if n_samples != 0:

                    exec('sample_df=example_sorting.sort_by_'+criterion+'(df)')

                    examples=' Here are some examples of the task.'
                    
                    sample_df=sample_df.reset_index()

                    if (shuffle_examples):
                        sample_df=sample_df.sample(frac=1,random_state=example_shuffling_random_state)

                    examples_file=os.path.join(os.getcwd(),'examples', drug+'_'+str(n_samples)+'_'+criterion+'.csv')
                    sample_df.head(n_samples).to_csv(examples_file)

                    for idx in range(n_samples):
                        examples+='Input: '
                        examples+=sample_df.at[idx,'text']
                        examples+='. Output:'
                        examples+=str(sample_df.at[idx,'output'])
            
                    examples+='.'            
            
                if (n_samples==0 and criterion=='random') or n_samples!=0 : 

                    for model in ucsf_models:
                        print ('Running UCSF '+model+' with '+criterion+' and '+str(n_samples)+ ' shot')
                        ucsf_gpt.run_openai(azure_api,model,task_description,max_tokens,temperature,examples,n_samples,drug_file,criterion)

            # Fine-tuning
            n_splits=1
            test_size=0.25
        
            # Values with only one count have to be removed
            df_filtered=df.groupby('output').filter(lambda x : len(x)>1)
            df_filtered=df_filtered.reset_index(drop=True)
            y=df_filtered['output']
            X=range(len(y))

            sss=StratifiedShuffleSplit(n_splits=n_splits,test_size=test_size,random_state=42)
            for i, (train_index, test_index) in enumerate(sss.split(X, y)):
                train=df_filtered.loc[train_index]
                test=df_filtered.loc[test_index]

                train['prompt']=train['text']
                train['completion']=train['output']
                train=train[['prompt','completion']]

                train.to_csv('train.csv')

                if 'set_'+str(i) not in df.columns:
                    df['set_'+str(i)]='test'
                    df=df.set_index('text',drop=True)
                    train=train.set_index('prompt',drop=True)
                    for idx in train.index:
                        df.at[idx,'set_'+str(i)]='train'

                test.to_csv('test_set.csv')
            
            
