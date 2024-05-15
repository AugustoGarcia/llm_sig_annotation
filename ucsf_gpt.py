import json
import requests
import pandas as pd
import os

API_VERSION = [API Version goes here]
RESOURCE_ENDPOINT = [Endpoint goes here]

def prompt_ucsf_openai(azure_api,deployment,prompt,model,max_tokens,temperature):
    url = [Query URL goes here]
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    })
    headers = {'Content-Type': 'application/json', 'api-key': azure_api}
    reply=''

    while reply=='':
        try:
            response = requests.post(url, headers=headers, data=body)
            print(response)
            reply=json.loads(response.text).get('choices')[0].get('message').get('content')
        except:
            pass

    return (reply)

def run_openai(azure_api,model,task_description,max_tokens,temperature,examples,n_samples,drug_file,criterion):
    
    df=pd.read_csv(drug_file)
    model_string=model+'_ucsf_'+str(n_samples)+'_shot_temperature_'+str(temperature)+'_'+criterion
    
    output_df=pd.DataFrame(columns=['text','y','pred'])
    
    output_df['text']=df['text']
    output_df['y']=df['output']
    
    output_file=os.path.join(os.getcwd(),'outputs', model_string+'.csv')

    deployment=model

    for idx in output_df.index:
        prompt=task_description+examples+'. Input: '+df.at[idx,'text']+'. Output: '
        output_df.at[idx,'pred']=prompt_ucsf_openai(azure_api,deployment,prompt,model,max_tokens,temperature)
        output_df.to_csv(output_file)
    
    return 0

