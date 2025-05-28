import numpy as np
from wordfreq import word_frequency
from torch.utils.data import Dataset
from transformers import AutoTokenizer




class MyDatasetBERT:
    def __init__(self, df_data, df_metrics_value, config):
        self.data = self.process_data(df_data, df_metrics_value, config)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        sample = {}
        sample['input_ids'] = self.data['input_ids'][idx]
        sample['word_ids'] = self.data['word_ids'][idx]
        sample['attention_mask'] = self.data['attention_mask'][idx]
        sample['output'] = self.data['output'][idx]
        return sample
    
    def process_data(self,df_data_base, df_metrics_value,config):
        input_ids_list, word_ids_list, attention_masks_list = [], [], []
        outputs = []
        df_data = df_data_base.copy()
        list_of_text_ids = df_data['Text_ID'].unique()
        for text_id in list_of_text_ids:
            text_data = df_data.loc[df_data['Text_ID'] == text_id]
            max_word_number = text_data['Word_Number'].max()
            
            # extract prefix from text_data where word_number is max_word_number
            last_row = text_data.loc[text_data['Word_Number'] == max_word_number]
            text = last_row['Prefix'].values[0] + " " + last_row['Word'].values[0]
            text_length = len(text.split())

            #tokenization and padding
            tokenizer = AutoTokenizer.from_pretrained(config["bert_model_name"])
            tokenizer.padding_side = 'right'
            text_str = '[CLS]' + ' ' + text + ' ' + '[SEP]'

            tokens = tokenizer.encode_plus(text_str.split(), 
                                        add_special_tokens = False,
                                        truncation=False,
                                        max_length = config['max_length'],
                                        padding = 'max_length',
                                        return_attention_mask=True,
                                        is_split_into_words=True)
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']

            word_ids = tokens.word_ids()
            word_ids = [val if val is not None else np.nan for val in word_ids]

            output = np.full(config["max_length"], -1,  dtype=float) 

            processed_positions = set()
            for word_unique_id in text_data['Word_Unique_ID']:
                if word_unique_id in df_metrics_value['Word_Unique_ID'].values:
                    value = df_metrics_value[df_metrics_value['Word_Unique_ID'] == word_unique_id].iloc[0]['Average_Metric']
                    position = text_data.loc[text_data['Word_Unique_ID'] == word_unique_id, 'Word_Number'].values[0]
                    
                    for idx, word_id in enumerate(word_ids):
                        if word_id == position and idx not in processed_positions:
                            output[idx] = round(value, 3)
                            processed_positions.add(idx)
                            break  # Assign only to the first subtoken
            
            input_ids_list.append(input_ids)
            word_ids_list.append(word_ids)
            attention_masks_list.append(attention_mask)
            outputs.append(output)
        
        input_ids_list = np.array(input_ids_list,dtype=np.int64)
        word_ids_list = np.array(word_ids_list)
        attention_masks_list = np.array(attention_masks_list,dtype=np.int64)
        outputs = np.array(outputs,dtype=np.float32)

        data = {'input_ids': input_ids_list, 
                'word_ids': word_ids_list, 
                'attention_mask': attention_masks_list, 
                'output': outputs}
        return data