import numpy as np
from wordfreq import word_frequency
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class MyDatasetLSTM(Dataset):
    def __init__(self, df_data, df_metrics_value):
        self.data = self.process_data(df_data, df_metrics_value) 
    
    def __len__(self):
        return len(self.data['features'])

    def __getitem__(self, idx):
        sample = {}
        sample['features'] = self.data['features'][idx]
        sample['output'] = self.data['outputs'][idx]
        return sample
    
    def process_data(self, df_data_base, df_metrics_value):
        features_list, outputs = [], []
        df_data = df_data_base.copy()
        list_of_text_ids = df_data['Text_ID'].unique()
        for text_id in list_of_text_ids:
            text_data = df_data[df_data['Text_ID'] == text_id]
            output = np.full(64, -1, dtype=float) 
            text_features_list = [np.full(20, -1, dtype=float).tolist() for _ in range(64)]  # Fixed size 64 x 20 initialized to -1
            for word_unique_id in text_data['Word_Unique_ID']:
                if word_unique_id in df_metrics_value['Word_Unique_ID'].values:
                    features = np.full(20, -1, dtype=float)

                    word_info = text_data[text_data['Word_Unique_ID'] == word_unique_id].iloc[0]
                    word_len, word_freq, word_pos, word_surp = len(word_info['Word']), word_info['Word_In_Sentence_Number'], word_info['Frequency'], word_info['Surprisal']
                    word_number = word_info['Word_Number']

                    features[8:12] = [word_len, word_freq, word_pos, word_surp]

                    # Check for neighbors and update features accordingly
                    for offset, start_index in zip([-2, -1, 1, 2], [0, 4, 12, 16]):
                        neighbor_word = text_data[text_data['Word_Number'] == word_number + offset]
                        if not neighbor_word.empty:
                            neighbor_info = neighbor_word.iloc[0]
                            features[start_index:start_index + 4] = [
                                len(neighbor_info['Word']),
                                neighbor_info['Word_In_Sentence_Number'],
                                neighbor_info['Frequency'],
                                neighbor_info['Surprisal']
                            ]
                    
                    value = df_metrics_value[df_metrics_value['Word_Unique_ID'] == word_unique_id].iloc[0]['Average_Metric']
                    output[word_number] = round(value, 3)   
                    text_features_list[word_number] = features  # Update features list for this word
            features_list.append(text_features_list)
            outputs.append(output)

        # Convert lists to numpy arrays of float32
        features_list = np.array(features_list, dtype=np.float32)
        outputs = np.array(outputs, dtype=np.float32)

        # Return processed data
        data = {'features': features_list, 'outputs': outputs}
        return data
