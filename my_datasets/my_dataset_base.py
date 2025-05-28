import numpy as np
from wordfreq import word_frequency
from torch.utils.data import Dataset


def process_data(df_data, df_metrics_value):
    words, previous_words = [], []
    word_lengths, previous_word_lengths = [], []
    word_freqs, previous_word_freqs = [], []
    word_positions, previous_word_positions = [], []
    surprisals = []
    outputs = []
    data = {}

    for _, row in df_data.iterrows():
        if row['Word_In_Sentence_Number'] != 1 or row['Sentence_Number'] != 1:
            word = row['Word']
            previous_word = row['Prefix'].split()[-1]
            word_length = len(word)
            previous_word_length = len(previous_word)
            word_freq = row['Frequency']
            previous_word_freq = word_frequency(previous_word.lower(), 'en')
            word_position = row['Word_In_Sentence_Number']
            previous_word_position = word_position - 1 
            surprisal = row['Surprisal']
            output = df_metrics_value.loc[df_metrics_value['Word_Unique_ID'] == row['Word_Unique_ID'], 'Average_Metric']
            if not output.empty:
                words.append(word)
                previous_words.append(previous_word)
                word_lengths.append(word_length)
                previous_word_lengths.append(previous_word_length)
                word_freqs.append(word_freq)
                previous_word_freqs.append(previous_word_freq)
                word_positions.append(word_position)
                previous_word_positions.append(previous_word_position)
                surprisals.append(surprisal)
                outputs.append(round(output.values[0],3))

    data = {'words': words, 
            'previous_words': previous_words, 
            'word_lengths': word_lengths, 
            'previous_word_lengths': previous_word_lengths, 
            'word_freqs': word_freqs, 
            'previous_word_freqs': previous_word_freqs, 
            'word_positions': word_positions,
            'previous_word_positions': previous_word_positions, 
            'surprisals': surprisals, 
            'outputs': outputs}
    return data

class MyDatasetBase(Dataset):
    def __init__(self, df_data, df_text_properties):
        self.data = process_data(df_data, df_text_properties)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {}
        sample['word'] = self.data['words'][idx]
        sample['previous_word'] = self.data['previous_words'][idx]
        sample['word_length'] = self.data['word_lengths'][idx]
        sample['previous_word_length'] = self.data['previous_word_lengths'][idx]
        sample['word_freq'] = self.data['word_freqs'][idx]
        sample['previous_word_freq'] = self.data['previous_word_freqs'][idx]
        sample['word_position'] = self.data['word_positions'][idx]
        sample['previous_word_position'] = self.data['previous_word_positions'][idx]
        sample['surprisal'] = self.data['surprisals'][idx]
        sample['output'] = self.data['outputs'][idx]
        return sample
  
    def get_features_and_outputs(self):
        features = np.array([
        [
            self.data['word_lengths'][i],
            self.data['previous_word_lengths'][i],
            self.data['word_freqs'][i],
            self.data['previous_word_freqs'][i],
            self.data['word_positions'][i],
            self.data['previous_word_positions'][i],
            self.data['surprisals'][i],
        ] for i in range(len(self.data['outputs']))
        ])
        outputs = np.array(self.data['outputs'])
        return features, outputs
      