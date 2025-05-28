# Beyond the Average Reader: the Reader Embedding Approach

This project introduces a novel approach to reading time prediction using reader embeddings. Traditional approaches typically average eye-tracking data across all readers, which fails to account for individual variability and specific reading behaviors. Our method creates embeddings to compactly describe individual reading fixation patterns, enabling more personalized predictions by identifying and leveraging data from readers with similar behaviors.

## Research Focus

Our work demonstrates that systems trained on average values across all readers are ill-suited for predicting reading times for specific subjects. We overcome this limitation by:
1. Creating embeddings that characterize individual reading behavior
2. Identifying similar readers based on these embeddings
3. Training models on values averaged only over subsets of similar readers

## Setup and Execution

1. **Data Preparation**: Download the Provo Corpus files (Luke, S.G., Christianson, K. The Provo Corpus: A large eye-tracking corpus with predictability norms. Behav Res 50, 826–833 (2018). https://doi.org/10.3758/s13428-017-0908-4) and place them in a data folder in the project root directory.
2. **Environment Setup**: Create a conda environment using the provided requirements.yaml file and activate it. Note that PyTorch and related packages are installed separately.
    ```bash
    conda env create -f requirements.yml
    conda activate beyond_the_average_reader_env
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
3. **Dataset Creation**: Execute the creation_dataset_freq_and_surprisal.ipynb notebook to correctly format the dataset. This will create a 'dataset_text_properties.csv' file in the 'data' folder.
4. **Model Training**: Run main.py with appropriate parameters.

## Parameter Options

- **Profilation Types** (`-p`, `--profilation`):
    - `all_readers`: Uses data from all readers
    - `top_k_similar_readers`: Selects the top k most similar readers for each subject based on cosine similarity of reader embeddings
    - `4_classes_ffd_nf`: Groups readers into four classes based on their mean FFD and NF values 
    - `above_threshold`: Associates each reader i only with readers j whose cosine similarity exceeds a predetermined threshold

- **Models** (`-m`, `--model_name`):
    - `LGBM`
    - `LSTM`
    - `MLP`
    - `LSTM_MLP`
    - `BertWithLayer`
    - `BertWithLayerFineTuning`

- **Embedding Settings** (`-n`, `--num_texts_embedding`): 
    - Controls the number of texts used for embedding creation (default: 10)

See example_script.sh for usage examples.