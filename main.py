import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import torch
import torch.nn as nn
import argparse
import time
import random
from utils import *
from profilation import *
from models import *
from my_datasets.my_dataset_base import MyDatasetBase
from my_datasets.my_dataset_lstm import MyDatasetLSTM
from my_datasets.my_dataset_mlp import MyDatasetMLP
from my_datasets.my_dataset_lstm_mlp import MyDatasetLSTMMLP
from my_datasets.my_dataset_bert import MyDatasetBERT
from my_data_loaders.my_data_loader import MyDataLoader
from models.model_lstm import MyLSTM
from models.model_mlp import MyMLP
from models.model_lstm_mlp import MyLSTMMLP
from models.model_bert_layer import BertWithLayer
from train.trainer_lstm import TrainerLSTM
from train.trainer_mlp import TrainerMLP
from train.trainer_lstm_mlp import TrainerLSTMMLP
from train.trainer_bert import TrainerBERT
from test.tester_lstm import TesterLSTM
from test.tester_mlp import TesterMLP
from test.tester_lstm_mlp import TesterLSTMMLP
from test.tester_bert import TesterBERT


def main(args):
    RANDOM_STATE = 42
    config = {}
    config["profilation"] = args.profilation
    config["model_name"] = args.model_name
    config["num_texts_embedding"] = args.num_texts_embedding
    config["metric_name"] = "IA_FIRST_FIXATION_DURATION"
    config["k_similar_readers"] = 20
    
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device

    print(f"Configuration: {config}\n")

    df_dataset_text_properties = pd.read_csv("data/dataset_text_properties.csv")
    dataset = pd.read_csv("data/Provo_Corpus-Eyetracking_Data.csv")
    df = pd.DataFrame(
        dataset,
        columns=[
            "Participant_ID",
            "Word_Unique_ID",
            "Text_ID",
            "Sentence_Number",
            "Word_In_Sentence_Number",
            "Word_Cleaned",
            "Word_Length",
            "Word_Content_Or_Function",
            "Word_POS",
            "IA_FIRST_FIXATION_DURATION",
            "IA_DWELL_TIME",
            "IA_FIXATION_COUNT",
        ],
    )

    # modify the values the metrics columns of to be 0 if NA
    df["IA_FIRST_FIXATION_DURATION"] = df["IA_FIRST_FIXATION_DURATION"].fillna(0)
    df["IA_FIXATION_COUNT"] = df["IA_FIXATION_COUNT"].fillna(0)
    # filter out the rows with Word_Unique_ID null
    df = df.dropna(subset=["Word_Unique_ID"])

    text_ids_embedding = np.random.choice(df["Text_ID"].unique(), config["num_texts_embedding"], replace=False)
    df_dataset_base = df.loc[~df["Text_ID"].isin(text_ids_embedding)]
    if (text_ids_embedding.size > 0):
        df_embedding = df.loc[df["Text_ID"].isin(text_ids_embedding)].copy()
        df_embedding["IA_FIRST_FIXATION_DURATION_NORM"] = df["IA_FIRST_FIXATION_DURATION"] / df["IA_FIRST_FIXATION_DURATION"].max()
        df_embedding["IA_FIXATION_COUNT_NORM"] = df["IA_FIXATION_COUNT"] / df["IA_FIXATION_COUNT"].max()

        df_reader_embedding = (
            df_embedding.groupby("Participant_ID")[
                ["IA_FIRST_FIXATION_DURATION_NORM", "IA_FIXATION_COUNT_NORM"]
            ]
            .apply(lambda x: x.to_numpy().flatten().tolist())
            .reset_index(name="Embedding")
        )

    participants = df_dataset_base["Participant_ID"].unique()
    list_mean_mse, list_mean_accL, list_mean_accT, list_mean_accS = [], [], [], []
    list_mean_mse_reader, list_mean_accL_reader, list_mean_accT_reader, list_mean_accS_reader = [], [], [], []

    time_start = time.time()
    for reader in participants:
                
        if (config["profilation"] == "top_k_similar_readers"):
            df_reader, df_dataset = top_k_similar_readers(df_reader_embedding, reader, df_dataset_base,config)
        elif (config["profilation"] == "4_classes_ffd_nf"):
            df_reader, df_dataset = get_similar_from_avg_fd_nf(df_reader_embedding, reader, df_dataset_base)
        elif(config["profilation"] == "above_threshold"):
            df_reader, df_dataset = get_similar_reader_above_threshold(df_reader_embedding, reader, df_dataset_base)
        elif (config["profilation"] == "all_readers"):
            df_reader = df_dataset_base.loc[df_dataset_base["Participant_ID"] == reader]
            df_dataset = df_dataset_base.loc[df_dataset_base["Participant_ID"] != reader]
        
        df_reader_values = df_reader[["Word_Unique_ID", config['metric_name']]].rename(columns={config['metric_name']: "Average_Metric"})

        list_mse, list_accL, list_accT, list_accS = [], [], [], []
        list_mse_reader, list_accL_reader, list_accT_reader, list_accS_reader = [], [], [], []

        text_ids = list(df_dataset_text_properties["Text_ID"].unique())
        text_ids = [tid for tid in text_ids if tid not in text_ids_embedding]
        text_ids = np.array(text_ids)

        # 5 runs for each reader with different random seeds
        for i in range(0,5):
            torch.manual_seed(RANDOM_STATE)
            np.random.seed(RANDOM_STATE)
            random.seed(RANDOM_STATE)
            
            # Shuffle and split 90% train, 5% validation, 5% test
            train_text_ids, temp_text_ids = train_test_split(text_ids, test_size=0.10, random_state=RANDOM_STATE+i, shuffle=True)
            val_text_ids, test_text_ids = train_test_split(temp_text_ids, test_size=0.50, random_state=RANDOM_STATE+i, shuffle=True)  # Split the remaining 10% into 5% val and 5% test

            df_train = df_dataset_text_properties.loc[df_dataset_text_properties["Text_ID"].isin(train_text_ids)]
            df_val = df_dataset_text_properties.loc[df_dataset_text_properties["Text_ID"].isin(val_text_ids)]
            df_test = df_dataset_text_properties.loc[df_dataset_text_properties["Text_ID"].isin(test_text_ids)]
                        
            df_dataset_avg = df_dataset.groupby(["Word_Unique_ID"]).agg(Average_Metric=(config['metric_name'], "mean")).reset_index()
            df_train_word_ids = df_train["Word_Unique_ID"].unique()
            df_dataset_train_avg = df_dataset[df_dataset["Word_Unique_ID"].isin(df_train_word_ids)].groupby(["Word_Unique_ID"]).agg(
                Average_Metric=(config['metric_name'], "mean"),
            ).reset_index()
            max_training_value = df_dataset_train_avg["Average_Metric"].max()
            df_dataset_avg["Average_Metric"] = df_dataset_avg["Average_Metric"] / max_training_value
            
            df_reader_values_fold = df_reader_values.copy()
            df_reader_values_fold["Average_Metric"] = df_reader_values_fold["Average_Metric"] / max_training_value
            
            
            if config["model_name"] == "LGBM":
                train_dataset = MyDatasetBase(df_train, df_dataset_avg)
                test_dataset = MyDatasetBase(df_test, df_dataset_avg)
                test_dataset_reader = MyDatasetBase(df_test, df_reader_values_fold)
                train_features, train_outputs = train_dataset.get_features_and_outputs()
                test_features, test_outputs = test_dataset.get_features_and_outputs()
                _, test_outputs_reader = test_dataset_reader.get_features_and_outputs()

                lgb_train_dataset = lgb.Dataset(train_features, label=train_outputs)
                params = {
                    "objective": "regression",
                    "metric": "rmse",
                    "verbose": -1,
                    "random_state": RANDOM_STATE
                }

                booster = lgb.train(params, lgb_train_dataset, num_boost_round=100)
                predictions = booster.predict(test_features)
                predictions = predictions * max_training_value
                outputs = test_outputs * max_training_value
                outputs_reader = test_outputs_reader * max_training_value
                
            elif (config["model_name"] == "MLP"):
                config['batch_size'] = 8
                config["epochs"] = 1000
                config["ft_epochs"] = 100
                train_dataset = MyDatasetMLP(df_train, df_dataset_avg)
                val_dataset = MyDatasetMLP(df_val, df_dataset_avg)
                test_dataset = MyDatasetMLP(df_test, df_dataset_avg)
                test_dataset_reader = MyDatasetMLP(df_test, df_reader_values_fold)

                train_loader = MyDataLoader(train_dataset, config)
                val_loader = MyDataLoader(val_dataset, config)
                test_loader = MyDataLoader(test_dataset, config, shuffle=False)
                test_loader_reader = MyDataLoader(test_dataset_reader, config, shuffle=False)
                
                model = MyMLP()
                trainer = TrainerMLP(model, train_loader, val_loader, config)
                model = trainer.train()
                tester = TesterMLP(model,test_loader, config)
                predictions, outputs_tensor = tester.evaluate()
                tester_reader = TesterMLP(model, test_loader_reader, config)
                _, outputs_reader = tester_reader.evaluate()
                predictions, outputs, outputs_reader = scale_predictions_output(predictions, outputs_tensor, outputs_reader, max_training_value)

            elif (config["model_name"] == "LSTM"):
                config['batch_size'] = 4
                config["epochs"] = 3000
                config["ft_epochs"] = 100
                train_dataset = MyDatasetLSTM(df_train, df_dataset_avg)
                val_dataset = MyDatasetLSTM(df_val, df_dataset_avg)
                test_dataset = MyDatasetLSTM(df_test, df_dataset_avg)
                test_dataset_reader = MyDatasetLSTM(df_test, df_reader_values_fold)
                
                train_loader = MyDataLoader(train_dataset, config)
                val_loader = MyDataLoader(val_dataset, config)
                test_loader = MyDataLoader(test_dataset, config, shuffle=False)
                test_loader_reader = MyDataLoader(test_dataset_reader, config, shuffle=False)

                model = MyLSTM()
                trainer =  TrainerLSTM(model, train_loader, val_loader, config)
                model = trainer.train()
                tester = TesterLSTM(model,test_loader, config)
                predictions, outputs_tensor = tester.evaluate()
                tester_reader = TesterLSTM(model, test_loader_reader, config)
                _, outputs_reader = tester_reader.evaluate()
                predictions, outputs, outputs_reader = scale_predictions_output(predictions, outputs_tensor, outputs_reader, max_training_value)
                                
            elif (config["model_name"] == "LSTM_MLP"):
                config['batch_size'] = 4
                config["epochs"] = 3000
                config["ft_epochs"] = 300
                train_dataset = MyDatasetLSTMMLP(df_train, df_dataset_avg)
                val_dataset = MyDatasetLSTMMLP(df_val, df_dataset_avg)
                test_dataset = MyDatasetLSTMMLP(df_test, df_dataset_avg)
                test_dataset_reader = MyDatasetLSTMMLP(df_test, df_reader_values_fold)
    
                train_loader = MyDataLoader(train_dataset, config)
                val_loader = MyDataLoader(val_dataset, config)
                test_loader = MyDataLoader(test_dataset, config, shuffle=False)
                test_loader_reader = MyDataLoader(test_dataset_reader, config, shuffle=False)

                model = MyLSTMMLP()
                trainer = TrainerLSTMMLP(model, train_loader, val_loader, config)
                model = trainer.train()
                tester = TesterLSTMMLP(model, test_loader, config)
                predictions, outputs_tensor = tester.evaluate()
                tester_reader = TesterLSTMMLP(model, test_loader_reader, config)
                _, outputs_reader = tester_reader.evaluate()
                predictions, outputs, outputs_reader = scale_predictions_output(predictions, outputs_tensor, outputs_reader, max_training_value)

            elif (config["model_name"] == "BertWithLayer" or config["model_name"] == "BertWithLayerFineTuning"):
                config["bert_model_name"] = 'bert-base-cased'
                config["max_length"] = 192
                config["batch_size"] = 16
                config["epochs"] = 100
                config["fine_tuning"] = True if config["model_name"] == "BertWithLayerFineTuning" else False

                train_dataset = MyDatasetBERT(df_train, df_dataset_avg, config)
                val_dataset = MyDatasetBERT(df_val, df_dataset_avg, config)
                test_dataset = MyDatasetBERT(df_test, df_dataset_avg, config)
                test_dataset_reader = MyDatasetBERT(df_test, df_reader_values_fold, config)

                train_loader = MyDataLoader(train_dataset, config)
                val_loader = MyDataLoader(val_dataset, config)
                test_loader = MyDataLoader(test_dataset, config, shuffle=False)
                test_loader_reader = MyDataLoader(test_dataset_reader, config, shuffle=False)

                model = BertWithLayer(config)
                trainer = TrainerBERT(model,train_loader,val_loader,config)
                model = trainer.train()
                tester = TesterBERT(model, test_loader, config)
                predictions, outputs_tensor = tester.evaluate()
                tester_reader = TesterBERT(model, test_loader_reader, config)
                _, outputs_reader = tester_reader.evaluate()
                predictions, outputs, outputs_reader = scale_predictions_output(predictions, outputs_tensor, outputs_reader, max_training_value)


            predictions = [p if p >= 0 else 0 for p in predictions]

            mse_value, accL_value, accT_value, accS_value = calculate_metrics(outputs, predictions, max_training_value)
            mse_value_reader, accL_value_reader, accT_value_reader, accS_value_reader = calculate_metrics(outputs_reader, predictions, max_training_value)
 
            list_mse.append(mse_value)
            list_accL.append(accL_value)
            list_accT.append(accT_value)
            list_accS.append(accS_value)
            list_mse_reader.append(mse_value_reader)
            list_accL_reader.append(accL_value_reader)
            list_accT_reader.append(accT_value_reader)
            list_accS_reader.append(accS_value_reader)
            
            
        print(f"Results for reader {reader}:")
        print(f"Results for datasets' values:")
        print(f"Mean accL: {round(np.mean(list_accL)*100,2)}%") 
        print(f"Mean accT: {round(np.mean(list_accT)*100,2)}%")
        print(f"Mean accS: {round(np.mean(list_accS)*100,2)}%")
        print(f"Results for reader's values:")
        print(f"Mean accL: {round(np.mean(list_accL_reader)*100,2)}%")
        print(f"Mean accT: {round(np.mean(list_accT_reader)*100,2)}%")
        print(f"Mean accS: {round(np.mean(list_accS_reader)*100,2)}%")
        print("\n")

        list_mean_mse.append(np.mean(list_mse))
        list_mean_accL.append(np.mean(list_accL))
        list_mean_accT.append(np.mean(list_accT))
        list_mean_accS.append(np.mean(list_accS))
        list_mean_mse_reader.append(np.mean(list_mse_reader))
        list_mean_accL_reader.append(np.mean(list_accL_reader))
        list_mean_accT_reader.append(np.mean(list_accT_reader))
        list_mean_accS_reader.append(np.mean(list_accS_reader))


    time_end = time.time()
    file_name = (
        config["profilation"]
        + "-"
        + config["model_name"]
        + "-"
        + str(config["num_texts_embedding"])
        + ".txt"
    )
    with open(f'results/{file_name}', "w") as f:
        f.write(f"Configuration:\n")
        f.write(f"Profilation: {config['profilation']}\n")
        f.write(f"Model: {config['model_name']}\n")
        f.write(f"Number of texts for embedding: {config['num_texts_embedding']}\n")
        f.write("---------------------------------\n")
        f.write("Final results for dataset:\n")
        f.write(f"Mean accL: {round(np.mean(list_mean_accL)*100,2)}% ({round(np.std(list_mean_accL)*100,2)})%\n")
        f.write(f"Mean accT: {round(np.mean(list_mean_accT)*100,2)}% ({round(np.std(list_mean_accT)*100,2)})%\n")
        f.write(f"Mean accS: {round(np.mean(list_mean_accS)*100,2)}% ({round(np.std(list_mean_accS)*100,2)})%\n")
        f.write("\n")
        f.write("Final results for reader:\n")
        f.write(f"Mean accL: {round(np.mean(list_mean_accL_reader)*100,2)}% ({round(np.std(list_mean_accL_reader)*100,2)})%\n")
        f.write(f"Mean accT: {round(np.mean(list_mean_accT_reader)*100,2)}% ({round(np.std(list_mean_accT_reader)*100,2)})%\n")
        f.write(f"Mean accS: {round(np.mean(list_mean_accS_reader)*100,2)}% ({round(np.std(list_mean_accS_reader)*100,2)})%\n")
        f.write("---------------------------------\n")
        f.write(f"Time elapsed: {round((time_end - time_start)/60,2)} minutes\n")
        f.write(f"Time: {time.asctime()}\n")
        f.write("------------------------------------\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--profilation", help="Profilation type: all_readers, top_k_similar_readers, 4_classes_ffd_nf, above_threshold", type=str, default="all_readers")
    parser.add_argument("-m", "--model_name", help="Model name: LGBM, LSTM, MLP, LSTM_MLP,  BertWithLayer, BertWithLayerFineTuning,", type=str, default="LSTM")
    parser.add_argument("-n", "--num_texts_embedding", help="Number of texts for the embedding creation", type=int, default=10)
    args = parser.parse_args()
    main(args)
