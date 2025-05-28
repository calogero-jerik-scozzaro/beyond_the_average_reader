import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def top_k_similar_readers(df_reader_embedding, reader, df_dataset,config):
    cosine_sim = cosine_similarity(df_reader_embedding["Embedding"].tolist())
    cosine_sim = pd.DataFrame(
            cosine_sim,
            columns=df_reader_embedding["Participant_ID"],
            index=df_reader_embedding["Participant_ID"],
        )
    reader_similarities = {}
    for participant_id in df_reader_embedding["Participant_ID"]:
        # Get top k+1 (includes self), then exclude self by taking [1:]
        top_similar = cosine_sim[participant_id].nlargest(config["k_similar_readers"]+1).iloc[1:]
        reader_similarities[participant_id] = {
            'similar_readers': top_similar.index.tolist(),
            'similarity_scores': top_similar.values.tolist()
        }      
  
    similar_readers = reader_similarities[reader]["similar_readers"]
    df_reader = df_dataset.loc[df_dataset["Participant_ID"] == reader]
    df_similar_readers = df_dataset.loc[df_dataset["Participant_ID"].isin(similar_readers)]
    
    return df_reader, df_similar_readers

def get_similar_from_avg_fd_nf(df_reader_embedding, reader, df_dataset):
    participant_ids = []
    ffd_averages = []
    nf_averages = []
    
    for i, row in df_reader_embedding.iterrows():
        embedding = row["Embedding"]
        participant_id = row["Participant_ID"]
        
        ffd_avg = sum(embedding[1::2]) / len(embedding[1::2])
        nf_avg = sum(embedding[0::2]) / len(embedding[0::2])
        
        participant_ids.append(participant_id)
        ffd_averages.append(ffd_avg)
        nf_averages.append(nf_avg)
    
    df_ffd_nf = pd.DataFrame({
        "Participant_ID": participant_ids,
        "FFD_Avg": ffd_averages,
        "NF_Avg": nf_averages
    })
    total_avg_ffd = df_ffd_nf["FFD_Avg"].mean()
    total_avg_nf = df_ffd_nf["NF_Avg"].mean()

    # now see if the reader has avg values bigger or smaller than the total avg for FFD and NF
    reader_ffd_avg = df_ffd_nf.loc[df_ffd_nf["Participant_ID"] == reader]["FFD_Avg"].values[0]
    reader_nf_avg = df_ffd_nf.loc[df_ffd_nf["Participant_ID"] == reader]["NF_Avg"].values[0]

    if reader_ffd_avg > total_avg_ffd and reader_nf_avg > total_avg_nf:
        similar_ids = df_ffd_nf.loc[
            (df_ffd_nf["FFD_Avg"] > total_avg_ffd) & (df_ffd_nf["NF_Avg"] > total_avg_nf),
            "Participant_ID"
        ].tolist()
    elif reader_ffd_avg < total_avg_ffd and reader_nf_avg < total_avg_nf:
        similar_ids = df_ffd_nf.loc[
            (df_ffd_nf["FFD_Avg"] < total_avg_ffd) & (df_ffd_nf["NF_Avg"] < total_avg_nf),
            "Participant_ID"
        ].tolist()
    elif reader_ffd_avg > total_avg_ffd and reader_nf_avg < total_avg_nf:
        similar_ids = df_ffd_nf.loc[
            (df_ffd_nf["FFD_Avg"] > total_avg_ffd) & (df_ffd_nf["NF_Avg"] < total_avg_nf),
            "Participant_ID"
        ].tolist()
    else:
        similar_ids = df_ffd_nf.loc[
            (df_ffd_nf["FFD_Avg"] < total_avg_ffd) & (df_ffd_nf["NF_Avg"] > total_avg_nf),
            "Participant_ID"
        ].tolist()

    df_reader = df_dataset.loc[df_dataset["Participant_ID"] == reader]
    df_similar_readers = df_dataset.loc[df_dataset["Participant_ID"].isin(similar_ids)]

    return df_reader, df_similar_readers

def get_similar_reader_above_threshold(df_reader_embedding, reader, df_dataset, threshold=0.6):
    cosine_sim = cosine_similarity(df_reader_embedding["Embedding"].tolist())
    cosine_sim = pd.DataFrame(
            cosine_sim,
            columns=df_reader_embedding["Participant_ID"],
            index=df_reader_embedding["Participant_ID"],
        )
    reader_similarities = {}
    for participant_id in df_reader_embedding["Participant_ID"]:
        # Filter readers above the threshold and not the same as the current reader
        similar_readers = cosine_sim.loc[participant_id]
        top_similar = similar_readers[(similar_readers > threshold) & (similar_readers.index != participant_id)]

        reader_similarities[participant_id] = {
            'similar_readers': top_similar.index.tolist(),
            'similarity_scores': top_similar.values.tolist()
        }

    similar_readers = reader_similarities[reader]["similar_readers"]
    df_reader = df_dataset.loc[df_dataset["Participant_ID"] == reader]
    df_similar_readers = df_dataset.loc[df_dataset["Participant_ID"].isin(similar_readers)]
    
    return df_reader, df_similar_readers