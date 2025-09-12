import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import pipeline

from datasets import Dataset


def get_first_n_words(text, n=1000):
    """Get the first n words of the text"""
    words = text.split()
    return " ".join(words[:n])


def preprocess_data(examples):
    sup_text = get_first_n_words(examples["text"], n=1000)
    clinical_note = f"""
Given the clinical note, if patient had a Transthoracic echocardiogram (TTE), then extract only the text related to the TTE or return nothing. Clinical Note: {sup_text}
"""
    return {"clinical_note": clinical_note}


def extract_echo_labels(
    input_csv,
    output_csv,
    model_name,
    text_column="text",
    discharge_column="discharge_text",
    batch_size=1,
    max_length=4000,
    min_length=30,
    device=None,
):
    """
    Extract echo labels from clinical notes using a summarization model.
    Args:
        input_csv (str): Path to input CSV file.
        output_csv (str): Path to output CSV file.
        model_name (str): Path or name of the summarization model.
        text_column (str): Name of the column containing text data.
        discharge_column (str): Name of the column for discharge text.
        batch_size (int): Batch size for DataLoader.
        max_length (int): Max length for generated summary.
        min_length (int): Min length for generated summary.
        device (int or str, optional): Device for model (e.g., CUDA device index).
    """
    df = pd.read_csv(input_csv)
    unique_texts = df[text_column].unique()
    dataset = Dataset.from_dict({"text": unique_texts})
    processed_dataset = dataset.map(preprocess_data, batched=False)
    dataloader = DataLoader(processed_dataset, batch_size=batch_size, shuffle=False)

    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else -1
    summarizer = pipeline("text-generation", model=model_name, device=device)

    dict_text = {}
    for batch in tqdm(dataloader):
        clinical_note = batch["clinical_note"][0]
        summary = summarizer(
            clinical_note, max_length=max_length, min_length=min_length, do_sample=True
        )
        generated_text = summary[0]["generated_text"]
        if generated_text.startswith(clinical_note):
            clean_summary = generated_text[len(clinical_note) :].strip()
        else:
            clean_summary = generated_text
        dict_text[batch["text"][0]] = clean_summary

    df["newtext"] = df[discharge_column].apply(lambda x: dict_text.get(x, ""))
    df.to_csv(output_csv, index=False)

    return df
