import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.config import TEXT_COLUMN
from src.preprocessing import prepare_dataframe
from src.model import load_model, predict_sentiment


def read_input_file(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError("Only CSV, XLSX, and XLS files are supported.")


def run_batch_prediction(input_path: str, output_path: str, text_column: str = TEXT_COLUMN) -> None:
    df = read_input_file(input_path)
    df = prepare_dataframe(df, text_column=text_column)

    tokenizer, model = load_model()

    predictions = []
    for text in tqdm(df["Translated_text_cleaned"], desc="Predicting sentiment"):
        predictions.append(predict_sentiment(text, tokenizer, model))

    pred_df = pd.DataFrame(predictions)
    output_df = pd.concat([df, pred_df], axis=1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_excel(output_path, index=False)

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch sentiment prediction")
    parser.add_argument("--input", required=True, help="Path to CSV or Excel input file")
    parser.add_argument("--output", default="outputs/sentiment_predictions.xlsx", help="Output file path")
    parser.add_argument("--text-column", default=TEXT_COLUMN, help="Text column to score")
    args = parser.parse_args()

    run_batch_prediction(args.input, args.output, args.text_column)
