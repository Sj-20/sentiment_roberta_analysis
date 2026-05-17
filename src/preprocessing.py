import re
import pandas as pd


def remove_bracketed_text(text: str) -> str:
    """Remove text inside square brackets, e.g. [system note]."""
    return re.sub(r"\[[^\[\]]*\]", "", str(text)).strip()


def clean_translated_text(text: str) -> str:
    """
    Clean translated customer text while keeping the logic close to the original project:
    - convert to string
    - remove bracketed system text
    - remove non-ASCII characters
    - remove URLs
    - remove email addresses
    - remove punctuation/special characters
    - normalize whitespace
    """
    text = str(text)
    text = remove_bracketed_text(text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Prepare a dataframe for prediction.
    Rows with blank text are removed, but the original columns are preserved.
    """
    if text_column not in df.columns:
        raise ValueError(
            f"Required text column '{text_column}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    output = df.copy()
    output = output[output[text_column].notna()]
    output = output[output[text_column].apply(lambda x: isinstance(x, str))]
    output[text_column] = output[text_column].astype(str)

    blocked_values = {"none", "None", "Blank", "blank", "no", ""}
    output = output[~output[text_column].str.strip().isin(blocked_values)]

    output["Translated_text_cleaned"] = output[text_column].apply(clean_translated_text)
    output = output[output["Translated_text_cleaned"].str.len() > 0]
    output = output.reset_index(drop=True)
    return output
