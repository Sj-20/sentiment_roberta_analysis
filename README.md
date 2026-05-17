# Live Demo

Get a hands-on experience with the app here: https://hossain-sentiment-roberta-analysis.streamlit.app/


# Public Sentiment Analysis with RoBERTa

This repository is a GitHub-safe version of a confidential customer sentiment analysis project.  
It keeps the same overall workflow while replacing real company/customer data with synthetic data.

## What this project does

- Loads customer interaction data from CSV or Excel.
- Uses the `translated text` column as the main input text field.
- Cleans text by removing bracketed system notes, URLs, emails, non-ASCII characters, punctuation, and extra spaces.
- Applies a pretrained RoBERTa sentiment model: `cardiffnlp/twitter-roberta-base-sentiment`.
- Produces sentiment labels: `Negative`, `Neutral`, and `Positive`.
- Allows batch prediction through a Streamlit app.

## Repository structure

```text
public_sentiment_roberta_project/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── synthetic_customer_feedback.csv
│   └── sample_upload_excel.xlsx
│
├── notebooks/
│   └── 01_colab_sentiment_pipeline.ipynb
│
└── src/
    ├── config.py
    ├── preprocessing.py
    ├── model.py
    └── batch_predict.py
```

## Dataset columns

The synthetic data keeps a practical customer-interaction structure:

| Column | Purpose |
|---|---|
| `Conversation ID` | Synthetic unique conversation identifier |
| `Conversation Participant type` | Example values: Agent, Customer, Traveler, Supplier |
| `Conversation Message Text` | Original-style message field |
| `translated text` | Main text column used by the model |
| `NPS score` | Synthetic NPS-style score |
| `Language` | Message language |
| `Devices used` | Channel/device source |
| `Inserted On` | Synthetic interaction date |
| `Expected_Sentiment_For_Demo` | Demo-only label for comparison; not required for prediction |

## How to run in Google Colab

1. Upload this project folder or ZIP to Google Drive.
2. Open `notebooks/01_colab_sentiment_pipeline.ipynb`.
3. Run all cells.
4. The notebook will:
   - install dependencies,
   - load `data/synthetic_customer_feedback.csv`,
   - clean the text,
   - run sentiment predictions,
   - save an output file.

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app.
4. Select:
   - Repository: your GitHub repo
   - Branch: `main`
   - Main file path: `app.py`
5. Deploy.

