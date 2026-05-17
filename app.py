import io
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import TEXT_COLUMN
from src.preprocessing import prepare_dataframe, clean_translated_text
from src.model import load_model, predict_sentiment


st.set_page_config(
    page_title="Public Sentiment Analysis Demo",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Customer Sentiment Analysis Demo")
st.caption(
    "GitHub-safe demo using synthetic/customer-style data and a pretrained RoBERTa sentiment model."
)


@st.cache_resource(show_spinner=True)
def cached_model():
    return load_model()


def read_uploaded_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    return output.getvalue()


tokenizer, model = cached_model()

tab1, tab2 = st.tabs(["Single Text Prediction", "Batch File Prediction"])

with tab1:
    sample_text = st.text_area(
        "Enter customer message",
        value="The support was helpful and my issue was solved quickly.",
        height=120,
    )

    if st.button("Predict Sentiment", type="primary"):
        cleaned = clean_translated_text(sample_text)
        result = predict_sentiment(cleaned, tokenizer, model)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prediction", result["Predicted_Sentiment"])
        c2.metric("Confidence", f'{result["Confidence"]:.2%}')
        c3.metric("Positive", f'{result["Positive_Probability"]:.2%}')
        c4.metric("Negative", f'{result["Negative_Probability"]:.2%}')

        st.write("Cleaned text:")
        st.code(cleaned)

with tab2:
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help=f"File must contain a '{TEXT_COLUMN}' column.",
    )

    st.info(
        f"Expected text column: `{TEXT_COLUMN}`. "
        "You can use the included synthetic file in the `data` folder for testing."
    )

    if uploaded_file is not None:
        try:
            raw_df = read_uploaded_file(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(raw_df.head(20), use_container_width=True)

            prepared_df = prepare_dataframe(raw_df, text_column=TEXT_COLUMN)

            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Predicting sentiment..."):
                    predictions = [
                        predict_sentiment(text, tokenizer, model)
                        for text in prepared_df["Translated_text_cleaned"]
                    ]

                pred_df = pd.DataFrame(predictions)
                result_df = pd.concat([prepared_df, pred_df], axis=1)

                st.success(f"Prediction completed for {len(result_df):,} rows.")

                sentiment_counts = (
                    result_df["Predicted_Sentiment"]
                    .value_counts()
                    .reset_index()
                )
                sentiment_counts.columns = ["Predicted_Sentiment", "Count"]

                chart = px.bar(
                    sentiment_counts,
                    x="Predicted_Sentiment",
                    y="Count",
                    title="Predicted Sentiment Distribution",
                )
                st.plotly_chart(chart, use_container_width=True)

                st.subheader("Prediction Output")
                st.dataframe(result_df.head(100), use_container_width=True)

                st.download_button(
                    "Download predictions as Excel",
                    data=dataframe_to_excel_bytes(result_df),
                    file_name="sentiment_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                st.download_button(
                    "Download predictions as CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="sentiment_predictions.csv",
                    mime="text/csv",
                )

        except Exception as exc:
            st.error(f"Unable to process file: {exc}")
