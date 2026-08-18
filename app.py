import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ---------------------------------------------------------
# Page settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="Spam Email Classification",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Spam Email Classification")
st.write(
    "This application compares different machine learning "
    "classification models on the Spambase dataset."
)

st.info(
    "Upload the test_data.csv file, select a model, "
    "and click Evaluate Model to view its performance."
)

# ---------------------------------------------------------
# Load saved models
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(
            "model/logistic_regression.pkl"
        ),
        "Decision Tree": joblib.load(
            "model/decision_tree.pkl"
        ),
        "k-Nearest Neighbors (kNN)": joblib.load(
            "model/knn.pkl"
        ),
        "Gaussian Naive Bayes": joblib.load(
            "model/naive_bayes.pkl"
        ),
        "Random Forest": joblib.load(
            "model/random_forest.pkl"
        )
    }


try:
    models = load_models()

except Exception as e:
    st.error("Unable to load the trained models.")
    st.error(str(e))
    st.stop()


# ---------------------------------------------------------
# Dataset upload
# ---------------------------------------------------------
st.subheader("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("Test dataset uploaded successfully.")

    st.write("### Preview of Uploaded Dataset")
    st.dataframe(data.head())

    st.write(
        f"Number of test records: **{data.shape[0]}**"
    )

    st.write(
        f"Number of columns: **{data.shape[1]}**"
    )

    # -----------------------------------------------------
    # Check target column
    # -----------------------------------------------------
    if "spam" not in data.columns:

        st.error(
            "The uploaded CSV must contain the target "
            "column named 'spam'."
        )

        st.stop()

    # Separate features and actual labels
    X_test = data.drop(columns=["spam"])
    y_test = data["spam"]

    # -----------------------------------------------------
    # Model selection
    # -----------------------------------------------------
    st.subheader("2. Select Machine Learning Model")

    selected_model = st.selectbox(
        "Choose a classification model:",
        list(models.keys())
    )

    model = models[selected_model]

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------
    if st.button("Evaluate Model", type="primary"):

        try:

            y_pred = model.predict(X_test)

            y_prob = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            auc = roc_auc_score(
                y_test,
                y_prob
            )

            precision = precision_score(
                y_test,
                y_pred
            )

            recall = recall_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred
            )

            mcc = matthews_corrcoef(
                y_test,
                y_pred
            )

            # -------------------------------------------------
            # Display selected model
            # -------------------------------------------------
            st.subheader(
                f"Results - {selected_model}"
            )

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------
            st.write("### Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Accuracy",
                    f"{accuracy:.4f}"
                )

                st.metric(
                    "Precision",
                    f"{precision:.4f}"
                )

            with col2:
                st.metric(
                    "AUC Score",
                    f"{auc:.4f}"
                )

                st.metric(
                    "Recall",
                    f"{recall:.4f}"
                )

            with col3:
                st.metric(
                    "F1 Score",
                    f"{f1:.4f}"
                )

                st.metric(
                    "MCC",
                    f"{mcc:.4f}"
                )

            # -------------------------------------------------
            # Confusion Matrix
            # -------------------------------------------------
            st.write("### Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                y_pred
            )

            fig, ax = plt.subplots()

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=[
                    "Non-Spam",
                    "Spam"
                ]
            )

            disp.plot(
                ax=ax,
                cmap="Blues",
                colorbar=False
            )

            ax.set_title(
                f"Confusion Matrix - {selected_model}"
            )

            st.pyplot(fig)

            # -------------------------------------------------
            # Classification Report
            # -------------------------------------------------
            st.write("### Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                target_names=[
                    "Non-Spam",
                    "Spam"
                ],
                output_dict=True
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(
                report_df.round(4),
                use_container_width=True
            )

            # -------------------------------------------------
            # Short interpretation
            # -------------------------------------------------
            st.write("### Result Summary")

            st.write(
                f"The **{selected_model}** model achieved "
                f"an accuracy of **{accuracy * 100:.2f}%** "
                f"and an F1 score of **{f1:.4f}** on the "
                f"uploaded test dataset."
            )

        except Exception as e:

            st.error(
                "An error occurred while evaluating the model."
            )

            st.error(str(e))


else:

    st.warning(
        "Please upload test_data.csv to begin evaluation."
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")

st.caption(
    "Machine Learning Assignment 2 | "
    "Spambase Classification Project"
)