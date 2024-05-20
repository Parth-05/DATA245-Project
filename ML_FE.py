import streamlit as st
import pandas as pd
import pickle
import base64

def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, data):
    return model.predict(data)

def get_base64_of_file(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

model = load_model(r"C:\Users\Vinay Bhati\Downloads\archive_data\bag_reg_lasso_model.pkl")

image_path = r"C:\Users\Vinay Bhati\Downloads\bk1.jpeg"
base64_image = get_base64_of_file(image_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_image}");
        background-size: cover;
        background-position: center center;
    }}
    /* Targeting the uploader text specifically */
    .stFileUploader .st-az, .stFileUploader .st-ay {{
        color: white !important;
    }}
    /* Universal text color settings */
    * {{
        color: white !important;
    }}
    /* Style dataframes to have semi-transparent black backgrounds with white text */
    .dataframe, .dataframe th, .dataframe td {{
        background-color: rgba(0,0,0,0.5) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Social Buzz Prediction')

with st.container():
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='1')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)
        if not df.empty:
            st.success("CSV File Uploaded Successfully!")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Here's a preview of the first few rows:")
                st.dataframe(df.head().style.highlight_max(axis=0))
            with col2:
                st.markdown("### Data to predict:")
                st.dataframe(df.iloc[0:1].style.set_properties(**{'background-color': 'black', 'color': 'lime'}))
                prediction = predict(model, df.iloc[0:1])
                st.metric(label="Prediction Result", value=f"{prediction[0]:,.2f}")
        else:
            st.error("The uploaded CSV file is empty.")
    else:
        st.info("Please upload a CSV file to proceed.")

# import matplotlib.pyplot as plt
# import numpy as np

# # Data
# models = ['LSTM (GloVe)', 'LSTM (Fasttext)', 'Transformer (GloVe)', 'Transformer (Fasttext)', 
#           'GRU (GloVe)', 'GRU (Fasttext)', 'RNN (GloVe)', 'RNN (Fasttext)']
# accuracy = [0.8189, 0.8202, 0.8203, 0.8187, 0.8198, 0.8142, 0.6240, 0.6249]
# precision = [0.7436, 0.7465, 0.7350, 0.7303, 0.7485, 0.7284, 0.4406, 0.3761]
# recall = [0.7768, 0.7757, 0.8015, 0.8059, 0.7698, 0.7911, 0.4303, 0.5331]
# f1_score = [0.7598, 0.7608, 0.7668, 0.7663, 0.7590, 0.7584, 0.4354, 0.5415]

# # Plotting
# x = np.arange(len(models))
# width = 0.2

# fig, ax = plt.subplots(figsize=(14, 8))

# rects1 = ax.bar(x - width, accuracy, width, label='Accuracy')
# rects2 = ax.bar(x, precision, width, label='Precision')
# rects3 = ax.bar(x + width, recall, width, label='Recall')
# rects4 = ax.bar(x + 2*width, f1_score, width, label='F1-score')

# # Adding labels, title and legend
# ax.set_xlabel('Models', fontsize=14)
# ax.set_ylabel('Scores', fontsize=14)
# ax.set_title('Comparison of Model Performance by Metric', fontsize=16)
# ax.set_xticks(x)
# ax.set_xticklabels(models, rotation=45, ha='right')
# ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')

# # Adding value labels on bars
# def add_labels(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate(f'{height:.3f}',
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# add_labels(rects1)
# add_labels(rects2)
# add_labels(rects3)
# add_labels(rects4)

# fig.tight_layout()
# plt.show()
