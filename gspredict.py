import streamlit as st
import pandas as pd 
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from scipy.sparse import coo_matrix, csr_matrix

# the data
data_path = r"C:\Users\Owner\Downloads\mod_video_games.csv"
dataset = pd.read_csv(data_path)

# Load prefitted classification model
link = r"C:\Users\Owner\Downloads\model_classification.pkl"
with open(link, 'rb') as model_file:
        classifier = pickle.load(model_file)

# Load pre-fitted OneHotEncoder
enc_link = r"C:\Users\Owner\Downloads\encoder (1).pkl"
with open(enc_link, 'rb') as enc_file:
        enc = pickle.load(enc_file)

# Load prefitted regression model
reg_link = r"C:\Users\Owner\Downloads\regression_model.pkl"
with open(reg_link, 'rb') as reg_file:
        regressor = pickle.load(reg_file)

# Load prefitted Label Encoder
le_link = r"C:\Users\Owner\Downloads\label_encoder.pkl"
with open(le_link, 'rb') as label_file:
        le = pickle.load(label_file)


def predict_global_sales(Platform,Year_of_Release,Genre,Publisher,
                         NA_Sales,EU_Sales,JP_Sales,Other_Sales,
                         Critic_Score,Critic_Count,
                         User_Score,User_Count,Developer,Rating):
    """Predict game global sale!
    ---
    parameters:
      - name: Platform
        in: query
        type: alphanum
        required: true
      - name
        in: query
        type: 
      - name
      - name
    responses:
      200:
          description: The output value
    ---
    """
    
    #get features
    features_df = pd.DataFrame({"Platform":[Platform], "Year_of_Release":[Year_of_Release], "Genre":[Genre], 
                                    "Publisher":[Publisher],
                                       "NA_Sales":[NA_Sales], "EU_Sales":[EU_Sales], "JP_Sales":[JP_Sales], "Other_Sales":[Other_Sales],
                                       "Critic_Score":[Critic_Score], "Critic_Count":[Critic_Count],
                                       "User_Score":[User_Score], "User_Count":[User_Count], "Developer":[Developer], "Rating":[Rating]})
  
    # for classification model
    # Perform one-hot encoding on categorical features
    cat_cols = ["Platform", "Genre", "Publisher", "Developer", "Rating"]


    X_input_cat_encoded = enc.transform(features_df[cat_cols])
    X_input_cat_encoded = X_input_cat_encoded


    # Drop categorical columns from X_train and X_test
    X_input_numeric = features_df.drop(columns=cat_cols)


    X_input_encoded = hstack((X_input_numeric.values, X_input_cat_encoded))
    
    # Convert COO matrix to CSR format
    sparse_matrix_csr = csr_matrix(X_input_encoded)

    # manipulating the CSR matrix to drop columns
    columns_to_drop = [2335]  # Column i want to drop (the extra column from the encoder)

    # Drop specified columns
    sparse_matrix_csr = sparse_matrix_csr[:, [col for col in range(sparse_matrix_csr.shape[1]) if col not in columns_to_drop]]

    # convert back to COO format
    X_input_encoded = sparse_matrix_csr.tocoo() #X_input_encoded



    prediction = classifier.predict(X_input_encoded)

    # for regressor
    reg_features_df = pd.DataFrame({"Platform":[Platform], "Year_of_Release":[Year_of_Release], "Genre":[Genre], 
                                    "Publisher":[Publisher],
                                       "NA_Sales":[NA_Sales], "EU_Sales":[EU_Sales], "JP_Sales":[JP_Sales], "Other_Sales":[Other_Sales],
                                       "Critic_Score":[Critic_Score], "Critic_Count":[Critic_Count],
                                       "User_Score":[User_Score], "User_Count":[User_Count], "Developer":[Developer], "Rating":[Rating]})
  
    
    categorical_columns = ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']
    combined_categorical_data = reg_features_df[categorical_columns]

    # Fit LabelEncoder on combined categorical data
    le.fit(combined_categorical_data.values.ravel())

    # Now transform each individual column
    for column in categorical_columns:
          reg_features_df[column] = le.transform(reg_features_df[column])
    
    reg_prediction = regressor.predict(reg_features_df)

    #show prediction
    return(f"The sales level is: {prediction[0]}",
           f"Global sales: {reg_prediction[0].round(2)}")
    



def main():
    # Load the model
    


    st.title('Game Sales Level-Value Predictor')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Global Sales Level Predictor ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    Platform = st.selectbox("Platform", sorted(list(dataset.Platform.unique())))
    Year_of_Release = st.slider("Year_of_Release", 1980, 2020, step=1)
    Genre = st.selectbox("Genre", sorted(list(dataset.Genre.unique())))
    Publisher = st.selectbox("Publisher", sorted(list(dataset.Publisher.unique())))
    NA_Sales = st.number_input("North America Sales", min_value=0.0, step=1.,format="%.2f")
    EU_Sales = st.number_input("European Union Sales", min_value=0., step=1.,format="%.2f")
    JP_Sales = st.number_input("Japan Sales", min_value=0., step=1.,format="%.2f")
    Other_Sales = st.number_input("Other Sales", min_value=0., step=1.,format="%.2f")
    Critic_Score = st.number_input("Critic Score", min_value=0., step=1.,format="%.2f")
    Critic_Count = st.number_input("Critic Count", min_value=0., step=1.,format="%.2f")
    User_Score = st.number_input("User Score", min_value=0., step=1.,format="%.2f")
    User_Count = st.number_input("User Count", min_value=0., step=1.,format="%.2f")
    Developer = st.selectbox("Developer", sorted(list(dataset.Developer.unique())))
    Rating = st.selectbox("Select Rating", sorted(list(dataset.Rating.unique())))

   
    
    result = ""
    if st.button("Predict"):
        result=predict_global_sales(Platform,Year_of_Release,Genre,Publisher,
                                      NA_Sales,EU_Sales,JP_Sales,Other_Sales,
                                      Critic_Score,Critic_Count,
                                      User_Score,User_Count,Developer,Rating)
  
        st.success(result[0]) # 1st index from prediction returned.
        st.success(result[1]) # 2nd index from prediction returned.

if __name__=="__main__":
    main()
