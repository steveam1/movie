
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv('/Users/ashleystevens/Desktop/movies.csv')
    
    df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype(float)
    
    genres = df['genres'].str.get_dummies(sep='|')
    df = pd.concat([df, genres], axis=1)
    return df

def knn_from_scratch(input_movie, df, k=4):
    feature_cols = ['year'] + list(df.columns[4:]) 
    features = df[feature_cols]
    
    try:
        input_idx = df[df['title'].str.contains(input_movie, case=False, na=False)].index[0]
        input_features = features.iloc[input_idx]
        
        distances = features.apply(lambda row: np.linalg.norm(row - input_features), axis=1)
        
        top_k_indices = distances.nsmallest(k + 1).iloc[1:].index  
        
        recommended_titles = df.iloc[top_k_indices]['title'].values
        return recommended_titles
    except IndexError:
        return ["Movie not found. Please try again."]

def main():
    st.title("Movie Recommendation System")
    st.write("Find movies similar to the one you love!")
    
    df = load_data()
    
    user_input = st.text_input("Enter a movie you like:")
    k = st.slider("Number of recommendations:", min_value=1, max_value=10, value=4)
    
    if user_input:
        recommendations = knn_from_scratch(user_input, df, k)
        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write(movie)

if __name__ == '__main__':
    main()


