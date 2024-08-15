import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Define the mappings
education_mapping = {
    '2n Cycle': 0,
    'Basic': 1,
    'Graduation': 2,
    'Master': 3,
    'PhD': 4
}

marital_status_mapping = {
    'Absurd': 0,
    'Alone': 1,
    'Divorced': 2,
    'Married': 3,
    'Single': 4,
    'Together': 5,
    'Widow': 6,
    'YOLO': 7
}

# Provide the absolute path to your model and data files
model_filename = r'C:\Users\Home\OneDrive\Desktop\python\final_model.sav'
data_filename = r'C:\Users\Home\OneDrive\Desktop\data project\modified_data_clusters.csv'

# Check if the model file exists
if not os.path.exists(model_filename):
    st.error(f"Model file not found: {model_filename}")
else:
    loaded_model = pickle.load(open(model_filename, 'rb'))

    # Check if the data file exists
    if not os.path.exists(data_filename):
        st.error(f"Data file not found: {data_filename}")
    else:
        df = pd.read_csv(data_filename)

        st.markdown('<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)
        st.title("Prediction")

        with st.form("my_form"):
            education = st.selectbox('Education', options=list(education_mapping.keys()))
            marital_status = st.selectbox('Marital Status', options=list(marital_status_mapping.keys()))
            income = st.number_input(label='Income', step=1000.0, format="%.2f")  # Ensure step is float for decimals
            kidhome = st.number_input(label='Kidhome', step=1, format="%d")
            teenhome = st.number_input(label='Teenhome', step=1, format="%d")
            recency = st.number_input(label='Recency', step=1, format="%d")
            mntwines = st.number_input(label='MntWines', step=0.01, format="%.2f")
            mntfruits = st.number_input(label='MntFruits', step=0.01, format="%.2f")
            mntmeatproducts = st.number_input(label='MntMeatProducts', step=0.01, format="%.2f")
            mntfishproducts = st.number_input(label='MntFishProducts', step=0.01, format="%.2f")
            mntsweetproducts = st.number_input(label='MntSweetProducts', step=0.01, format="%.2f")
            mntgoldprods = st.number_input(label='MntGoldProds', step=0.01, format="%.2f")
            numdealspurchases = st.number_input(label='NumDealsPurchases', step=1)
            numwebpurchases = st.number_input(label='NumWebPurchases', step=1)
            numcatalogpurchases = st.number_input(label='NumCatalogPurchases', step=1)
            numstorepurchases = st.number_input(label='NumStorePurchases', step=1)
            numwebvisitsmonth = st.number_input(label='NumWebVisitsMonth', step=1)
            acceptedcmp3 = st.number_input(label='AcceptedCmp3', step=1)
            acceptedcmp4 = st.number_input(label='AcceptedCmp4', step=1)
            acceptedcmp5 = st.number_input(label='AcceptedCmp5', step=1)
            acceptedcmp1 = st.number_input(label='AcceptedCmp1', step=1)
            acceptedcmp2 = st.number_input(label='AcceptedCmp2', step=1)
            age = st.number_input(label='Age', step=1)
            ttl_yrs_in_comp = st.number_input(label='Total Years in Company', step=1)
            total_amount_spend = st.number_input(label='Total Amount Spend', step=0.01, format="%.2f")

            # Create a list of input data
            data = [[
                education_mapping.get(education, -1),
                marital_status_mapping.get(marital_status, -1),
                income,
                kidhome,
                teenhome,
                recency,
                mntwines,
                mntfruits,
                mntmeatproducts,
                mntfishproducts,
                mntsweetproducts,
                mntgoldprods,
                numdealspurchases,
                numwebpurchases,
                numcatalogpurchases,
                numstorepurchases,
                numwebvisitsmonth,
                acceptedcmp3,
                acceptedcmp4,
                acceptedcmp5,
                acceptedcmp1,
                acceptedcmp2,
                age,
                ttl_yrs_in_comp,
                total_amount_spend
            ]]

            # Add placeholders for missing features
            while len(data[0]) < 36:
                data[0].append(0)  # Assuming 0 as a placeholder; adjust if necessary

            submitted = st.form_submit_button("Submit")

        if submitted:
            if -1 in data[0]:
                st.error("Invalid input detected. Please ensure all fields are filled out correctly.")
            else:
                try:
                    clust = loaded_model.predict(data)[0]
                    st.write(f'Data Belongs to Cluster {clust}')

                    cluster_df1 = df[df['cluster_num'] == clust]
                    plt.rcParams["figure.figsize"] = (20, 3)
                    for c in cluster_df1.drop(['cluster_num'], axis=1):
                        fig, ax = plt.subplots()
                        sns.histplot(cluster_df1[c], kde=False, bins=30)
                        plt.title(c)
                        st.pyplot(fig)
                except ValueError as e:
                    st.error(f"Prediction failed: {e}")
