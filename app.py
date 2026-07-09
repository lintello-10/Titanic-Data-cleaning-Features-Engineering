import streamlit as st
import joblib
import pandas as pd
 
# Load the trained model
model = joblib.load('titanic_model.joblib')

# Title of the application
st.title("Titanic Survival Prediction")
st.write("Enter the passenger's details to predict their survival probability.")

# --- USER INPUTS ---

# 1. Ticket Class
pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])

# 2. Gender
sex = st.selectbox("Gender", ["male", "female"])
sex_encoded = int(sex == 'female')

# 3. Age
age = st.slider("Age", 0, 100, 28)

# 4. Family Members
sibsp = st.number_input("Number of Siblings/Spouses aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children aboard (Parch)", min_value=0, max_value=10, value=0)

# 5. Title
title = st.selectbox("Passenger Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
title_mr = int(title == 'Mr')
title_mrs = int(title == 'Mrs')
title_miss = int(title == 'Miss')
title_master = int(title == 'Master')
title_rare = int(title == 'Rare')

# 6. Port of Embarkation
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
embarked_c = int(embarked == 'C')
embarked_q = int(embarked == 'Q')
embarked_s = int(embarked == 'S')

# 7. Ticket Fare
fare = st.number_input("Ticket Fare (in USD)", min_value=0.0, max_value=512.0, value=32.0)


# --- FEATURE ENGINEERING ---

# Calculate Family Size and IsAlone
family_size = sibsp + parch + 1
is_alone = int(family_size == 1)

# Process Age Groups
age_5_18 = 0
age_18_35 = 0
age_35_60 = 0
age_60_80 = 0

if 5 < age <= 18:
    age_5_18 = 1
elif 18 < age <= 35:
    age_18_35 = 1
elif 35 < age <= 60:
    age_35_60 = 1
elif 60 < age <= 80:
    age_60_80 = 1


# --- MODEL INFERENCE ---

# Create the final DataFrame with all 17 features in the exact training order
input_data = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex_encoded,
    'AgeGroup_(5, 18]': age_5_18,
    'AgeGroup_(18, 35]': age_18_35,
    'AgeGroup_(35, 60]': age_35_60,
    'AgeGroup_(60, 80]': age_60_80,
    'FamilySize': family_size,
    'IsAlone': is_alone,
    'Title_Master': title_master,
    'Title_Miss': title_miss,
    'Title_Mr': title_mr,
    'Title_Mrs': title_mrs,
    'Title_Rare': title_rare,
    'Embarked_C': embarked_c,
    'Embarked_Q': embarked_q,
    'Embarked_S': embarked_s,
    'Fare': fare
}])
 
 # Predict button
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("🎉 The passenger is predicted to **SURVIVE**.")
    else:
        st.error("💀 The passenger is predicted to **NOT SURVIVE**.")