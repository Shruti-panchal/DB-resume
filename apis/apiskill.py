from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import logging

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

class UserInput(BaseModel):
    category: str
    experience: int
    skills: list
    
@app.post("/recommend_skills")   
def recommend_skills(user_input: UserInput):
    try:
        # Assuming you have a list of resumes and skills
        final = pd.read_csv(r'DB-resume\resume_parser\final.csv')

        resumes = final['Resume'].tolist()
        skills = final['Skill'].tolist()
        categories = final['Category'].tolist()
        experience = final['Experience(months)'].tolist()

        # # User inputs
        # user_category = input("Enter your category/job title: ")
        # user_experience = int(input("Enter your total experience in months: "))
        # user_skills = input("Enter your skills : ").strip('[]').split(',')

        # Filter data based on user input
        filtered_data = final[(final['Category'].str.lower() == user_input.category.lower()) & (final['Experience(months)'] <= user_input.experience)]

        # print(f"filtered data : {filtered_data}")
        # print("\n")

        if filtered_data.empty:
            raise HTTPException(status_code=404, detail=f"No matching resumes found for the category '{user_input.category}' and experience '{user_input.experience}' months.")

        # Separate filtered resumes for training
        filtered_resumes = filtered_data['Resume'].tolist()
        filtered_labels = list(range(len(filtered_data)))

        # Combine user details with existing skills for training
        user_details = f"{user_input.category}, {', '.join(user_input.skills)}, {user_input.experience}"
        all_data = filtered_resumes + [user_details]

        # Convert text data to TF-IDF features
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(all_data)

        # Train the model only on the filtered data
        model = MultinomialNB()
        model.fit(X[:-1], filtered_labels)  # Exclude the user details for training

        # User details to predict skills
        user_data = vectorizer.transform([user_details])

        # Predict skills for user details
        predicted_labels = model.predict(user_data)

        # Get the corresponding skills based on predicted labels
        filtered_predicted_skills = [filtered_data.iloc[i]['Skill'] for i in predicted_labels]

        # Provide recommendations
        if filtered_predicted_skills:
            return {"Recommended Skills ": filtered_predicted_skills}
        else:
            raise HTTPException(status_code=404, detail="Pass your skills. No matching skills found for the given category and experience.")
    except Exception as e:
        print(f"Error occured :::::::::::::::: {e}")
        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apiskill:app", host="127.0.0.1", port=8000, reload=True)    