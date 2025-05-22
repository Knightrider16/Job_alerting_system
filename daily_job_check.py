import requests, time, json, smtplib, re
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.preprocessing import normalize
import joblib
from email.message import EmailMessage

def scrape_latest_jobs():
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = "https://www.karkidi.com/Find-Jobs/1/all/India?search=data%20science"
    jobs_list = []

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    job_blocks = soup.find_all("div", class_="ads-details")

    for job in job_blocks:
        try:
            title = job.find("h4").get_text(strip=True)
            company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
            location = job.find("p").get_text(strip=True)
            experience = job.find("p", class_="emp-exp").get_text(strip=True)
            skills = job.find("span", string="Key Skills")
            skills = skills.find_next("p").get_text(strip=True) if skills else ""

            jobs_list.append({
                "Title": title,
                "Company": company,
                "Location": location,
                "Experience": experience,
                "Skills": skills
            })
        except:
            continue
    return pd.DataFrame(jobs_list)

def clean_skills(text):
    skills = re.split(r'[|,/\n]+', text)
    return ' '.join([re.sub(r'[^a-zA-Z\s]', '', s).lower().strip() for s in skills if s.strip()])

def send_email(to_email, jobs):
    msg = EmailMessage()
    msg["Subject"] = "New Jobs Matching Your Skills!"
    msg["From"] = "youremail@gmail.com"
    msg["To"] = to_email

    content = "\n\n".join([f"{j['Title']} at {j['Company']} ({j['Location']})\nSkills: {j['Skills']}" for j in jobs])
    msg.set_content(f"Here are new jobs matching your skills:\n\n{content}")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login("youremail@gmail.com", "your-app-password")
        smtp.send_message(msg)

def notify_users(new_jobs):
    kmeans = joblib.load("kmeans_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    new_jobs['CleanedSkills'] = new_jobs['Skills'].apply(clean_skills)
    X = vectorizer.transform(new_jobs['CleanedSkills'])
    new_jobs['Cluster'] = kmeans.predict(X)

    with open("user_preferences.json", "r") as f:
        users = json.load(f)

    for user in users:
        preferred_cluster = int(user["preferred_cluster"])
        matching_jobs = new_jobs[new_jobs['Cluster'] == preferred_cluster]
        if not matching_jobs.empty:
            send_email(user["email"], matching_jobs.to_dict(orient="records"))

if __name__ == "__main__":
    jobs = scrape_latest_jobs()
    if not jobs.empty:
        notify_users(jobs)
        print("User notifications sent.")
