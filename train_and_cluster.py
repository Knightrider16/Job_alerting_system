import requests, time, re, joblib
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def scrape_karkidi_jobs(keyword="data science", pages=2):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        print(f"Scraping: {url}")
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, "html.parser")
        except Exception as e:
            print(f"Request failed: {e}")
            continue

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
            except Exception as e:
                continue
        time.sleep(1)

    return pd.DataFrame(jobs_list)

def clean_skills(text):
    skills = re.split(r'[|,/\n]+', text)
    return ' '.join([re.sub(r'[^a-zA-Z\s]', '', s).lower().strip() for s in skills if s.strip()])

def train_and_cluster(df, n_clusters=5):
    df['CleanedSkills'] = df['Skills'].apply(clean_skills)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['CleanedSkills'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    joblib.dump(kmeans, "kmeans_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    df.to_csv("clustered_jobs.csv", index=False)

    return df

if __name__ == "__main__":
    df = scrape_karkidi_jobs()
    clustered = train_and_cluster(df)
    print("Training complete. Model and data saved.")
