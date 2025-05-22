import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import requests
import time

def scrape_karkidi_jobs(keyword="data science", pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        job_blocks = soup.find_all("div", class_="ads-details")
        for job in job_blocks:
            try:
                title = job.find("h4").get_text(strip=True)
                company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                location = job.find("p").get_text(strip=True)
                experience = job.find("p", class_="emp-exp").get_text(strip=True)
                key_skills_tag = job.find("span", string="Key Skills")
                skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
                summary_tag = job.find("span", string="Summary")
                summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

                jobs_list.append({
                    "Title": title,
                    "Company": company,
                    "Location": location,
                    "Experience": experience,
                    "Summary": summary,
                    "Skills": skills
                })
            except Exception as e:
                print(f"Error parsing job block: {e}")
                continue

        time.sleep(1)

    return pd.DataFrame(jobs_list)

def generate_cluster_names(df, skill_column='Skills', cluster_column='Cluster', top_n=2):
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_names = {}

    for cluster_id in sorted(df[cluster_column].unique()):
        cluster_skills = df[df[cluster_column] == cluster_id][skill_column].str.cat(sep=' ')
        if not cluster_skills.strip():
            cluster_names[cluster_id] = "Miscellaneous"
            continue
        tfidf = TfidfVectorizer(stop_words='english')
        X = tfidf.fit_transform([cluster_skills])
        scores = zip(tfidf.get_feature_names_out(), X.toarray()[0])
        sorted_keywords = sorted(scores, key=lambda x: x[1], reverse=True)
        top_keywords = [word for word, score in sorted_keywords[:top_n]]
        cluster_label = " / ".join(top_keywords).title()
        cluster_names[cluster_id] = cluster_label

    return cluster_names

def main():
    # Load model and vectorizer
    model = joblib.load("kmeans_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    # Scrape jobs
    df = scrape_karkidi_jobs(keyword="data science", pages=2)
    df = df[df['Skills'].str.strip() != ""]  # remove empty skills

    # Vectorize skills
    X = vectorizer.transform(df['Skills'])

    # Predict clusters
    df['Cluster'] = model.predict(X)

    # Generate cluster names
    cluster_name_map = generate_cluster_names(df)
    df['Cluster_Name'] = df['Cluster'].map(cluster_name_map)

    # Save clustered jobs
    df.to_csv("clustered_jobs.csv", index=False)
    print("Jobs scraped, clustered, and saved successfully.")

if __name__ == "__main__":
    main()
