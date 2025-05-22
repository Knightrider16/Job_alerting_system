import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def scrape_karkidi_jobs(keyword="data science", pages=2):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        print(f"Scraping page: {page}")
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

        time.sleep(1)  # Be kind to the server

    df = pd.DataFrame(jobs_list)
    print(f"Scraped {len(df)} jobs.")
    return df

def find_optimal_clusters(X, max_k=10):
    """Compute silhouette scores for K=2 to max_k and plot."""
    scores = []
    K = range(2, max_k + 1)
    for k in K:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"Silhouette score for k={k}: {score:.4f}")
        scores.append(score)

    best_k = K[scores.index(max(scores))]
    print(f"Best number of clusters by silhouette score: {best_k}")
    return best_k

def generate_cluster_names(df, top_n=5):
    """Generate meaningful cluster names by extracting top TF-IDF keywords for each cluster."""
    cluster_names = {}
    for cluster in df['Cluster'].unique():
        cluster_skills = df[df['Cluster'] == cluster]['Skills']
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = vectorizer.fit_transform(cluster_skills)
        # Average TF-IDF scores per term across all docs in cluster
        mean_tfidf = X.mean(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        # Top terms for cluster
        top_terms = [terms[i] for i in mean_tfidf.argsort()[::-1][:top_n]]
        cluster_names[cluster] = ", ".join(top_terms)
    return cluster_names

def main():
    print("Starting job scraping and clustering pipeline...")

    # Step 1: Scrape data
    df = scrape_karkidi_jobs(keyword="data science", pages=2)

    # Step 2: Remove jobs with empty skills
    df = df[df['Skills'].str.strip() != ""].reset_index(drop=True)
    if df.empty:
        print("No jobs with skills found. Exiting.")
        return

    # Step 3: Vectorize skills using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Skills'])

    # Save vectorizer for later use
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("TF-IDF vectorizer saved as 'vectorizer.pkl'.")

    # Step 4: Find best number of clusters using silhouette score
    best_k = find_optimal_clusters(X, max_k=10)

    # Step 5: Train final KMeans model with best_k clusters
    model = KMeans(n_clusters=best_k, random_state=42)
    clusters = model.fit_predict(X)
    df['Cluster'] = clusters

    # Save model
    joblib.dump(model, "kmeans_model.pkl")
    print(f"KMeans model trained with k={best_k} and saved as 'kmeans_model.pkl'.")

    # Step 6: Evaluate and print final silhouette score
    final_score = silhouette_score(X, clusters)
    print(f"Final silhouette score for k={best_k}: {final_score:.4f}")

    # Step 7: Generate cluster names (top keywords)
    cluster_name_map = generate_cluster_names(df)
    df['Cluster_Name'] = df['Cluster'].map(cluster_name_map)

    print("Cluster names assigned:")
    for c, name in cluster_name_map.items():
        print(f"Cluster {c}: {name}")

    # Step 8: Save clustered jobs to CSV
    df.to_csv("clustered_jobs.csv", index=False)
    print("Clustered job data saved to 'clustered_jobs.csv'.")

if __name__ == "__main__":
    main()
