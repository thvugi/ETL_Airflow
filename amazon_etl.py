import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
import plotly.io as pio
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import logging
from sklearn.cluster import KMeans
from summarizer import Summarizer
from wordcloud import WordCloud as wc


def run_amazon_etl():
        df = pd.read_csv('final_book_dataset_kaggle_2.csv')




        #prive vs reviews
        px.scatter(df, x = "price", y = "avg_reviews", size = "n_reviews")
        px.scatter(df, x = "price", y = "avg_reviews", size = "n_reviews").write_image("images/fig1.png")

        #price vs pages
        px.scatter(df, x = "price", y="pages", size = "n_reviews")
        px.scatter(df, x = "price", y="pages", size = "n_reviews").write_image("images/fig4.png")


        #Best python books
        python_books = df[df['title'].str.contains("Python")]
        best_python_books = python_books.nlargest(7,['n_reviews','avg_reviews'])

        #Best machine learning books

        ML_books = df[df['title'].str.contains("Machine Learning")]
        best_ML_books = ML_books.nlargest(7,['n_reviews','avg_reviews'])




        vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range=(1,2))
        X = vectorizer.fit_transform(df['title'])
        pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names_out())

        #Kmeans clustering from tfidf vectorization

        sum_of_squared_distance = []


        K = range(2,10)
        for k in K:
            km = KMeans(n_clusters = k, max_iter=600, n_init=10)
            km.fit(X)
            sum_of_squared_distance.append(km.inertia_)

        plt.plot(K, sum_of_squared_distance, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Sum of Squared Distances')
        plt.title ('Elbow Method for Optimal k')
        plt.savefig('images/fig2.png')

        #from the above visualization using the elbow method, there's a decrease in rate at around 5 or 7 we will choose 6

        true_k = 6
        model = KMeans(n_clusters = true_k, init = 'k-means++', max_iter = 600, n_init = 10)
        model.fit(X)

        labels = model.labels_
        book_cl = pd.DataFrame(list(zip(df['title'],labels)),columns = ['title','cluster'])
        print(book_cl.sort_values(by=['cluster']))


        for k in range(true_k):
            text = book_cl[book_cl.cluster == k]['title'].str.cat(sep = ' ')
            wordcloud = wc(max_font_size = 50, max_words = 100, background_color = "white").generate(text)
            
            #create subplots
            plt.subplot(2,3,k + 1).set_title("Cluster " + str(k))
            plt.plot()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
        plt.savefig('images/fig3')
                                                                

        def get_review_url(product_url):
            try:
                split_url = product_url.split('dp')
                product_number = split_url[1].split('/')[1]
                review_url = split_url[0] + 'product-reviews/' + product_number + "/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"
            except:
                review_url = None
            return review_url

        df['review_urls'] = df['complete_link'].apply(lambda x: get_review_url(x))

        df_reviews = df.loc[~df['review_urls'].isnull()].reset_index()

        # code to retrieve amazon review and save into df

        headers = {
            "authority": "www.amazon.com",
            "pragma": "no-cache",
            "cache-control": "no-cache",
            "dnt": "1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "sec-fetch-site": "none",
            "sec-fetch-mode": "navigate",
            "sec-fetch-dest": "document",
            "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        }

        URLS = df_reviews['review_urls']
        book_titles = df_reviews['title']

        def get_page_html(page_url: str) -> str:
            resp = requests.get(page_url, headers=headers)
            return resp.text

        def get_reviews_from_html(page_html: str) -> BeautifulSoup:
            soup = BeautifulSoup(page_html, "lxml")
            reviews = soup.find_all("div", {"class": "a-section celwidget"})
            return reviews

        def get_review_text(soup_object: BeautifulSoup) -> str:
            review_text = soup_object.find(
                "span", {"class": "a-size-base review-text review-text-content"}
            ).get_text()
            return review_text.strip()

        def get_number_stars(soup_object: BeautifulSoup) -> str:
            stars = soup_object.find("span", {"class": "a-icon-alt"}).get_text()
            return stars.strip()

        def orchestrate_data_gathering(single_review: BeautifulSoup) -> dict:
            return {
                "review_text": get_review_text(single_review),
                "review_stars": get_number_stars(single_review)
            }

        if __name__ == '__main__':
            logging.basicConfig(level=logging.INFO)
            all_results = []

            for i in range(len(URLS)):
                logging.info(URLS[i])
                html = get_page_html(URLS[i])
                reviews = get_reviews_from_html(html)
                for rev in reviews:
                    data = orchestrate_data_gathering(rev)
                    data.update({'title': df_reviews['title'][i]})
                    all_results.append(data)

            out = pd.DataFrame.from_records(all_results)
            logging.info(f"Total number of reviews {out.shape[0]}")
            save_name = f"book_reviews_{datetime.now().strftime('%Y-%m-%d-%m')}.csv"
            logging.info(f"saving to {save_name}")
            out.to_csv(save_name, index=False)
            logging.info('Done yayy')


        book_reviews = out

        #aggregrating reviews for each book

        book_reviews['review_text'] = book_reviews['review_text'].astype(str)
        book_reviews_agg = book_reviews.groupby(['title'], as_index = False).agg({'review_text': ' '.join})
        book_reviews_agg

        #using bert extractive summarizer to summarize the reviews into shorter versions
      

        bert_model = Summarizer()
        bert_summary = ''.join(bert_model(book_reviews_agg.review_text[2],ratio = 0.2))

        text_file = open("summarizer.txt", "w")
        n = text_file.write(bert_summary)
        text_file.close()