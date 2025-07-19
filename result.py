import tkinter as tk
from tkinter import scrolledtext
from bs4 import BeautifulSoup
import requests
import pandas as pd
from joblib import load
import pickle
from scipy.sparse import csr_matrix

# Define the sentiment analysis function
def analyze_sentiment(text):
    predictor = load("SVM_MODEL1.joblib")
    vec = open('vectorizer1.pickle', 'rb')
    tf_vect = pickle.load(vec)
    X_test_tf = csr_matrix(tf_vect.transform([text]))  # Use csr_matrix explicitly
    y_predict = predictor.predict(X_test_tf)
    sentiment = "Positive Review" if y_predict[0] == 2 else "Negative Review"
    return text, sentiment


def cus_data(soup):
    cus_list = []

    for item in soup.find_all("span", class_="a-profile-name"):
        cus_list.append(item.get_text().strip())

    return cus_list


def clean_review_text(review_text):
    # Remove '\n' characters
    clean_text = review_text.replace('\n', ' ')
    
    # Remove "Read more" text
    clean_text = clean_text.replace('Read more', '')

    # Remove extra whitespaces
    clean_text = ' '.join(clean_text.split())

    return clean_text

def cus_rev(soup):
    result = []

    for item in soup.find_all("div", class_="a-row a-spacing-small review-data"):
        review_text = item.get_text().strip()
        cleaned_text = clean_review_text(review_text)
        result.append(cleaned_text)

    return [i for i in result if i != ""]


def get_and_analyze_reviews():
    url = url_entry.get()
    if url:
        HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
                   'Accept-Language': 'en-US, en;q=0.5'}
        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.text, 'html.parser')

        reviews_text.delete(1.0, tk.END)  # Clear previous content

        cus_res = cus_data(soup)
        rev_data = cus_rev(soup)
        rev_result = []

        positive_reviews_count = 0
        negative_reviews_count = 0

        for i in rev_data:
            if i != "":
                review_text, sentiment = analyze_sentiment(i)
                rev_result.append(review_text)

                if sentiment == "Fake Review":
                    positive_reviews_count += 1
                    
                else:
                    negative_reviews_count += 1

                formatted_review = f"Review: {review_text}\nSentiment: {sentiment}\n\n"

                # Set the text color based on sentiment
                if sentiment == "Real Review":
                    reviews_text.insert(tk.END, formatted_review, "positive")
                else:
                    reviews_text.insert(tk.END, formatted_review, "negative")

        # Update the GUI with the counts
        positive_count_label.config(text=f"Positive Reviews: {positive_reviews_count}")
        negative_count_label.config(text=f"Negative Reviews: {negative_reviews_count}")

        min_length = min(len(cus_res), len(rev_result))
        data = {'Name': cus_res[:min_length], 'Review': rev_result[:min_length], 'Sentiment': [analyze_sentiment(review)[1] for review in rev_result]}
        df = pd.DataFrame(data)
        print(df)
        
        

        # Append the results to the existing CSV file
        df.to_csv('amazon_review_with_sentiment.csv', mode='a', header=False, index=False)
        
        if positive_reviews_count > negative_reviews_count:
            negative_count_label1.config(text="Based on Above Product Reviews show  positive reviews and good product. so you can purchase the product. !!")
        else:
            negative_count_label1.config(text="Based on Above Product Reviews  show Negative reviews. so you can't purchase the product. !!")

            



from PIL import Image, ImageTk
       
# Create the main window
root = tk.Tk()
root.title("Amazon Reviews Analyzer")
root.configure(background="white")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Product Review Sentiment Analysis Using ML ")


####For background Image
image2 = Image.open('img3.jpg')
image2 = image2.resize((w, h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)

# label_l1 = tk.Label(root, text="Product Review Sentiment Analysis Using ML",font=("Times New Roman", 30, 'bold'),
#                     background="#000000", fg="white", width=70, height=2)
# label_l1.place(x=0, y=0)

frame = tk.LabelFrame(root,text="Control Panel",width=700,height=650,bd=3,background="cyan2",font=("Tempus Sanc ITC",15,"bold"))
frame.place(x=300,y=50)
# Create and place the URL entry and submit button
url_label = tk.Label(frame, text="Enter Amazon URL:",height=2,bg='black',fg='white',font=("Tempus Sanc ITC",15))
url_label.grid(row=0, column=0, padx=10, pady=10)

url_entry = tk.Entry(frame, width=100,bd=5)
url_entry.grid(row=0, column=1, padx=10, pady=10)

submit_button = tk.Button(frame, text="Submit", command=get_and_analyze_reviews,bg="black",fg="white",width=10,font=("Times New Roman",15,"bold"))
submit_button.grid(row=0, column=2, padx=10, pady=10)

# Create and place the scrolled text widget for displaying reviews
reviews_text = scrolledtext.ScrolledText(frame, width=120, height=30, wrap=tk.WORD)
reviews_text.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

# Configure text colors for positive and negative sentiments
reviews_text.tag_config("positive", foreground="green")
reviews_text.tag_config("negative", foreground="red")

# Create and place the labels for displaying counts
positive_count_label = tk.Label(frame, text="Positive Reviews: 0", bg='cyan2', font=("Tempus Sanc ITC", 12))
positive_count_label.grid(row=2, column=0, padx=10, pady=5)

negative_count_label = tk.Label(frame, text="Negative Reviews: 0", bg='cyan2', font=("Tempus Sanc ITC", 12))
negative_count_label.grid(row=2, column=1, padx=10, pady=5)


negative_count_label1 = tk.Label(root, text="**************-------------------      Show Result     -------------------**************",bg='black', fg='white',font=("times", 15))
negative_count_label1.place(x=370,y=750)
# Run the Tkinter main loop
root.mainloop()
