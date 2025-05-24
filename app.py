from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)

# load files
trending_products = pd.read_csv('models/trending_products.csv')
train_data = pd.read_csv('models/clean_data.csv')

def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

def content_based_recommendations(train_data, search_term, top_n=10):
    # Create a TF-IDF vectorizer for item descriptions and names
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2))
    
    # Combine Name and Tags for better matching
    combined_features = train_data['Name'].fillna('') + ' ' + train_data['Tags'].fillna('')
    
    # Apply TF-IDF vectorization to combined features
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_features)
    
    # Transform the search term using the same vectorizer
    search_vector = tfidf_vectorizer.transform([search_term.lower()])
    
    # Calculate cosine similarity between search term and all products
    similarity_scores = cosine_similarity(search_vector, tfidf_matrix).flatten()
    
    # Get indices of products sorted by similarity score
    similar_indices = similarity_scores.argsort()[::-1]
    
    # Filter out products with very low similarity (threshold = 0.01)
    filtered_indices = [idx for idx in similar_indices if similarity_scores[idx] > 0.01]
    
    if not filtered_indices:
        print(f"No similar products found for '{search_term}'")
        return pd.DataFrame()
    
    # Get top N recommendations
    top_indices = filtered_indices[:top_n]
    
    # Get the details of the recommended items
    recommended_items = train_data.iloc[top_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']].copy()
    
    # Add similarity scores for reference
    recommended_items['Similarity'] = [similarity_scores[idx] for idx in top_indices]
    
    print(f"Found {len(recommended_items)} recommendations for '{search_term}'")
    return recommended_items

# List of predefined image URLs
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]

#routes
@app.route('/')
def index():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html',trending_products=trending_products.head(8),truncate = truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price = random.choice(price))

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/index')
def indexRedirect():
    # Use the same data as the main index route
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html',trending_products=trending_products.head(8),truncate = truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price = random.choice(price))

@app.route('/recommendations', methods=['POST','GET'])
def recommendations():
    if request.method == 'POST':
        search_term = request.form.get('prod', '').strip()
        nbr_input = request.form.get('nbr')
        
        # Handle empty or invalid nbr input more robustly
        nbr = 10  # Default value
        if nbr_input is not None and nbr_input.strip():
            try:
                nbr = int(nbr_input.strip())
                if nbr <= 0:
                    nbr = 10  # Reset to default if negative or zero
            except (ValueError, TypeError):
                nbr = 10  # Default to 10 if invalid input
        
        if not search_term:
            message = "Please enter a search term."
            return render_template('main.html', message=message)
        
        content_based_rec = content_based_recommendations(train_data, search_term, top_n=nbr)
        
        if content_based_rec.empty:
            message = f"No recommendations found for '{search_term}'. Try different keywords."
            return render_template('main.html', message=message)
        else:
            # Create a list of random image URLs for each recommended product
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(content_based_rec))]
            print(content_based_rec)
            print(random_product_image_urls)

            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main.html', content_based_rec=content_based_rec, truncate=truncate,
                                    random_product_image_urls=random_product_image_urls,
                                    random_price=random.choice(price))
    
    # Handle GET request - just show the main page
    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True)