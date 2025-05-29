from flask import Flask, request, render_template, send_file, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from model_loader import load_model, aspects, id2label
import torch
import pandas as pd
import os
import plotly.express as px
from io import BytesIO, TextIOWrapper
import csv
from datetime import datetime
import plotly.graph_objects as go
from plotly.offline import plot
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://sentimen_user:password123@localhost/sentimen_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Model Database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def check_password(self, password):
        return self.password == password

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100))
    ulasan = db.Column(db.Text, nullable=False)
    fasilitas = db.Column(db.String(20))
    harga = db.Column(db.String(20))
    pelayanan = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()
    
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            password='admin123',
            created_at=datetime.utcnow()
        )
        db.session.add(admin)
        db.session.commit()

# Text preprocessing function
def preprocess_text(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Case folding
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        
        return ' '.join(tokens)
    
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return text

# Load Model
models = {aspect.lower(): load_model(aspect) for aspect in aspects}

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Login gagal!', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

def predict_sentiment(aspect, text):
    try:
        processed_text = preprocess_text(text)
        
        model, tokenizer = models[aspect.lower()]
        inputs = tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)
            predicted = torch.argmax(outputs.logits, dim=1).item()
        return id2label[predicted]
    except Exception as e:
        print(f"Error prediksi {aspect}: {str(e)}")
        return 'error'

def migrate_csv():
    csv_path = './data/reviews.csv'
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f, delimiter=',', quotechar='"')
                headers = next(csv_reader)
                headers = [h.strip().lower() for h in headers]
                
                nama_idx = headers.index('nama_pengirim')
                ulasan_idx = headers.index('isi_ulasan')
                
                total = 0
                for row_idx, row in enumerate(csv_reader):
                    try:
                        nama = row[nama_idx].strip() or 'Anonim'
                        text = row[ulasan_idx].strip().strip('"').strip()
                        processed_text = preprocess_text(text)
                        
                        if len(processed_text) < 5:
                            continue
                            
                        results = {
                            'fasilitas': predict_sentiment('Fasilitas', processed_text),
                            'harga': predict_sentiment('Harga', processed_text),
                            'pelayanan': predict_sentiment('Pelayanan', processed_text)
                        }
                        
                        review = Review(
                            nama=nama,
                            ulasan=text,  # Simpan teks asli
                            **results
                        )
                        db.session.add(review)
                        total += 1
                        
                    except Exception as e:
                        print(f"Error baris {row_idx+2}: {str(e)}")
                        continue
                
                db.session.commit()
                print(f"Berhasil migrasi {total} data")
                os.rename(csv_path, './data/reviews_imported.csv')
            
        except Exception as e:
            print(f"Error migrasi: {str(e)}")

# Run migration when starting app
with app.app_context():
    migrate_csv()

@app.route("/")
@login_required
def home():
    search_query = request.args.get('q', '').strip()

    if search_query:
        reviews = Review.query.filter(Review.ulasan.ilike(f'%{search_query}%')).order_by(Review.created_at.desc()).all()
    else:
        reviews = Review.query.order_by(Review.created_at.desc()).all()

    data_for_template = [{
        'no': idx + 1,
        'nama': r.nama,
        'ulasan': r.ulasan,
        'fasilitas': r.fasilitas,
        'harga': r.harga,
        'pelayanan': r.pelayanan
    } for idx, r in enumerate(reviews)]

    df = pd.DataFrame(data_for_template)
    charts = {}

    for aspect in aspects:
        aspect_lower = aspect.lower()
        try:
            if not df.empty:
                counts = df[aspect_lower].value_counts().to_dict()
                
                label_order = ['positif', 'netral', 'negatif']
                color_map = {
                    'positif': '#28a745',
                    'netral': '#6c757d',
                    'negatif': '#dc3545'
                }

                labels = sorted(counts.keys(), key=lambda x: label_order.index(x))
                values = [counts[label] for label in labels]
                colors = [color_map[label] for label in labels]

                fig = go.Figure(
                    data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.4,
                        pull=[0.05] * len(labels),
                        marker=dict(
                            line=dict(color='white', width=2),
                            colors=colors
                        ),
                        textinfo='label+percent',
                        hoverinfo='label+value+percent'
                    )]
                )

                fig.update_layout(
                    title=dict(text=f'Sentimen {aspect}', x=0.5),
                    height=350,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=14),
                    margin=dict(t=40, b=10, l=10, r=10)
                )

                charts[aspect] = {
                    'html': plot(fig, output_type='div', include_plotlyjs=False),
                    'counts': counts
                }
            else:
                charts[aspect] = {'html': "<p>Tidak ada data</p>", 'counts': {}}
        except Exception as e:
            print(f"Error chart {aspect}: {str(e)}")
            charts[aspect] = {'html': "<p>Error visualisasi</p>", 'counts': {}}

    return render_template("visualization.html", charts=charts, data=data_for_template, aspects=aspects, search_query=search_query)

@app.route("/download")
@login_required
def download():
    reviews = Review.query.all()
    df = pd.DataFrame([{
        'No': r.id,
        'Nama': r.nama,
        'Ulasan': r.ulasan,
        'Fasilitas': r.fasilitas,
        'Harga': r.harga,
        'Pelayanan': r.pelayanan
    } for r in reviews])
    
    buffer = BytesIO()
    df.to_csv(buffer, index=False, encoding='utf-8', quoting=1)
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='text/csv',
        download_name=f'data_sentimen_{datetime.now().strftime("%Y%m%d")}.csv',
        as_attachment=True
    )

@app.route("/add_review", methods=['GET', 'POST'])
@login_required
def add_review():
    if request.method == 'POST':
        if 'text' in request.form:
            text = request.form.get('text', '').strip()
            nama = request.form.get('nama', 'Anonim').strip()
            processed_text = preprocess_text(text)
            
            if not processed_text or len(processed_text) < 5:
                flash('Ulasan harus minimal 5 karakter setelah pra-processing!', 'danger')
                return redirect(url_for('add_review'))
            
            try:
                results = {
                    'fasilitas': predict_sentiment('Fasilitas', processed_text),
                    'harga': predict_sentiment('Harga', processed_text),
                    'pelayanan': predict_sentiment('Pelayanan', processed_text)
                }
                
                new_review = Review(
                    nama=nama,
                    ulasan=text,  # Simpan teks asli
                    **results
                )
                db.session.add(new_review)
                db.session.commit()
                
                return render_template("add_review.html", 
                                    result=results,
                                    nama=nama,
                                    text=text)
            
            except Exception as e:
                db.session.rollback()
                flash(f'Gagal menyimpan data: {str(e)}', 'danger')

        if 'csv_file' in request.files:
            file = request.files['csv_file']
            if file.filename == '':
                flash('File tidak dipilih!', 'danger')
                return redirect(url_for('add_review'))
            
            if not file.filename.endswith('.csv'):
                flash('File harus berekstensi CSV!', 'danger')
                return redirect(url_for('add_review'))
            
            try:
                csv_stream = TextIOWrapper(
                    file.stream, 
                    encoding='utf-8',
                    newline=''
                )
                csv_reader = csv.reader(
                    csv_stream,
                    delimiter=',',
                    quotechar='"',
                    skipinitialspace=True
                )
                
                try:
                    headers = next(csv_reader)
                except StopIteration:
                    flash('File CSV kosong!', 'danger')
                    return redirect(url_for('add_review'))
                
                headers = [h.strip().lower() for h in headers]
                
                required_columns = {'nama', 'ulasan'}
                if not required_columns.issubset(set(headers)):
                    missing = required_columns - set(headers)
                    flash(f'Kolom wajib tidak ada: {", ".join(missing)}', 'danger')
                    return redirect(url_for('add_review'))
                
                nama_idx = headers.index('nama')
                ulasan_idx = headers.index('ulasan')
                
                total_success = 0
                errors = []
                new_reviews = []
                aspect_counts = {
                    aspect.lower(): {'positif': 0, 'negatif': 0, 'netral': 0} 
                    for aspect in aspects
                }
                
                for row_idx, row in enumerate(csv_reader):
                    try:
                        if len(row) <= max(nama_idx, ulasan_idx):
                            raise ValueError("Jumlah kolom tidak cukup")
                        
                        nama = row[nama_idx].strip() or 'Anonim'
                        text = row[ulasan_idx].strip().strip('"').strip()
                        processed_text = preprocess_text(text)
                        
                        if len(processed_text) < 5:
                            raise ValueError(f"Ulasan terlalu pendek ({len(processed_text)} karakter setelah pra-processing)")
                        
                        results = {
                            'fasilitas': predict_sentiment('Fasilitas', processed_text),
                            'harga': predict_sentiment('Harga', processed_text),
                            'pelayanan': predict_sentiment('Pelayanan', processed_text)
                        }

                        for aspect, sentiment in results.items():
                            aspect_counts[aspect][sentiment] += 1

                        new_review = Review(
                            nama=nama,
                            ulasan=text,  # Simpan teks asli
                            **results
                        )
                        db.session.add(new_review)
                        db.session.commit()
                        new_reviews.append(new_review)
                        total_success += 1
                        
                    except Exception as e:
                        db.session.rollback()
                        error_msg = f"Baris {row_idx+2}: {str(e)}"
                        errors.append(error_msg)

                summary_new = {}
                for aspect, counts in aspect_counts.items():
                    summary_new[aspect] = {
                        'positif': counts['positif'],
                        'negatif': counts['negatif'],
                        'netral': counts['netral']
                    }

                return render_template("add_review.html",
                                     summary_new=summary_new,
                                     total_success=total_success,
                                     errors=errors[:5])

            except Exception as e:
                flash(f'Error processing CSV: {str(e)}', 'danger')
    
    return render_template("add_review.html")

if __name__ == "__main__":
    app.run(debug=True)