import os
import joblib
import re
import nltk
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Baixar recursos necessários do NLTK (se ainda não foram baixados)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# Caminho do diretório raiz do projeto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Função de pré-processamento de texto
def limpar_texto(texto):
    texto = texto.lower()  # Converter para minúsculas
    texto = unidecode(texto)  # Remover acentos
    texto = re.sub(r'[^a-z\s]', '', texto)  # Remover caracteres especiais e números
    return texto

def preprocessar_texto(texto):
    texto_limpo = limpar_texto(texto)
    tokens = word_tokenize(texto_limpo)
    stop_words = set(stopwords.words('portuguese'))
    # Preservar palavras negativas
    palavras_negativas = {'não', 'nao', 'nunca', 'jamais', 'nem'}
    stop_words = stop_words - palavras_negativas
    tokens_sem_stopwords = [palavra for palavra in tokens if palavra not in stop_words]
    # Stemização com RSLPStemmer
    stemmer = RSLPStemmer()
    tokens_stemizados = [stemmer.stem(palavra) for palavra in tokens_sem_stopwords]
    return " ".join(tokens_stemizados)

# Função para treinar ou carregar o modelo
def treinar_ou_carregar_modelo(force_retrain=False):
    MODELO_PATH = os.path.join(ROOT_DIR, 'models', 'modelos_incremental.pkl')
    VECTORIZER_PATH = os.path.join(ROOT_DIR, 'models', 'vectorizer.pkl')

    # Verificar se o diretório 'models/' existe, se não, criá-lo
    models_dir = os.path.join(ROOT_DIR, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not force_retrain and os.path.exists(MODELO_PATH) and os.path.exists(VECTORIZER_PATH):
        # Carregar os modelos e o vectorizer já treinados
        modelos = joblib.load(MODELO_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Modelos carregados do arquivo.")
    else:
        print("Nenhum dado de treinamento externo será carregado.")
        
        # Inicializar o vectorizer com TfidfVectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))

        # Aguardar a coleta das avaliações para treinamento do modelo

        # Inicializar os modelos
        modelos = {}

        # Treinar modelos
        # Naive Bayes Multinomial
        naive_bayes = MultinomialNB()
        modelos['Naive Bayes'] = naive_bayes
        print("Naive Bayes preparado para treinamento.")

        # SGDClassifier
        sgd_classifier = SGDClassifier()
        modelos['SGD Classifier'] = sgd_classifier
        print("SGD Classifier preparado para treinamento.")

        # Salvar os modelos e o vectorizer treinados
        joblib.dump(modelos, MODELO_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        print("Modelos preparados e salvos.")

    return modelos, vectorizer