import argparse
from threading import Thread
from scripts.flask_server import iniciar_servidor_flask
from scripts.web_scraping import coletar_avaliacoes
from scripts.sentiment_analysis import treinar_ou_carregar_modelo, preprocessar_texto
from time import sleep
import signal
import sys
import pandas as pd

# Variável global para a thread do servidor Flask
flask_thread = None

# Função para lidar com o sinal de interrupção (Ctrl+C)
def signal_handler(sig, frame):
    print("\nInterrupção detectada! Encerrando o servidor Flask...")
    sys.exit(0)  # Encerra o programa

# Função para iniciar o servidor Flask em uma nova thread
def iniciar_servidor():
    global flask_thread
    flask_thread = Thread(target=iniciar_servidor_flask)
    flask_thread.daemon = True  # Torna a thread como "daemon" para encerrar junto com o programa principal
    flask_thread.start()
    sleep(5)  # Esperar o servidor inicializar

def main():
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Análise de Sentimentos em Avaliações de Produtos')
    parser.add_argument('--retrain', action='store_true', help='Retrainar o modelo do zero')
    args = parser.parse_args()

    # Definir o handler para o sinal de interrupção (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    # Iniciar o servidor Flask
    iniciar_servidor()

    # Fazer web scraping das avaliações
    url = "http://127.0.0.1:5000/"
    avaliacoes_originais = coletar_avaliacoes(url)

    if avaliacoes_originais:
        # Carregar ou treinar o modelo de análise de sentimentos
        modelos, vectorizer = treinar_ou_carregar_modelo(force_retrain=args.retrain)

        if modelos is None or vectorizer is None:
            print("Não foi possível carregar ou treinar os modelos.")
            sys.exit(1)

        # Pré-processar os textos das avaliações
        dados = []
        for avaliacao in avaliacoes_originais:
            texto_preprocessado = preprocessar_texto(avaliacao['texto'])
            dados.append(texto_preprocessado)

        # Transformar os dados das avaliações
        vetores_dados = vectorizer.transform(dados)

        # Inicializando a tabela para armazenar os resultados
        tabela = []

        # Aplicar o modelo em todas as avaliações
        for i in range(len(dados)):
            vetor_avaliacao = vetores_dados[i]
            linha = {
                'Avaliação original': avaliacoes_originais[i]['texto'],
                'Nota': avaliacoes_originais[i]['nota']
            }

            for nome_modelo, modelo in modelos.items():
                sentimento = modelo.predict(vetor_avaliacao)
                sentimento_label = {1: "Positivo", 0: "Neutro", -1: "Negativo"}.get(sentimento[0], "Indeterminado")
                linha[f'{nome_modelo}'] = sentimento_label

            tabela.append(linha)

        # Exibir a tabela de resultados
        df = pd.DataFrame(tabela)
        print("\nTabela de Resultados:\n")
        print(df.to_string(index=False))

        df.to_csv('resultados_analise.csv', index=False, encoding='utf-8-sig')

        # Exibir uma contagem de sentimentos
        print("\nContagem de Sentimentos Preditos (Naive Bayes):")
        print(df['Naive Bayes'].value_counts())

        print("\nContagem de Sentimentos Preditos (SGD Classifier):")
        print(df['SGD Classifier'].value_counts())

    else:
        print("Nenhum dado coletado.")

    # O servidor Flask será encerrado automaticamente ao sair
    print("\nProcessamento concluído.")
    sys.exit(0)  # Encerra o programa após finalizar

if __name__ == "__main__":
    main()
