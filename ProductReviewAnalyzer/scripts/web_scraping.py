import requests
from bs4 import BeautifulSoup

# Função para realizar o web scraping
def coletar_avaliacoes(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')
        avaliacoes = soup.find_all('div', class_='avaliacao')
        
        avaliacoes_originais = []  # Armazenar texto e nota originais

        for avaliacao in avaliacoes:
            texto_tag = avaliacao.find('p', class_='texto-avaliacao')
            nota_tag = avaliacao.find('span', class_='nota-avaliacao')

            # Verificar se ambos os elementos foram encontrados
            if texto_tag is None or nota_tag is None:
                print(f"Aviso: Avaliação com estrutura inválida ignorada. Conteúdo: {avaliacao}")
                continue  # Pular para a próxima avaliação

            texto_avaliacao = texto_tag.get_text(strip=True)
            nota_avaliacao = nota_tag.get_text(strip=True)
            
            # Armazenar o texto e a nota originais
            avaliacoes_originais.append({
                "texto": texto_avaliacao,
                "nota": nota_avaliacao
            })

        return avaliacoes_originais
    else:
        print(f"Falha ao acessar a página: {response.status_code}")
        return None