from google import genai
import json

client = genai.Client(api_key="AIzaSyDzvAkCOTlBsqcEHPHeEeVSBw-iFc0kgvk")

medicamentos = ['novorapid', 'basaglar']
bolus_alimentar = 15 #15g de carboidrato por 1 unidade de insulina
bolus_correcao = 60 #60mg/dL por 1 unidade de insulina
glicemia_atual = 132

descricao_alimentacao = """
hoje de manhã comi um pão frances com uma fatia de queijo prato mais um ovo cozido. Também tomei um café com leite, mais café do que leite.
"""

def montar_json(medicamentos, bolus_alimentar, bolus_correcao, glicemia_atual, descricao_alimentacao):
  # Cria um dicionário Python com os dados
  dados = {
      "medicamentos": medicamentos,
      "parametros_insulina": {
          "bolus_alimentar": f"1/{bolus_alimentar}g",
          "bolus_correcao": f"1/{bolus_correcao}mg/dL"
      },
      "medicoes": {
          "glicemia_atual_mg_dl": glicemia_atual
      },
      "alimentacao": {
          "descricao": descricao_alimentacao.strip()
      }
  }

  # Converte o dicionário para uma string JSON formatada (com indentação para melhor leitura)
  return json.dumps(dados, indent=4, ensure_ascii=False)

contexto_json =  montar_json(medicamentos, bolus_alimentar, bolus_correcao, glicemia_atual, descricao_alimentacao)

# print(contexto_json)

papel_esperado_ia = """
Você é nutricionista especializada em contagem de carboidratos, calorias e cálculo da insulina necessária com base na alimentação e nos níveis de glicose informados
"""

resposta = """
voce deve retornar SOMENTE um JSON contendo nome do alimento, quantidade de carboidrato, quantidade de caloria, quantidade de glicemia enviada e a quantidade de insulina necessária, tendo como base o JS

"""

prompt = f"""
Papel:
{papel_esperado_ia}

Contexto:
texto contextualizando o JSON de entrada.....
{contexto_json}

Resposta:
{resposta}
"""

response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
)

resposta_json = response.candidates[0].content.parts[0].text.strip()

# Função para calcular os tokens
# palavras = pergunta.split()
# palavras += contexto.split()
# palavras += json.split()
# palavras += str(response).split()
# num_palavras = len(palavras)
# # Estimativa: 1 palavra ≈ 1.33 tokens (média comum)
# tokens_estimados = int(num_palavras * 1.33)
# print(f"Tokens estimados: {tokens_estimados}")

# print(resposta_json)

def desmontar_json(resposta_json_str):
  try:
    # 1. Converte a string JSON para um dicionário Python
    dados = json.loads(resposta_json_str)

    # 2. Extrai os valores das chaves esperadas
    # Extrai apenas os nomes dos alimentos para uma lista
    alimentos_detalhados = dados.get("analise_refeicao", {}).get("alimentos", [])
    lista_alimentos = [alimento.get("nome", "N/A") for alimento in alimentos_detalhados]

    # Extrai os totais do cálculo
    calculo = dados.get("calculo_insulina", {})
    total_calorias = calculo.get("total_calorias_kcal")
    total_carboidratos = calculo.get("total_carboidratos_g")
    qtd_insulina = calculo.get("unidades_insulina_necessarias")

    # 3. Retorna os valores extraídos
    return lista_alimentos, total_calorias, total_carboidratos, qtd_insulina

  except json.JSONDecodeError:
    print("Erro: A resposta recebida não é um JSON válido.")
    return None, None, None, None
  except KeyError as e:
    print(f"Erro: A chave {e} não foi encontrada no JSON. A estrutura da resposta da IA pode ter mudado.")
    return None, None, None, None

lista_alimentos, calorias, carboidratos, qtd_insulina = desmontar_json(resposta_json)

if lista_alimentos is not None:
    print("--- Dados Extraídos ---")
    print(f"Lista de Alimentos: {lista_alimentos}")
    print(f"Total de Calorias (kcal): {calorias}")
    print(f"Total de Carboidratos (g): {carboidratos}")
    print(f"Insulina Necessária (unidades): {qtd_insulina}")
