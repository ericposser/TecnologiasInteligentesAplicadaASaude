import json
from google import genai

# Configuração do cliente da Google Gemini
client = genai.Client(api_key="AIzaSyDzvAkCOTlBsqcEHPHeEeVSBw-iFc0kgvk")

 # Função para montar o JSON de entrada
def montar_json(medicamentos, bolus_alimentar, bolus_correcao, glicemia_atual, descricao_alimentacao):
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
    return json.dumps(dados, indent=4, ensure_ascii=False)

# Função para limpar a resposta da IA (remover blocos de crase e prefixos)
def limpar_json_da_ia(resposta_json):
    resposta_json = resposta_json.strip()
    if resposta_json.startswith("```"):
        resposta_json = resposta_json.strip("` \n")
        if resposta_json.startswith("json"):
            resposta_json = resposta_json[4:].lstrip()
    return resposta_json

# Função para desmontar o JSON de saída da IA
def desmontar_json(json_recebido):
    if isinstance(json_recebido, str):
        dados = json.loads(json_recebido)
    else:
        dados = json_recebido
    nome_alimento = dados.get("nome_do_alimento")
    carboidrato = dados.get("quantidade_de_carboidrato")
    caloria = dados.get("quantidade_de_caloria")
    glicemia_enviada = dados.get("quantidade_de_glicemia_enviada")
    insulina_necessaria = dados.get("quantidade_de_insulina_necessaria")
    return nome_alimento, carboidrato, caloria, glicemia_enviada, insulina_necessaria

# Função para estimar tokens de um texto (input/output)
def contar_tokens(texto):
    # Aproximação: 1 token ≈ 4 caracteres (português)
    return int(len(texto) / 4)

# Parâmetros de entrada
medicamentos = ['novorapid', 'basaglar']
bolus_alimentar = 15  # 15g de carboidrato por 1 unidade de insulina
bolus_correcao = 60   # 60mg/dL por 1 unidade de insulina
glicemia_atual = 132
descricao_alimentacao = """
hoje de manhã comi um pão frances com uma fatia de queijo prato mais um ovo cozido. Também tomei um café com leite, mais café do que leite.
"""

# Monta o JSON de contexto
contexto_json = montar_json(medicamentos, bolus_alimentar, bolus_correcao, glicemia_atual, descricao_alimentacao)

# Prompt para IA
papel_esperado_ia = "Nutricionista: calcule carboidratos, calorias e insulina pela alimentação e glicemia informada."
exemplo_json = '{"nome_do_alimento": "", "quantidade_de_carboidrato": , "quantidade_de_caloria": , "quantidade_de_glicemia_enviada": , "quantidade_de_insulina_necessaria": }'
resposta = "Responda apenas com esse JSON, sem comentários, sem blocos de código e com o nome de todos alimentos."
prompt = f"{papel_esperado_ia}\n{contexto_json}\n{exemplo_json}\n{resposta}"

tokens_input = contar_tokens(prompt)

# Chamada do modelo IA
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

# Limpa e desmonta o JSON de resposta da IA
resposta_json = response.candidates[0].content.parts[0].text.strip()
resposta_json_limpo = limpar_json_da_ia(resposta_json)

# Conta tokens de output (resposta recebida)
tokens_output = contar_tokens(resposta_json_limpo)
tokens_total = tokens_input + tokens_output

nome, carb, cal, glic, insu = desmontar_json(resposta_json_limpo)

# Mostra os resultados
print(f"Alimentos: {nome}")
print(f"Carboidratos: {carb}")
print(f"Calorias: {cal}")
print(f"Glicemia enviada: {glic}")
print(f"Insulina necessária: {insu}")

print("\nContagem de Tokens")
print(f"Tokens de input: {tokens_input}")
print(f"Tokens de output: {tokens_output}")
print(f"Total de tokens: {tokens_total}")
