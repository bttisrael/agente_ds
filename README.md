# Auto Data Scientist - Pipeline End-to-End

> Pipeline autonomo de Data Science gerado por agentes CrewAI + Llama 3.1

---

## 1. Entendimento do Negocio

Dataset de triagem de curriculos com ~200k candidatos.
Objetivo: extrair insights e construir um modelo preditivo automaticamente.

**Fonte:** [Kaggle - Resume Screening Dataset](https://www.kaggle.com/datasets/rhythmghai/resume-screening-dataset-200k-candidates)

---

## 2. Preparacao dos Dados (Arquitetura Medalhao)

| Camada   | Arquivo              | Descricao                          |
|----------|----------------------|------------------------------------|
| Silver   | df1_silver.parquet   | Dados brutos limpos e padronizados |
| Gold     | df2_gold.parquet     | Features de engenharia adicionadas |
| ML-Ready | df3_ml_ready.parquet | Colunas redundantes/IDs removidos  |

### Amostra do Dataset

![Amostra do Dataset](amostra_dataset.png)

Estatistica descritiva completa: [Estatistica_Descritiva.md](Estatistica_Descritiva.md)

---

## 3. Analise Exploratoria (EDA)

### Distribuicao das Variaveis

![Distribuicoes](distribuicoes.png)

### Boxplots

![Boxplots](boxplots.png)

### Correlacao entre Variaveis

![Matriz de Correlacao](matriz_correlacao.png)

---

## 4. Modelagem (Machine Learning)

# Metricas do Modelo

**Tipo:** Regressao
**Coluna Alvo:** `resume_length_words`
**RMSE:** 121.0317


### Features Mais Importantes

![Feature Importance](feature_importance.png)

---

## 5. Artefatos Gerados

| Artefato                  | Descricao                        |
|---------------------------|----------------------------------|
| modelo_final.pkl          | Modelo treinado (pickle)         |
| Metricas_Modelo.md        | Metricas de performance          |
| matriz_correlacao.png     | Mapa de calor de correlacao      |
| distribuicoes.png         | Histogramas das variaveis        |
| boxplots.png              | Boxplots das variaveis           |
| amostra_dataset.png       | Primeiras linhas do dataset      |
| feature_importance.png    | Importancia das features         |
| Estatistica_Descritiva.md | Estatisticas descritivas         |

---

## 6. Como Reproduzir
```bash
git clone <url-do-repo>
echo "KAGGLE_USERNAME=seu_usuario" >> .env
echo "KAGGLE_KEY=sua_chave"        >> .env
pip install crewai kagglehub pandas pyarrow python-dotenv scikit-learn matplotlib seaborn tabulate numpy
python ollama_ds_v4.py
```

---
*Gerado automaticamente pelo Auto Data Scientist - CrewAI + Llama 3.1*
