# Relatorio de Qualidade — Analise por IA

**Contexto:** Dataset de triagem de curriculos.
**Shape:** 200000 x 17

## Imputacao Aplicada
- Nenhuma imputacao necessaria.

## Outliers Detectados (IQR)
{
  "cgpa": 1456,
  "internships": 13172,
  "projects": 2413,
  "certifications": 119,
  "experience_years": 9470,
  "hackathons": 2909,
  "research_papers": 36609,
  "skills_score": 1219,
  "resume_length_words": 1423
}

## Analise Inteligente por Claude

### Target Identificado
**Coluna:** `hired`
**Justificativa:** A coluna 'hired' é claramente o target pois é binária (0/1), representa o resultado do processo de triagem (contratado ou não), e tem uma taxa de contratação de 70.61%, o que indica um processo de seleção. Todas as outras features descrevem características dos candidatos que seriam usadas para prever esta decisão.

### Colunas Problematicas
['candidate_id', 'resume_length_words']

### Top Insights do Dataset
1. Dataset desbalanceado com 70.61% de contratações - pode necessitar ajuste de threshold ou técnicas de balanceamento para otimizar precisão vs recall
2. ALERTA CRÍTICO: resume_length_words tem valores negativos (min=-79), indicando erro grave na coleta de dados que precisa investigação imediata
3. Experiência profissional é altamente skewed (skew=2.02, max=23.55 anos) com média de 1.5 anos - sugere população predominantemente júnior com outliers sênior que podem distorcer modelos
4. Features acadêmicas (research_papers, hackathons) são extremamente esparsas (médias de 0.20 e 0.50) - considerar binarização ou agrupamento para evitar overfitting
5. CGPA possui range anômalo (4.15 a 11.23) sugerindo mistura de escalas diferentes (10.0 vs 4.0) entre universidades - necessita normalização por university_tier

### Estrategia de Feature Engineering Recomendada
1) LIMPEZA URGENTE: Investigar e corrigir valores negativos em resume_length_words. 2) NORMALIZAÇÃO: Padronizar CGPA por university_tier (detectar escala 4.0 vs 10.0) e aplicar StandardScaler em features com skew>1 (experience_years, certifications, hackathons). 3) FEATURE ENGINEERING: Criar 'experience_category' (Junior/Mid/Senior), 'academic_excellence' combinando cgpa+research_papers+university_tier, 'hands_on_score' agregando internships+projects+hackathons, 'technical_breadth' ratio entre programming_languages e certifications. 4) ENCODING: OneHotEncoder para education_level, company_type e university_tier (baixa cardinalidade). 5) BINNING: Agrupar features esparsas (research_papers: 0 vs 1+, hackathons: 0 vs 1-2 vs 3+). 6) INTERAÇÕES: Testar cgpa*university_tier, experience_years*education_level, skills_score*soft_skills_score. 7) TRATAMENTO DE DESBALANCEAMENTO: Aplicar SMOTE ou ajustar class_weight dado 70.61% de hired=1, priorizando recall se custo de falso negativo for alto.

### Output da Analise Executada
```
=== Estatísticas por Education Level ===
                 hired           cgpa experience_years skills_score internships
                  mean   count   mean             mean         mean        mean
education_level                                                                
Bachelors        0.707  129827  7.501            1.501       13.999       1.504
Masters          0.705   60333  7.496            1.494       14.009       1.500
PhD              0.707    9840  7.505            1.503       13.979       1.501

=== Distribuição do Target (hired) ===
hired
1    0.70606
0    0.29394
Name: proportion, dtype: float64

Taxa de contratação: 70.61%

=== Top 10 Correlações com Target ===
hired                    1.000000
experience_years         0.074603
internships              0.047885
skills_score             0.045687
projects                 0.035384
programming_languages    0.025373
certifications           0.012352
cgpa                     0.012175
hackathons               0.007092
candidate_id             0.001460
Name: hired, dtype: float64

Gráfico salvo em: C:\Users\israb\Documents\Agente_RPA\analise_inteligente.png

```

---
*Analise gerada por Claude 3.5 Sonnet*
