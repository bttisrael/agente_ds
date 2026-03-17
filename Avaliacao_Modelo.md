# Avaliacao do Modelo

## `GradientBoosting`
**Tipo:** classificacao | **Alvo:** `hired`

| Conjunto | Accuracy |
|----------|-------|
| Treino   | 0.7062 |
| Teste    | 0.7060 |
| Gap      | 0.0001  |

## Diagnostico por IA

## Diagnóstico do Modelo

O modelo está **adequadamente ajustado**. O gap mínimo de 0.0001 entre os desempenhos de treino e teste indica que não há overfitting - o modelo está generalizando bem para dados não vistos. A performance praticamente idêntica em ambos os conjuntos demonstra estabilidade e capacidade de generalização.

Porém, há sinais de **underfitting leve**. Uma acurácia de ~70% sugere que o modelo pode não estar capturando toda a complexidade dos dados. Para um problema de contratação, isso significa que 30% das decisões estão sendo classificadas incorretamente. Recomendo aumentar a complexidade do modelo (mais estimadores, profundidade maior) ou realizar engenharia de features para melhorar a performance geral.

## Parametros Otimizados (Optuna)
```json
{
  "n_estimators": 189,
  "learning_rate": 0.03272599711177244,
  "max_depth": 4,
  "subsample": 0.6418997025052154
}
```
