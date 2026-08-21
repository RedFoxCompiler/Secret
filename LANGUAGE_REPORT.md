# Relatório do núcleo linguístico 0.2

## Configuração treinada

- Fonte: `FreedomIntelligence/alpaca-gpt4-portuguese`
- Amostra de treino: 5.000 conversas filtradas
- Validação: 200 conversas separadas
- Vocabulário: 6.000 tokens
- Reservatório máximo: 128 unidades
- Atualizações locais: 641.084
- Backpropagation: não utilizado
- Consulta ao dataset durante geração: não utilizada

## Interpretação correta das métricas

O fitness é o negativo da perda em um conjunto de candidatos formado pelas transições linguísticas compiladas. Não é perplexidade comparável à de um LLM sobre todo o vocabulário. Seu uso é interno: verificar se as gerações evolutivas melhoram a mesma função objetiva.

## Limitações observadas

O modelo aprende gramática local e consegue produzir trechos em português, porém ainda apresenta deriva temática e junções incoerentes. Respostas curtas são melhores. Fatos não devem ser confiados ao gerador; chamadas de ferramentas e conhecimento verificável continuam sob responsabilidade do núcleo agentic.

Por isso, a interface padrão usa uma camada controlada para reconhecer explicação, lista e resumo. O modo `--raw` permanece disponível para pesquisa e deixa visível a qualidade real do reservatório gerativo.

## Próximos experimentos

1. Memórias separadas para sintaxe, assunto e estrutura discursiva.
2. Evolução multiobjetivo com coerência semântica e diversidade.
3. Unidades recorrentes crescentes acionadas por surpresa.
4. Consolidação em duas velocidades para reduzir esquecimento.
5. Avaliação humana e benchmarks próprios em português.
