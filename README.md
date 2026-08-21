# Lynx NumPy 0.4

Núcleo funcional de compreensão de frases e roteamento agentic inspirado na arquitetura conceitual do PDF *LynxSparsity - Volume I*. Usa NumPy e biblioteca padrão; não usa Transformer, atenção neural, autodiferenciação, backpropagation, PyTorch ou TensorFlow.

## O que existe de verdade

- codificação semântica incremental por Random Indexing, coocorrência local, subpalavras e ordem curta;
- ativação esparsa `k-winners`;
- rede de conceitos inspirada em Growing Neural Gas, com crescimento por novidade, conexões, envelhecimento e esquecimento;
- rede de largura variável treinada por população, fitness, elitismo, cruzamento, mutação de pesos e mutação estrutural;
- roteamento de intenções e chamadas de função estruturadas, com parâmetros obrigatórios e execução por lista explícita de permissões;
- aprendizado online local, salvamento em `.npz`, testes e métricas reproduzíveis.

Não é um LLM. O núcleo agentic original continua sendo um NLU compacto; o módulo linguístico novo acrescenta geração experimental limitada. “Evolução” significa melhoria mensurável de fitness e alteração de topologia; não implica consciência ou evolução aberta ilimitada.

## Cérebro híbrido: sentido, MTP e agente

A versão 0.3 não tenta mascarar n-grams como escrita fluente. Ela combina o gerador experimental com um caminho controlado para tarefas em que coerência importa:

- `FactGraph`: relações `sujeito — relação — objeto` consolidadas; uma pergunta como “o que é X?” responde pela memória, sem reabrir o dataset;
- `MultiTokenPredictor` (MTP): compila transições de blocos de 1–3 tokens e expõe a previsão, contexto e confiança; não é atenção;
- roteador evolutivo + âncoras de alta precisão para calcular, hora, pesquisa e leitura local;
- `data/agentic_like_v2.jsonl`: 204 exemplos auditáveis com `available_functions`, `reasoning_summary` curto e `function_call`; não armazena cadeia de pensamento privada;
- `data/language_programming_v3.jsonl`: 210 exemplos em português/inglês para classificação de programação e explicações; o compositor gera esqueletos Python válidos, incluindo programas longos de Fibonacci e leitura de arquivo;
- limite verificável de 500 MB no `manifest.json`. O cérebro treinado distribuído ocupa cerca de 3,3 MB.

## Núcleo de linguagem experimental

A versão 0.2 acrescenta geração condicionada em português sem Transformers e sem backpropagation:

- tokenizador de palavras e pontuação;
- reservatório recorrente esparso com conexões fixas por permutações;
- associação local estado-próximo-token do tipo Hebb;
- memória linguística n-gram de ordem variável, compilada em parâmetros;
- associação comprimida entre palavras da instrução e palavras da resposta;
- evolução do decodificador, temperatura e quantidade de unidades ativas;
- geração sem reabrir ou pesquisar exemplos do dataset.

O comando normal usa uma camada linguística controlada para explicações, listas e resumos. `--raw` expõe o gerador recorrente puro, que produz frases locais razoáveis, mas ainda pode perder o assunto ou combinar trechos incompatíveis. Ele é um experimento de linguagem compacta, não substitui um modelo generativo moderno.

## Relação com o anexo

| Princípio do PDF | Implementação |
|---|---|
| surpresa e erro de previsão | distância/novidade que cria novos nós |
| dinâmica esparsa | camada `k-winners` e índices esparsos |
| formação de conceitos | topologia Growing Concept Network |
| memória semântica e consolidação | vetores incrementais persistidos no modelo |
| esquecimento seletivo | utilidade, idade das arestas e poda |
| simulação/seleção | população de candidatos avaliada antes da sobrevivência |
| evolução cognitiva aberta | aproximação limitada: mutação estrutural e aprendizado online |

O livro é uma taxonomia extensa de princípios cognitivos, não pseudocódigo validado. A implementação escolhe mecanismos verificáveis compatíveis com esses princípios.

## Executar

```bash
python -m unittest discover -s tests -v
python train.py --generations 70
python demo.py "pesquise sobre redes neurais evolutivas"
python demo.py "como está o tempo em Palmas?"
python train_language.py --download --examples 1600 --validation 120
python chat_language.py "Explique de forma simples o que é energia solar"
python chat_language.py --raw "Continue esta frase"
python train_brain.py --download --generations 32
python lynx_cli.py --auto "calcule 13 * 7"
python lynx_cli.py --auto "que horas são?"
python lynx_cli.py --auto --allow-network "pesquise sobre redes neurais evolutivas"
python lynx_cli.py --allow-network --learn-url "https://exemplo.org/fonte"
```

Exemplo de plano:

```json
{
  "type": "function_call",
  "name": "web_search",
  "arguments": {"query": "redes neurais evolutivas"},
  "confidence": 0.71
}
```

A IA apenas produz o plano. `execute()` exige que a ferramenta esteja registrada no modelo e que uma função executora seja fornecida explicitamente pelo aplicativo hospedeiro.

Na CLI, `--auto` executa ferramentas permitidas; `--allow-network` libera apenas a pesquisa web de leitura. `:teach sujeito | relação | objeto` adiciona um fato à memória local. A execução contínua existe apenas enquanto a CLI está aberta: ela não deve realizar ações externas ou persistentes sem configuração e confirmação explícitas.

## Aprendizado externo

`web_search` usa Google através de `curl`, em modo somente leitura. Google pode devolver CAPTCHA, uma página reduzida ou nenhum trecho útil; o projeto não tenta contornar bloqueios. Para aprendizagem externa confiável, forneça uma fonte escolhida: `--allow-network --learn-url URL` lê o texto da página e só consolida sentenças simples do tipo `X é Y` ou `X is Y`, registrando a URL como origem. Depois disso, a inferência usa a memória persistida, não a página nem o dataset.

## Como treina sem backpropagation

1. Coocorrências atualizam vetores de palavras com uma regra local.
2. Uma projeção esparsa transforma frases em estados compactos.
3. A rede topológica aprende protótipos com competição local tipo Hebb.
4. Uma população de redes recebe fitness de acurácia, confiança, esparsidade e custo estrutural.
5. Elites sobrevivem; cruzamento e mutações produzem a geração seguinte.
6. O melhor genoma e a topologia consolidada são persistidos. Na inferência não há busca nos exemplos de treino.

As bases metodológicas incluem Growing Neural Gas de Fritzke (NeurIPS 1994), Random Indexing de Sahlgren e estratégias evolutivas como alternativa escalável a métodos baseados em gradiente. Consulte `DATASETS.md` para dados externos avaliados.

## Limitações e próximos passos

- O conjunto inicial tem poucas frases e serve como teste de funcionamento, não como treinamento de produção.
- Extração de argumentos é híbrida e determinística; a classificação da intenção é aprendida.
- Aprendizado online atualiza semântica e conceitos, mas uma nova classe exige nova rodada evolutiva.
- Próxima iteração recomendada: MASSIVE em português, calibração de `out_of_scope`, avaliação de robustez, evolução multiobjetivo e snapshots de memória lenta/rápida.

## Dataset linguístico

O importador suporta o JSON `conversations` de `FreedomIntelligence/alpaca-gpt4-portuguese`. A página informa 49.969 registros em uma única divisão de treino, mas não declara licença nos metadados do dataset. Por isso, o corpus completo não é redistribuído no pacote. O comando `--download` obtém o arquivo da fonte para treinamento local.
