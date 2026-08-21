# Datasets recomendados

O projeto inclui apenas um conjunto inicial pequeno, em português, para que o núcleo seja executável sem downloads. Para experimentos maiores, os melhores encaixes são:

| Dataset | Escopo | Uso no Lynx | Observação |
|---|---:|---|---|
| MASSIVE 1.1 | Mais de 1 milhão de frases, 52 idiomas, 60 intenções e 55 tipos de slots | Português, intenção e argumentos de ferramentas | Melhor candidato multilíngue; filtrar `pt-PT` e adaptar localização para `pt-BR` |
| CLINC150 | 150 intenções em 10 domínios, mais exemplos fora de escopo | Roteamento amplo e calibração de rejeição | Excelente para medir `out_of_scope` |
| Banking77 | 13.083 consultas e 77 intenções bancárias | Teste de intenções semanticamente próximas | Bom teste de granularidade, mas domínio restrito |
| SNIPS NLU | Intenções e slots de assistente | Chamadas de função e extração de argumentos | Pequeno e adequado a protótipos locais |
| ATIS | Consultas aéreas com intenções e slots | Teste clássico de composição e slots | Domínio estreito e linguagem antiga |
| [NLP-CISUC/RelEx-PT](https://huggingface.co/datasets/NLP-CISUC/RelEx-PT) | 1.800 frases PT com sujeito, relação e objeto | Memória factual e avaliação de “X é Y” | JSONL pequeno; usado no `train_brain.py` |
| [Jpzinn654/qa-portuguese-small](https://huggingface.co/datasets/Jpzinn654/qa-portuguese-small) | 500 mil perguntas, contexto e resposta | Expansão de perguntas factuais | MIT; Parquet, requer `pyarrow` para leitura local |
| [rishiraj/portuguesechat](https://huggingface.co/datasets/rishiraj/portuguesechat) | ~9,5 mil conversas curtas | Ajuste de respostas controladas | CC-BY-NC-4.0; não usar comercialmente sem revisar licença |
| [nicholasKluge/Pt-Corpus-Instruct](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct) | Corpus instrucional muito grande | Pré-treino opcional por amostra | Grande demais para o perfil de 500 MB; amostrar, não baixar inteiro |

Fontes primárias/oficiais:

- MASSIVE: https://github.com/alexa/massive e https://arxiv.org/abs/2204.08582
- CLINC150: https://github.com/clinc/oos-eval e https://aclanthology.org/D19-1131/
- Banking77: https://github.com/PolyAI-LDN/task-specific-datasets
- SNIPS NLU: https://github.com/snipsco/snips-nlu
- RelEx-PT: https://huggingface.co/datasets/NLP-CISUC/RelEx-PT

Antes de combinar conjuntos, converta cada rótulo para uma taxonomia única `chat.*`/`tool.*`, mantenha divisões oficiais de treino e teste e verifique licenças e possíveis erros de rótulo. O runtime não consulta esses datasets durante inferência.
