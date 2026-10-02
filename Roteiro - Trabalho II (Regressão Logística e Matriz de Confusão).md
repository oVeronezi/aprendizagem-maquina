# Roteiro do vídeo — Trabalho II: Regressão Logística e Matriz de Confusão

**Arquivo usado na gravação:** `Aula 6/regressao_logistica_matriz_confusao.ipynb` (dataset `titanic.csv`)
**Duração-alvo:** ~8 a 9 min (o enunciado exige entre 5 e 10 min; fique com folga abaixo de 10).
**Prazo:** 04/10/26 às 23:59 — mp4 ou mkv direto no Classroom (sem link, sem zip).

> Como usar: as partes em **negrito/citação** são o que falar, em linguagem natural. **Não leia palavra por palavra**: o critério "Domínio do Conteúdo e Naturalidade" (3 pts) penaliza leitura de texto. Use o roteiro como mapa e fale com as suas palavras. Os 🖥️ indicam o que deixar na tela.

---

## Antes de gravar (checklist rápido)

- [ ] Rodar o notebook inteiro do zero (Kernel → Restart & Run All) e conferir se os números batem com os deste roteiro (ver "Números de referência" no fim). Se o seu der diferente, use o seu.
- [ ] Gravação **contínua, sem cortes nem edição**, voz própria, sem acelerar (1,5 pt).
- [ ] Resolução mínima **720p**, fonte do notebook grande (zoom do navegador ~125–150%) para o código ficar legível (1,5 pt).
- [ ] Microfone testado, ambiente silencioso; OBS Studio com filtro de supressão de ruído (sugestão do professor).
- [ ] Fechar notificações/abas. Deixar a célula 7 (execução) já rodada, mas as células 1–6 visíveis para mostrar.
- [ ] Ter um cronômetro por perto (não precisa aparecer no vídeo).

---

## Mapa de tempo

| Bloco | Tempo | Atende ao enunciado |
|---|---|---|
| 1. Abertura e problema | 0:00–1:00 | Item 1 |
| 2. Como a regressão logística funciona | 1:00–3:00 | Item 1 e 2 (funcionamento e utilidade) |
| 3. Dados do Titanic e tratamento | 3:00–4:30 | Item 2 (justificar ferramentas) |
| 4. Treinamento (custo e gradiente) | 4:30–5:45 | Item 2 |
| 5. Matriz de confusão e métricas | 5:45–8:15 | Interpretação dos resultados |
| 6. Fechamento e relato pessoal | 8:15–9:00 | Item 3 |

---

## 1. Abertura e problema (0:00–1:00)

🖥️ Notebook aberto no topo, ainda sem rolar.

> "Olá, professor. Neste vídeo eu vou explicar como funciona a regressão logística, para que ela serve, e depois avaliar o modelo com a matriz de confusão, usando o dataset do Titanic."

> "O **problema** que eu escolhi resolver é: dado um passageiro do Titanic, com informações como classe da passagem, sexo, idade, quantos familiares estavam a bordo e quanto pagou, **o modelo consegue prever se ele sobreviveu ou não?**"

> "Repara que a resposta não é um número qualquer, como o preço de uma casa na regressão linear. A resposta é **uma entre duas classes**: sobreviveu (1) ou não sobreviveu (0). Isso é um problema de **classificação binária**, e é exatamente para isso que a regressão logística serve. Apesar do nome 'regressão', ela é um algoritmo de classificação."

Ideia para guardar: *a ideia/lógica da solução é calcular a **probabilidade** de sobreviver e depois decidir a classe a partir dela.*

---

## 2. Como a regressão logística funciona (1:00–3:00)

🖥️ Rolar até a célula 3 (`sigmoid`, `compute_cost`, `gradient_descent`). Se quiser, desenhe/mostre a fórmula da sigmoide no papel ou no slide, mas não é obrigatório.

> "Começando pela regressão linear: ela faz `z = β0 + β1·x1 + β2·x2 + ...`, uma soma ponderada das características. O problema é que esse `z` pode ser qualquer valor, de menos infinito a mais infinito, e eu preciso de uma **probabilidade**, que fica entre 0 e 1."

> "Por isso a regressão logística pega esse `z` e passa pela **função sigmoide**: `1 / (1 + e^-z)`. Ela 'espreme' qualquer número para dentro do intervalo de 0 a 1. Se `z` é muito positivo, o resultado chega perto de 1; se é muito negativo, perto de 0; e se `z` é zero, dá exatamente 0,5."

🖥️ Mostrar a função `sigmoid` no código (uma linha).

> "Com a probabilidade em mãos, eu preciso tomar uma decisão. Uso o **limiar de 0,5**: se a probabilidade for maior ou igual a 0,5, o modelo prevê que sobreviveu; se for menor, que não sobreviveu. Isso está na função `predict`."

🖥️ Mostrar a função `predict` (célula 5) rapidamente.

> "E **para que serve** isso na prática? Qualquer situação de decisão com duas saídas: aprovar ou negar crédito, e-mail spam ou não spam, paciente doente ou saudável, cliente que vai cancelar ou não. Além de dar a classe, ela dá a **probabilidade**, o que ajuda a entender o quão confiante o modelo está."

---

## 3. Dados do Titanic e tratamento (3:00–4:30)

🖥️ Células 1 e 2 (`load_data`, `fill_missing_age`, `normalize`, `add_bias`).

> "O dataset tem 1309 passageiros. A coluna que eu quero prever, o **alvo**, é `survived`. Como características eu escolhi seis: classe, sexo, idade, número de irmãos/cônjuges, número de pais/filhos e tarifa paga."

> "Antes de treinar, precisei tratar os dados, e cada passo tem um motivo:"

- **Sexo** → "O modelo só trabalha com números, então converti: feminino vira 1 e masculino vira 0."
- **Idade ausente** → "Faltavam cerca de 263 idades, uns 20% do dataset. Em vez de jogar essas linhas fora, eu preenchi com a **média** das idades conhecidas, para não perder dados."
- **Normalização** → "Idade vai de 0 a 80, a tarifa passa de 500, e o sexo é 0 ou 1. Em escalas tão diferentes, o gradiente descendente fica lento e instável. Então eu padronizei: subtraio a média e divido pelo desvio padrão, e todas as características ficam na mesma escala."
- **Bias (coluna de 1s)** → "Adiciono uma coluna de 1s para o modelo aprender o termo independente β0, que é o 'ponto de partida' da equação."

🖥️ Mostrar a função `train_test_split` (célula 6).

> "Por fim, separei os dados em **80% para treino e 20% para teste**: 1047 passageiros para o modelo aprender e 262 para testar. Isso é essencial, porque avaliar o modelo nos mesmos dados em que ele treinou seria como dar a prova com o gabarito: o resultado não diria nada sobre como ele se comporta com passageiros novos. Fixei a semente aleatória (seed 42) para o resultado ser reprodutível."

> "As ferramentas: **NumPy** para fazer as contas com vetores e matrizes de forma rápida, em vez de laços; e o módulo **csv** da biblioteca padrão para ler o arquivo."

---

## 4. Treinamento: custo e gradiente descendente (4:30–5:45)

🖥️ Célula 3 (`compute_cost`, `gradient_descent`) e depois a saída do treinamento (a lista `Epoch ... Cost`).

> "Treinar significa achar os valores de β que fazem o modelo errar o mínimo possível. Para medir o erro eu uso a **entropia cruzada**, também chamada de *log loss*. Ela pune muito forte quando o modelo está confiante e errado, por exemplo, diz 99% de chance de sobreviver e a pessoa morreu."

> "Por que não usar o erro quadrático médio, como na regressão linear? Porque, combinado com a sigmoide, ele cria uma superfície cheia de mínimos locais. A entropia cruzada deixa o problema bem comportado, com um único vale."

> "Para descer esse vale eu uso o **gradiente descendente**. A cada época ele calcula `X transposto vezes (h − y)`, que é o gradiente, e ajusta os pesos um pouquinho na direção contrária, controlado pela **taxa de aprendizado**."

🖥️ Apontar na saída o custo caindo: ~0,69 → ~0,45.

> "Dá para ver que o custo começa em cerca de 0,69, que é o 'chute', e cai até uns 0,45. Isso mostra que o modelo realmente aprendeu. Como o custo vai estabilizando nas últimas épocas, o modelo convergiu."

*(Opcional, 15 s, mostra domínio):* "Olhando os pesos aprendidos, o **sexo** tem o maior peso positivo, ser mulher aumenta muito a chance de sobreviver, e a **classe** tem peso negativo: quanto pior a classe, menor a chance. A idade também pesa contra. Isso bate com o que se sabe do naufrágio: 'mulheres e crianças primeiro'."

---

## 5. Matriz de confusão e métricas (5:45–8:15)

🖥️ Célula 7, saída da "MATRIZ DE CONFUSÃO" e das "MÉTRICAS DE AVALIAÇÃO". Se possível, desenhe/mostre a tabela 2x2 (slide da Aula 6).

> "Agora, como saber se esse modelo é bom? Só olhar a porcentagem de acerto esconde detalhes. A **matriz de confusão** mostra exatamente onde o modelo acerta e que tipo de erro comete. Aqui a classe positiva é 'sobreviveu'."

Explique os quatro quadrantes com os números reais (262 passageiros de teste):

> - "**Verdadeiros Positivos, 68**: o modelo disse que sobreviveu, e sobreviveu."
> - "**Verdadeiros Negativos, 123**: disse que não sobreviveu, e realmente não sobreviveu."
> - "**Falsos Positivos, 32**: disse que sobreviveu, mas morreu. É o alarme falso."
> - "**Falsos Negativos, 39**: disse que morreu, mas na verdade sobreviveu. É o caso que o modelo deixou passar."
>
> "68 + 123 + 32 + 39 dá 262, o tamanho do conjunto de teste, então as contas fecham."

Agora as métricas — **diga a fórmula em palavras, o número, e o que significa**:

> - "**Acurácia, 72,9%**: (68 + 123) / 262. De todos os passageiros, o modelo acertou cerca de 73%. É um bom resumo, mas a base é um pouco desbalanceada: só ~38% sobreviveram. Então um modelo que dissesse 'ninguém sobreviveu' já acertaria ~62%. Por isso olho as outras métricas também."
> - "**Precisão, 68%**: 68 / (68 + 32). Quando o modelo diz que a pessoa sobreviveu, ele acerta 68% das vezes. Importa quando o falso positivo é caro."
> - "**Revocação (sensibilidade), 63,6%**: 68 / (68 + 39). De todos que realmente sobreviveram, o modelo identificou cerca de 64%. Importa quando deixar passar um caso positivo é grave."
> - "**Especificidade, 79,4%**: 123 / (123 + 32). De todos que morreram, ele identificou corretamente 79%. Mostra que o modelo é melhor em reconhecer quem **não** sobreviveu."
> - "**F1-Score, 65,7%**: é a média harmônica entre precisão e revocação. Ela só é alta se as duas forem altas, então resume o equilíbrio entre elas num número só."

**Interpretação (fale com as suas palavras, esse é o ponto que mais vale):**

> "Interpretando: o modelo é **razoável**, com ~73% de acurácia usando só seis variáveis e uma regressão simples feita à mão. Ele é **mais conservador**: erra menos quando prevê 'não sobreviveu' (especificidade 79%) do que consegue encontrar quem sobreviveu (revocação 64%). Tem 39 falsos negativos contra 32 falsos positivos, ou seja, deixa escapar mais sobreviventes do que inventa. Se o objetivo fosse, por exemplo, não deixar nenhum sobrevivente de fora, eu poderia **baixar o limiar** de 0,5 para algo como 0,4, ganhando revocação, mas pagando com mais falsos positivos. Também dá para melhorar com mais variáveis, como o título no nome, ou o tratamento mais cuidadoso da idade."

---

## 6. Fechamento e relato pessoal (8:15–9:00) — item 3 obrigatório

🖥️ Pode voltar ao topo do notebook ou deixar a matriz na tela.

> "Para finalizar: o que eu **mais gostei** de fazer foi ______ (ex.: ver o custo caindo e o modelo realmente aprendendo; implementar a sigmoide e o gradiente na mão com NumPy e entender o que está por trás das bibliotecas; ver como a matriz de confusão muda a leitura do resultado)."

> "E no que eu **mais tive dificuldade** foi ______ (ex.: entender de onde vem a fórmula do gradiente; distinguir precisão de revocação; lidar com os dados faltantes e com a normalização; montar a matriz de confusão e não trocar falso positivo com falso negativo)."

> "É isso. Obrigado pela atenção!"

⚠️ **Preencha os dois espaços com o que é verdade para você.** O professor quer um relato breve e genuíno (2–3 frases cada); não deixe genérico demais. Eu não coloquei suas respostas por você de propósito.

---

## Pontos de atenção no notebook (leia antes de gravar)

Ao rodar e ler o código, vi dois detalhes que podem gerar pergunta ou confusão. Você decide se corrige antes de gravar ou se apenas comenta no vídeo:

1. **`train()` ignora `lr` e `epochs`.** Na célula 7 a chamada é `train(X_train, y_train, 0.1, 500)`, mas dentro de `train` o gradiente é chamado com `lr=0.01, epochs=2000` fixos. Por isso a saída mostra épocas até 1900. *Correção simples:* trocar por `gradient_descent(X, y, beta, lr=lr, epochs=epochs)`. Se corrigir, rode de novo e **use os números novos** no vídeo (com lr=0,1 e 500 épocas a acurácia fica ~73,7%, VN=125 e FP=30).
2. **O teste é normalizado com a média/desvio do próprio teste**, e o comentário do código diz "MESMAS transformações". O correto seria usar a média e o desvio **do treino** (evita vazamento de informação do teste). O impacto no resultado é pequeno (73,3% vs 72,9%), mas se alguém perguntar, a resposta é essa. Se não quiser mexer no código, mencione brevemente que "o ideal é reaproveitar as estatísticas do treino".

Outros detalhes: `calculate_metrics` divide sem proteger contra zero (não é problema aqui); e as linhas comentadas no final (exemplo do "Pedro") não precisam aparecer.

---

## Números de referência (saída salva do notebook)

- 1309 passageiros, ~38,2% sobreviveram; 263 idades ausentes.
- Treino 1047 / teste 262 (107 sobreviventes e 155 não-sobreviventes no teste).
- Custo: 0,6918 (época 0) → ~0,4501 (época 1900).
- Matriz: **VP 68 · VN 123 · FP 32 · FN 39**.
- Acurácia **72,9%** · Precisão **68,0%** · Revocação **63,6%** · Especificidade **79,4%** · F1 **65,7%**.
- Pesos (bias, pclass, sex, age, sibsp, parch, fare), com lr=0,1/500 épocas: −0,62; −0,62; +1,20; −0,31; −0,26; −0,01; +0,23.

## Checklist final contra o enunciado

- [ ] Explicou o problema e a lógica da solução (bloco 1 e 2)
- [ ] Explicou o funcionamento **e a utilidade** da regressão logística (bloco 2)
- [ ] Usou o `titanic.csv` nos exemplos (blocos 3 a 5)
- [ ] Montou a matriz de confusão, calculou as 5 métricas e **interpretou** (bloco 5)
- [ ] Justificou as ferramentas: NumPy, csv, sigmoide, entropia cruzada, gradiente descendente, normalização, bias, split treino/teste, limiar 0,5, matriz e métricas (blocos 2 a 5)
- [ ] Relato final: o que mais gostou e maior dificuldade (bloco 6)
- [ ] Entre 5 e 10 min, contínuo, voz própria, 720p+, áudio limpo
- [ ] (Opcional) Trabalho salva-vidas: seguir as instruções da Aula 1 e enviar o link do LinkedIn
