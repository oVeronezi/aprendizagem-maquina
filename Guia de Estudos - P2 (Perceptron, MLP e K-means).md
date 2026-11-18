# Guia de Estudos para a P2 — Perceptron, MLP e K-means

> Material de consulta consolidando o conteúdo das **Aula 10 (Perceptron)**, **Aula 11 (MLP)** e
> **Aula 12 (K-means)**, no mesmo padrão do `Guia de Estudos - Prova.md` (Aulas 2 a 7): explicações
> detalhadas, exemplos numéricos resolvidos passo a passo, código no estilo usado em aula,
> exercícios com gabarito e um **guia de decisão**. No final há uma seção de **orientação para a
> P2 de 2025** (pasta `p2-2025/`): ela **não traz as respostas** — traz perguntas, checklists e
> dicas progressivas para você chegar nas soluções sozinho.

## Como usar este guia

Cada seção de algoritmo segue a mesma estrutura: **quando usar → fórmulas → exemplo resolvido
passo a passo → código no estilo usado em aula → exercícios com gabarito**.

Ordem sugerida de estudo:

1. Leia as Seções 1 a 3 (um algoritmo por vez) e refaça os exemplos numéricos **à mão**.
2. Faça os exercícios de cada seção **antes** de abrir o gabarito.
3. Leia a Seção 4 (Guia de Decisão) — na prova, "qual algoritmo usar?" é metade da nota.
4. Só então vá para a Seção 5 e tente resolver a P2 2025 usando as dicas progressivas.
5. Na véspera, releia a Seção 6 (Cola Final).

> **Atenção:** a prova é **offline, com consulta ao seu material** e você entrega um *Jupyter
> Notebook + os CSVs*. Ou seja: ter as classes `Perceptron`, `MLPClassifier` e a função `kmeans`
> **prontas, entendidas e testadas** vale mais do que decorar fórmulas.

---

## Sumário

1. [Aula 10 — Perceptron](#1-aula-10--perceptron)
2. [Aula 11 — MLP (Multilayer Perceptron)](#2-aula-11--mlp-multilayer-perceptron)
3. [Aula 12 — K-means](#3-aula-12--k-means)
4. [Guia de Decisão — Perceptron × MLP × K-means](#4-guia-de-decisão--perceptron--mlp--k-means)
5. [Resolvendo a P2 2025 — orientação sem respostas](#5-resolvendo-a-p2-2025--orientação-sem-respostas)
6. [Cola Final — fórmulas e hiperparâmetros](#6-cola-final--fórmulas-e-hiperparâmetros)

---

## 1. Aula 10 — Perceptron

### 1.1 Quando usar

O Perceptron é o **neurônio artificial mais simples**: um classificador **binário** e **linear**.
Use quando:

- O problema é de **classificação com 2 classes** (supervisionado — os dados têm rótulo).
- Os dados são (ou parecem ser) **linearmente separáveis**: dá para separar as duas classes com
  **uma reta** (2 features), **um plano** (3 features) ou um **hiperplano** (mais features).
- O enunciado pede o **modelo mais simples**, rápido e barato computacionalmente.

Quando **não** usar: se as classes formam círculos, espirais, "xadrez" (XOR) ou qualquer padrão
em que uma única reta não separa, o Perceptron **nunca** zera o erro — aí o caminho é a MLP
(Seção 2).

### 1.2 O modelo

Para um vetor de entrada $x = (x_1, x_2, \dots, x_n)$, pesos $w = (w_1, \dots, w_n)$ e bias $b$:

$$z = w \cdot x + b = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$

$$\hat{y} = \text{sign}(z) = \begin{cases} +1 & z > 0 \\ -1 & z < 0 \\ 0 & z = 0 \end{cases}$$

- A **fronteira de decisão** é o conjunto de pontos com $z = 0$: em 2D, a reta
  $w_1x_1 + w_2x_2 + b = 0$.
- O vetor $w$ é **perpendicular** à reta e aponta para o lado da classe $+1$.
- O **bias** $b$ desloca a reta (sem ele a reta é obrigada a passar pela origem).

> **Pegadinha do código da aula:** `np.sign(0)` devolve **0**, que nunca é igual a $-1$ nem a
> $+1$. Logo, um ponto exatamente sobre a fronteira ($z=0$) **conta como erro** e provoca
> atualização. Isso acontece de verdade no início do treino com pesos zerados (veja o exemplo 1.4).

### 1.3 A regra de aprendizado

Para cada amostra $(x, y)$ com $y \in \{-1, +1\}$, em cada época:

1. Calcule $\hat{y} = \text{sign}(w \cdot x + b)$.
2. **Se $\hat{y} = y$: não faz nada.**
3. **Se $\hat{y} \neq y$: corrige**

$$w \leftarrow w + \eta \cdot y \cdot x \qquad\qquad b \leftarrow b + \eta \cdot y$$

onde $\eta$ é a **taxa de aprendizado** (`learning_rate`).

Por que funciona? Se a amostra é $y=+1$ e foi classificada como $-1$, somar $\eta x$ ao peso
**aumenta** $w\cdot x$ na próxima vez (empurra a fronteira para classificar certo). Se é $y=-1$ e
foi classificada como $+1$, subtrair $\eta x$ **diminui** $w\cdot x$.

**Conceitos-chave:**

| Termo | Significado |
|---|---|
| **Época** | Uma passada completa por **todas** as amostras do treino |
| **Taxa de erro da época** | `erros_na_época / n_amostras` (é o que a aula plota em `plot_error`) |
| **Convergência** | Uma época inteira sem nenhum erro → o algoritmo para (*parada antecipada*) |
| **Teorema de convergência** | Se os dados **são** linearmente separáveis, o Perceptron converge em número finito de atualizações |
| **Dados não separáveis** | O erro oscila e **nunca** chega a 0; o treino só para em `n_epochs` |
| **Rótulos** | Precisam ser $-1$ e $+1$. Se o CSV tem 0/1, converta: `y = np.where(y == 1, 1, -1)` |
| **Inicialização** | `random`, `zeros` ou `normal` (pequenos valores) — afeta o caminho, não o resultado final em dados separáveis |

**Taxa de aprendizado $\eta$ no Perceptron:** com pesos iniciais **zero**, $\eta$ só muda a
*escala* de $w$ e $b$ (a reta final é a mesma); com pesos iniciais aleatórios ele influencia o
caminho. Um $\eta$ enorme não "quebra" o Perceptron como quebraria uma regressão por gradiente,
mas pode fazer a reta "pular" de um lado a outro enquanto o erro oscila.

### 1.4 Exemplo resolvido passo a passo

Dados (2 features), $\eta = 1$, pesos e bias iniciais zero ($w=(0,0)$, $b=0$):

| Amostra | $x_1$ | $x_2$ | $y$ |
|---|---|---|---|
| A | −3 | −3 | −1 |
| B | 0 | −2 | +1 |
| C | −3 | 4 | −1 |
| D | 4 | −2 | +1 |

**Época 1** (começa com $w=(0,0)$, $b=0$):

| Passo | $z = w\cdot x + b$ | $\hat y$ | Erro? | Atualização | $w$ depois | $b$ depois |
|---|---|---|---|---|---|---|
| A | $0$ | 0 | **Sim** ($0 \neq -1$) | $w \mathrel{+}= (-1)\cdot(-3,-3)$ ; $b \mathrel{+}= (-1)$ | $(3;\ 3)$ | $-1$ |
| B | $3\cdot0 + 3\cdot(-2) - 1 = -7$ | −1 | **Sim** (era +1) | $w \mathrel{+}= (+1)\cdot(0,-2)$ ; $b \mathrel{+}= 1$ | $(3;\ 1)$ | $0$ |
| C | $3\cdot(-3) + 1\cdot4 + 0 = -5$ | −1 | Não | — | $(3;\ 1)$ | $0$ |
| D | $3\cdot4 + 1\cdot(-2) + 0 = 10$ | +1 | Não | — | $(3;\ 1)$ | $0$ |

Erros na época 1: **2** de 4 → taxa de erro = **50 %**.

**Época 2** (começa com $w=(3;1)$, $b=0$):

| Passo | $z$ | $\hat y$ | Erro? | $w$ depois | $b$ depois |
|---|---|---|---|---|---|
| A | $-9-3+0 = -12$ | −1 | Não | $(3;\ 1)$ | $0$ |
| B | $0-2+0 = -2$ | −1 | **Sim** (era +1) | $(3;\ -1)$ | $1$ |
| C | $-9-4+1 = -12$ | −1 | Não | $(3;\ -1)$ | $1$ |
| D | $12+2+1 = 15$ | +1 | Não | $(3;\ -1)$ | $1$ |

Erros na época 2: **1** de 4 → taxa de erro = **25 %**.

**Época 3** (começa com $w=(3;-1)$, $b=1$): A → $z=-5$ (−1 ✔) · B → $z=3$ (+1 ✔) ·
C → $z=-12$ (−1 ✔) · D → $z=15$ (+1 ✔). **Zero erros → o treino para** (`errors == 0`).

Evolução da curva de erro (o que `plot_error()` desenha): $0{,}50 \to 0{,}25 \to 0$.

**Reta final:** $3x_1 - x_2 + 1 = 0$, isto é, $x_2 = 3x_1 + 1$. B e D ($y=+1$) ficam **abaixo**
da reta e A e C ($y=-1$) ficam **acima** — separação perfeita.

> **Lição 1:** o Perceptron corrige **somente quando erra**; cada correção é proporcional a $\eta$
> e à amostra errada, e a reta vai "girando" até separar tudo.
> **Lição 2:** no primeiro passo, com pesos zerados, $z=0$ e $\text{sign}(0)=0$ — conta como erro
> (veja a pegadinha da Seção 1.2).

### 1.5 Código (estilo usado em aula — só Numpy)

Arquivo: `aula10/perceptron.ipynb`. Resumo do que cada parte faz:

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=1000, weight_init="random"):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weight_init = weight_init
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []            # taxa de erro de cada época

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)  # random / zeros / normal
        self.bias = 0                         # (a aula zera o bias aqui, depois de inicializar)

        for epoch in range(self.n_epochs):
            errors = 0
            for i in range(n_samples):                       # uma amostra por vez
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = np.sign(linear_output)
                if y_pred != y[i]:                           # só atualiza se errou
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias    += self.learning_rate * y[i]
                    errors += 1
            self.errors_per_epoch.append(errors / n_samples)
            if errors == 0:                                  # parada antecipada
                break

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)
```

Uso típico na aula (dataset de treino com 3 colunas: `x1, x2, label`):

```python
train = np.loadtxt("train_dataset1.csv", delimiter=",", skiprows=1)
X = train[:, :2]                 # as duas primeiras colunas são os atributos
y = train[:, 2]                  # a terceira é o rótulo
y = np.where(y == -1, -1, 1)     # garante rótulos em {-1, +1}

perceptron = Perceptron(learning_rate=0.1, n_epochs=100)
perceptron.fit(X, y)
perceptron.plot_error()          # curva: taxa de erro x época
plot_comparison(X, y, perceptron)  # fronteira de decisão + dados lado a lado
```

**O que olhar nos gráficos (e escrever na conclusão):**

- **Curva de erro chegou a 0?** → dados linearmente separáveis, Perceptron adequado.
- **Curva oscila e não chega a 0?** → dados **não** separáveis linearmente (ou ruído/overlap).
- **Fronteira de decisão** (`plot_decision_boundary`): é sempre uma **reta**. Se os pontos de
  uma classe ficam dos dois lados da reta, o modelo é inadequado.
- **Acurácia de treino × teste**: `np.mean(y_pred == y)`; a aula tem pares
  `train_datasetN.csv` / `test_datasetN.csv` justamente para você medir generalização.

> **Observações sobre o código da aula:** (1) `fit` imprime um resumo ao final; (2) `predict2`
> não está nos slides, é uma extensão do professor; (3) a linha `self.bias = 0` dentro de `fit`
> sobrescreve o bias inicializado — só afeta os modos `random` e `normal`.

### 1.6 Exercícios (com gabarito)

#### Exercício 1 — Uma época à mão

$\eta = 0{,}5$; $w=(0,0)$; $b=0$. Dados:

| Amostra | $x_1$ | $x_2$ | $y$ |
|---|---|---|---|
| P1 | 2 | 1 | +1 |
| P2 | 1 | −1 | −1 |
| P3 | −1 | −2 | −1 |

a) Faça a época 1 completa, preenchendo $z$, $\hat y$, erro e novos $w$, $b$.
b) Qual a taxa de erro da época 1?
c) Faça a época 2 e diga se o treino terminou.

<details>
<summary><b>Gabarito</b></summary>

a)

| Passo | $z$ | $\hat y$ | Erro? | $w$ depois | $b$ depois |
|---|---|---|---|---|---|
| P1 | $0$ | 0 | Sim | $(0+0{,}5\cdot2;\ 0+0{,}5\cdot1) = (1;\ 0{,}5)$ | $0{,}5$ |
| P2 | $1\cdot1 + 0{,}5\cdot(-1) + 0{,}5 = 1$ | +1 | Sim (era −1) | $(1-0{,}5;\ 0{,}5+0{,}5) = (0{,}5;\ 1)$ | $0$ |
| P3 | $0{,}5\cdot(-1) + 1\cdot(-2) + 0 = -2{,}5$ | −1 | Não | $(0{,}5;\ 1)$ | $0$ |

b) 2 erros em 3 amostras → **66,7 %**.

c) Época 2 com $w=(0{,}5;1)$, $b=0$: P1 → $z=2$ (+1 ✔), P2 → $z=-0{,}5$ (−1 ✔),
P3 → $z=-2{,}5$ (−1 ✔). **Zero erros → o treino termina** (2 épocas).
Fronteira final: $0{,}5x_1 + x_2 = 0$.
</details>

#### Exercício 2 — Por que o Perceptron falha no XOR?

Considere $(0,0)\to-1$, $(0,1)\to+1$, $(1,0)\to+1$, $(1,1)\to-1$.
a) Desenhe os 4 pontos. É possível separá-los com **uma** reta?
b) O que acontece com a curva de erro do Perceptron?
c) Que modelo da disciplina resolve e por quê?

<details>
<summary><b>Gabarito</b></summary>

a) Não. Os dois pontos de classe +1 ficam em cantos opostos do quadrado, assim como os dois de
classe −1; qualquer reta que separe um par deixa o outro misturado.

b) Nunca chega a zero — a cada época os pesos "pulam" e a taxa de erro oscila (num teste, 4 de 4
amostras erradas em cada época com $\eta=1$). O Perceptron só tem garantia de convergência para
dados linearmente separáveis.

c) A **MLP**: com ao menos uma camada oculta com ativação não linear, ela combina várias
retas/curvas e cria fronteiras não lineares — XOR é o exemplo clássico que motivou as redes
com camadas ocultas.
</details>

#### Exercício 3 — Interpretando resultados

Você treinou um Perceptron por 100 épocas. A taxa de erro da última época é 0,12 e a acurácia de
teste é 87 %. A curva de erro oscilava entre 0,10 e 0,15 desde a época 20. Quais conclusões são
válidas?

<details>
<summary><b>Gabarito</b></summary>

- O erro **não convergiu a 0** e estabilizou oscilando: os dados **não são perfeitamente
  separáveis por uma reta** (há sobreposição, ruído ou fronteira curva).
- Treinar mais épocas **não ajuda** (a curva já está "chata").
- A acurácia de ~87 % mostra que a reta captura boa parte do padrão, mas se a aplicação exige
  mais, é motivo para testar um modelo não linear (MLP) e **comparar**.
- Não é correto concluir "o código está errado": oscilação com dados não separáveis é o
  comportamento esperado do algoritmo.
</details>

---

## 2. Aula 11 — MLP (Multilayer Perceptron)

### 2.1 Quando usar

A MLP é uma rede neural com **uma ou mais camadas ocultas**. Use quando:

- O problema é de **classificação supervisionada** e a fronteira **não é uma reta** (círculos,
  espirais, regiões irregulares) — o Perceptron falharia.
- A saída tem **mais de 2 classes** (multiclasse): basta a camada de saída ter K neurônios.
- O enunciado fala em "fronteiras complexas/curvas", "padrões não lineares", "mais de duas
  categorias".

Custo: mais hiperparâmetros (arquitetura, ativação, otimizador, regularização), mais tempo de
treino e risco de **overfitting**. Se uma reta resolve, o Perceptron é preferível (mais simples).

### 2.2 Anatomia de uma MLP

```
Entrada        Camada oculta 1     Camada oculta 2        Saída
(n features)   (ex.: 50 neurônios) (ex.: 64 neurônios)    (K classes)

  x1 ───┐        ┌─ h1 ─┐            ┌─ g1 ─┐           ┌─ z1 ─► softmax ─► P(classe 1)
  x2 ───┼──────► │ ...  │ ─────────► │ ...  │ ────────► │ ...  │
  ...   ┘        └─ h50 ┘            └─ g64 ┘           └─ zK ─► softmax ─► P(classe K)
        pesos W1, b1       pesos W2, b2          pesos W3, b3
```

Cada camada faz: **combinação linear** + **função de ativação**:

$$h = f(W x + b)$$

- Sem a ativação não linear, empilhar camadas **não adiciona poder**: várias camadas lineares
  equivalem a uma só. A ativação é o que permite fronteiras curvas.
- Na lista `hidden_layers=[50, 64, 4]` do código da aula, **todos** os números são camadas
  **ocultas** (50, 64 e 4 neurônios). A camada de saída é criada **automaticamente** com tantos
  neurônios quanto classes existirem em `y`.
- **Número de parâmetros** de uma camada: $\text{entradas} \times \text{saídas} + \text{saídas}$
  (pesos + biases).

### 2.3 Funções de ativação

| Ativação | Fórmula | Saída | Observações |
|---|---|---|---|
| **ReLU** | $f(z) = \max(0, z)$ | $[0, +\infty)$ | Barata, não satura para $z>0$; padrão moderno. Neurônios com $z<0$ ficam "mortos" (saída 0) |
| **Tanh** | $f(z) = \dfrac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, +1)$ | Centrada em zero; satura nas pontas (gradiente pequeno em redes profundas) |
| **Sigmoide** | $\dfrac{1}{1+e^{-z}}$ | $(0, 1)$ | Usada em saída binária (regressão logística) |
| **Softmax** | $\dfrac{e^{z_k}}{\sum_j e^{z_j}}$ | probabilidades que somam 1 | Saída **multiclasse** |

Na aula só **ReLU** e **Tanh** são usadas nas camadas ocultas (`activation='relu'` ou `'tanh'`).

### 2.4 Saída multiclasse, softmax e perda (loss)

A camada de saída produz $K$ valores brutos (**logits**) $z_1, \dots, z_K$. O **softmax** os
transforma em probabilidades, e a classe prevista é o **índice do maior logit**
(`torch.max(outputs, 1)` no código — o softmax não muda qual é o maior).

A **perda** usada é a **entropia cruzada** (`nn.CrossEntropyLoss`), que já aplica o softmax
internamente — por isso a camada de saída do código **não tem ativação**:

$$\mathcal{L} = -\log p_{\text{classe correta}}$$

Quanto menor a probabilidade dada à classe certa, maior a perda ($p=1 \Rightarrow \mathcal{L}=0$;
$p\to0 \Rightarrow \mathcal{L}\to\infty$).

> **Rótulos para a `CrossEntropyLoss`:** precisam ser **inteiros $0, 1, \dots, K-1$**
> (`torch.LongTensor`). Datasets com rótulos $-1/+1$ precisam ser convertidos — é exatamente o
> que a aula faz com `y = (y + 1) // 2` ($-1\to0$, $+1\to1$). Datasets que já vêm 0/1/2 **não**
> precisam de conversão.

### 2.5 Como a rede aprende

1. **Forward**: calcula as saídas camada a camada.
2. **Loss**: mede o erro (entropia cruzada).
3. **Backpropagation**: calcula o gradiente $\partial\mathcal{L}/\partial w$ de cada peso pela
   regra da cadeia, da saída para a entrada (`loss.backward()`).
4. **Otimizador**: atualiza os pesos na direção que reduz a perda (`optimizer.step()`).

**Mini-batches:** em vez de usar todas as amostras de uma vez, o código fatia o treino em lotes
de `batch_size` (padrão 32). Cada lote gera uma atualização. Uma **época** = uma passada por todos
os lotes.

| Otimizador | Atualização | Características |
|---|---|---|
| **SGD** | $w \leftarrow w - \eta\,\nabla\mathcal{L}$ | Simples; sensível ao $\eta$; costuma precisar de mais épocas |
| **Adam** | Combina *momento* + taxa **adaptativa** por parâmetro | Converge mais rápido, menos sensível ao $\eta$ inicial; $\eta=0{,}001$ é o padrão |

### 2.6 Hiperparâmetros e regularização

| Hiperparâmetro | O que controla | Efeito |
|---|---|---|
| `hidden_layers` | Profundidade e largura | Mais neurônios/camadas → mais capacidade (e mais risco de overfitting) |
| `activation` | `'relu'` ou `'tanh'` | Forma da não linearidade |
| `optimizer` | `'sgd'` ou `'adam'` | Velocidade/estabilidade da convergência |
| `learning_rate` | Tamanho do passo | Alto demais: loss oscila/explode. Baixo demais: aprende muito devagar |
| `epochs` | Máx. de passadas | Pouco → *underfitting*; muito → *overfitting* (a menos que haja early stopping) |
| `batch_size` | Amostras por atualização | Menor → mais ruído e mais atualizações; maior → mais estável e rápido por época |
| `patience` | Early stopping | Quantas épocas esperar sem a loss de validação melhorar |
| `regularization` | `None`, `'dropout'` ou `'l2'` | Combate overfitting (abaixo) |

**Overfitting:** o modelo "decora" o treino. Sintoma típico nas curvas: a loss/acurácia de
**treino** melhora, mas a de **validação** estagna ou **piora**; a acurácia de treino fica muito
acima da de teste.

**Regularizações:**

- **Dropout** ($p$): a cada passo do treino, "desliga" aleatoriamente uma fração $p$ dos
  neurônios da camada. Obriga a rede a não depender de poucos neurônios. É desligado automaticamente
  em `model.eval()` (predição).
- **Penalização L2** ($\lambda$): adiciona $\lambda\sum w^2$ à perda → penaliza pesos grandes,
  produzindo uma fronteira mais suave. No PyTorch isso é o parâmetro `weight_decay`.

**Os três conjuntos de dados:**

| Conjunto | Para quê | Pode ajustar hiperparâmetros com ele? |
|---|---|---|
| **Treino** | Aprender os pesos | — |
| **Validação** | Escolher hiperparâmetros, early stopping | Sim |
| **Teste** | Avaliar o resultado **final**, uma única vez | **Não** (senão vira "treino disfarçado") |

### 2.7 O Protocolo Experimental (36 cenários)

O arquivo `aula 11/Protocolo Experimental.pdf` define um experimento sistemático. Fixe um dataset,
varie as 4 dimensões e compare a acurácia de teste:

| Dimensão | Opções | Nº |
|---|---|---|
| Otimizador | SGD (lr=0,01) · Adam (lr=0,001) | 2 |
| Arquitetura | Pequena `[50,64,4]` · Média `[50,128,64,4]` · Grande `[50,256,128,64,4]` | 3 |
| Regularização | Nenhuma · Dropout (p=0,1) · L2 (λ=0,001) | 3 |
| Ativação | ReLU · Tanh | 2 |

$2 \times 3 \times 3 \times 2 = 36$ cenários (os de 1 a 18 usam SGD; de 19 a 36, Adam).

Dica de organização para a prova: monte uma lista de dicionários com os cenários e rode em
`for`, guardando acurácia de treino/validação/teste em uma tabela (pode ser só uma lista de
tuplas impressa). Não precisa rodar os 36 se o tempo é curto — **um subconjunto bem escolhido e
bem justificado** (ex.: variando uma dimensão por vez) é defensável.

### 2.8 Exemplo resolvido — forward pass e loss à mão

Rede: 2 entradas → 2 neurônios ocultos (**ReLU**) → 2 saídas (softmax). Entrada $x=(1,\ 2)$.

$$W_1 = \begin{pmatrix} 0{,}5 & -1 \\ 1 & 0{,}5 \end{pmatrix},\ b_1 = \begin{pmatrix} 0 \\ -1 \end{pmatrix}
\qquad
W_2 = \begin{pmatrix} 1 & -1 \\ -0{,}5 & 2 \end{pmatrix},\ b_2 = \begin{pmatrix} 0 \\ 0{,}5 \end{pmatrix}$$

**Passo 1 — camada oculta (pré-ativação):** $z^{(1)} = W_1 x + b_1$

- $z^{(1)}_1 = 0{,}5\cdot1 + (-1)\cdot2 + 0 = -1{,}5$
- $z^{(1)}_2 = 1\cdot1 + 0{,}5\cdot2 - 1 = 1$

**Passo 2 — ReLU:** $h = (\max(0,-1{,}5),\ \max(0,1)) = (0,\ 1)$

**Passo 3 — saída (logits):** $z^{(2)} = W_2 h + b_2$

- $z^{(2)}_1 = 1\cdot0 + (-1)\cdot1 + 0 = -1$
- $z^{(2)}_2 = -0{,}5\cdot0 + 2\cdot1 + 0{,}5 = 2{,}5$

**Passo 4 — softmax:**

$$e^{-1} = 0{,}368,\quad e^{2{,}5} = 12{,}182,\quad \text{soma} = 12{,}550$$

$$p_1 = \frac{0{,}368}{12{,}550} = 0{,}0293,\qquad p_2 = \frac{12{,}182}{12{,}550} = 0{,}9707$$

**Passo 5 — predição:** classe de maior probabilidade → **classe 2** (índice 1 no código).

**Passo 6 — loss (entropia cruzada):**

- Se o rótulo verdadeiro é a **classe 2**: $\mathcal{L} = -\ln 0{,}9707 = \mathbf{0{,}030}$ (quase zero — acertou com confiança).
- Se o rótulo verdadeiro fosse a **classe 1**: $\mathcal{L} = -\ln 0{,}0293 = \mathbf{3{,}53}$ (erro "confiante" é punido duramente).

### 2.9 Código (estilo usado em aula — PyTorch)

Arquivo: `aula 11/MLP.ipynb`. Estrutura da classe `MLPClassifier`:

```python
import numpy as np, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt

class MLPClassifier:
    def __init__(self, hidden_layers=[64, 32], activation='relu', learning_rate=0.01,
                 epochs=100, patience=5, batch_size=32, optimizer='adam',
                 regularization=None, dropout_p=0.1, lambda_l2=0.001, random_state=None):
        ...

    def _build_model(self, input_dim, output_dim):
        layers, prev = [], input_dim
        for dim in self.hidden_layers:
            layers.append(nn.Linear(prev, dim))          # combinação linear
            layers.append(nn.ReLU() if self.activation == 'relu' else nn.Tanh())
            if self.regularization == 'dropout':
                layers.append(nn.Dropout(p=self.dropout_p))
            prev = dim
        layers.append(nn.Linear(prev, output_dim))       # saída SEM ativação (logits)
        return nn.Sequential(*layers)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # 1) converte para tensores (X -> FloatTensor, y -> LongTensor)
        # 2) output_dim = número de classes distintas em y
        # 3) criterion = CrossEntropyLoss ; optimizer = SGD ou Adam
        # 4) laço de épocas -> laço de mini-batches: zero_grad, forward, loss, backward, step
        # 5) com validação: guarda loss/acurácia de validação e aplica early stopping
        ...

    def predict(self, X):  ...   # model.eval(); argmax dos logits
    def evaluate(self, X, y):    # acurácia = np.mean(predict(X) == y)
    def plot_training_history(self): ...   # curvas de loss e acurácia (treino x validação)
    def plot_decision_boundary(self, X, y): ...  # só funciona para X com 2 colunas
```

Uso na aula (treino/teste, rótulos −1/+1 convertidos para 0/1):

```python
data = np.genfromtxt('train_dataset1.csv', delimiter=',', skip_header=1)
X_train, y_train = data[:, :-1], data[:, -1]
y_train = (y_train + 1) // 2                 # {-1,+1} -> {0,1}

data = np.genfromtxt('test_dataset1.csv', delimiter=',', skip_header=1)
X_test, y_test = data[:, :-1], (data[:, -1] + 1) // 2

mlp = MLPClassifier(optimizer='adam', learning_rate=0.001,
                    hidden_layers=[50, 128, 64, 4], activation='relu',
                    regularization='dropout', dropout_p=0.1)
mlp.fit(X_train, y_train)
print(f"Acurácia do Teste: {mlp.evaluate(X_test, y_test)*100:.2f}%")
mlp.plot_training_history()
mlp.plot_decision_boundary(X_train, y_train)
```

Com validação (datasets "4" da aula: `train_dataset.csv`, `validation_dataset.csv`,
`test_dataset.csv`):

```python
model = MLPClassifier(optimizer='SGD', learning_rate=0.01,
                      hidden_layers=[50, 64, 4], activation='relu', regularization=None)
model.fit(X_train, y_train, X_val, y_val)      # passa validação -> early stopping ativo
```

### 2.10 Armadilhas e bugs do código da aula (leia antes da prova!)

Ao reaproveitar a classe `MLPClassifier` na prova, estes pontos podem te custar tempo:

| # | Problema | Consequência | Como lidar |
|---|---|---|---|
| 1 | Rótulos precisam ser **inteiros 0..K−1** | `CrossEntropyLoss` dá erro (ou resultado absurdo) com −1/+1 ou floats não inteiros | Converta (`(y+1)//2` para ±1) e use `y.astype(int)` se vier como float |
| 2 | **Typo** no early stopping: `patiente_counter += 1` (com "e") | Se a loss de validação **não** melhorar, dá `NameError` ao passar `X_val, y_val` | Corrija para `patience_counter += 1` na sua cópia |
| 3 | `best_weights = self.model.state_dict()` guarda **referência**, não cópia | Ao restaurar no early stopping, você "restaura" os pesos atuais (não os melhores) | Use `copy.deepcopy(self.model.state_dict())` |
| 4 | A docstring cita `'12'` mas o código testa `'l2'` | Se escrever `'12'` a regularização **não é aplicada** (sem erro!) | Use exatamente `regularization='l2'` |
| 5 | L2 só é implementado quando `optimizer == 'adam'` | Com **SGD + L2** o L2 é ignorado silenciosamente (o cenário vira "sem regularização") | Para SGD, passe `weight_decay=self.lambda_l2` ao `optim.SGD` |
| 6 | `plot_decision_boundary` só desenha para **2 features** | Com 4 features imprime aviso e nada é plotado | Use curvas de treino + matriz de confusão como evidência |
| 7 | Uma mensagem de log diz "Validação acurácia" mas imprime a de **treino** | Pode confundir a leitura | A acurácia de validação aparece logo depois, na mesma linha |
| 8 | `__init__` imprime info de GPU e chama `torch.cuda.empty_cache()` | Só ruído em máquina sem GPU | Pode ignorar |
| 9 | `epochs=100` e `batch_size=32` padrão | Pode ser pouco/muito para o seu dataset | Ajuste e **observe as curvas** |
| 10 | Variar `random_state` muda o resultado | Dois treinos iguais dão acurácias diferentes | Fixe `random_state` para comparar cenários de forma justa |

### 2.11 Exercícios (com gabarito)

#### Exercício 1 — Forward pass com Tanh

Rede 2→2→1, ativação **tanh** na oculta e **sigmoide** na saída. $x=(1,-1)$,

$$W_1 = \begin{pmatrix} 1 & 1 \\ 0{,}5 & -0{,}5 \end{pmatrix},\ b_1=(0,0),\qquad w_2 = (2,\ 1),\ b_2 = -0{,}5.$$

Calcule a probabilidade de saída e a classe prevista (limiar 0,5).

<details>
<summary><b>Gabarito</b></summary>

- $z^{(1)}_1 = 1\cdot1 + 1\cdot(-1) = 0$ ; $z^{(1)}_2 = 0{,}5\cdot1 - 0{,}5\cdot(-1) = 1$
- $h = (\tanh 0,\ \tanh 1) = (0;\ 0{,}7616)$
- $z^{(2)} = 2\cdot0 + 1\cdot0{,}7616 - 0{,}5 = 0{,}2616$
- $p = \sigma(0{,}2616) = \dfrac{1}{1+e^{-0{,}2616}} = 0{,}565$ → como $p>0{,}5$, **classe 1**.
</details>

#### Exercício 2 — Contando parâmetros

Um `MLPClassifier(hidden_layers=[50, 64, 4])` é treinado em um dataset com **2 features** e
**2 classes**. Quantos parâmetros treináveis tem a rede? E se o dataset tivesse 4 features e
3 classes com `hidden_layers=[50, 128, 64, 4]`?

<details>
<summary><b>Gabarito</b></summary>

Fórmula por camada: entradas × saídas + saídas. A camada de saída é adicionada pelo código.

Caso 1: arquitetura $2 \to 50 \to 64 \to 4 \to 2$

- $2\cdot50+50 = 150$
- $50\cdot64+64 = 3264$
- $64\cdot4+4 = 260$
- $4\cdot2+2 = 10$
- **Total = 3684** parâmetros.

Caso 2: arquitetura $4 \to 50 \to 128 \to 64 \to 4 \to 3$

- $4\cdot50+50=250$ ; $50\cdot128+128=6528$ ; $128\cdot64+64=8256$ ; $64\cdot4+4=260$ ; $4\cdot3+3=15$
- **Total = 15 309** parâmetros.
</details>

#### Exercício 3 — Diagnóstico pelas curvas

Duas execuções do mesmo modelo:

- **Execução A:** acurácia de treino 99 %, validação 71 %, teste 70 %; a loss de validação
  começa a subir na época 15, enquanto a de treino continua caindo.
- **Execução B:** acurácia de treino 62 %, validação 61 %, teste 60 %; as duas losses caem
  devagar e ainda não estabilizaram na última época.

Diagnostique cada uma e sugira ações.

<details>
<summary><b>Gabarito</b></summary>

- **A = overfitting** (grande diferença treino × validação; loss de validação sobe). Ações:
  aumentar a regularização (dropout/L2), reduzir a arquitetura, usar early stopping com
  `patience`, ter mais dados.
- **B = underfitting / treino incompleto** (acurácias baixas e parecidas; loss ainda caindo).
  Ações: mais épocas, taxa de aprendizado maior (ou trocar SGD por Adam), arquitetura maior,
  conferir se as features estão em escala parecida.
</details>

#### Exercício 4 — Perguntas conceituais

a) Por que não basta empilhar camadas lineares?
b) Para 3 classes, quantos neurônios tem a camada de saída e qual loss é usada?
c) Por que o teste não pode ser usado para escolher hiperparâmetros?
d) Qual a diferença entre o que o Dropout faz no treino e na predição?

<details>
<summary><b>Gabarito</b></summary>

a) A composição de funções lineares é linear: sem ativação não linear, a rede inteira equivale
a uma única camada e só desenha retas.

b) 3 neurônios (um por classe) + `CrossEntropyLoss` (softmax embutido). A classe prevista é o
argmax.

c) Se você escolhe a configuração que melhor se saiu no teste, o teste passa a influenciar o
modelo e a acurácia final fica otimista (vazamento). Use **validação** para decidir e o teste
só no final.

d) No **treino** desliga neurônios aleatoriamente (fração $p$); na **predição**
(`model.eval()`) todos ficam ligados — o comportamento é determinístico.
</details>

---

## 3. Aula 12 — K-means

### 3.1 Quando usar

K-means é um algoritmo de **agrupamento (clustering)**, **não supervisionado**: os dados **não
têm rótulo**. Ele descobre $k$ grupos a partir da similaridade (proximidade) entre pontos.

Use quando o enunciado diz:

- "não sabemos as categorias / não há rótulos";
- "identificar **grupos**/**perfis**/**segmentos** automaticamente";
- "segmentar clientes", "agrupar documentos", "separar cores de uma imagem".

Não use quando existe uma coluna-alvo (classe) que você quer **prever** — aí é classificação
(Perceptron, MLP, KNN...).

### 3.2 O algoritmo (4 passos)

1. **Escolha $k$** (número de clusters) — o algoritmo **não** descobre $k$ sozinho.
2. **Inicialize** os $k$ centróides, normalmente escolhendo $k$ pontos aleatórios do dataset.
3. **Associação:** cada ponto vai para o cluster do centróide **mais próximo** (distância
   euclidiana):

$$d(a,b) = \sqrt{\sum_i (a_i - b_i)^2}$$

4. **Atualização:** cada centróide vira a **média** dos pontos do seu cluster:

$$\mu_j = \frac{1}{|C_j|}\sum_{x \in C_j} x$$

Repita 3 e 4 até **convergir** (centróides pararam de mudar) ou atingir `max_iter`.

**Função objetivo (inércia / WCSS):** a soma dos quadrados das distâncias de cada ponto ao seu
centróide. O K-means diminui (ou mantém) esse valor a cada iteração:

$$J = \sum_{j=1}^{k}\sum_{x\in C_j}\lVert x-\mu_j\rVert^2$$

### 3.3 Pontos importantes

| Tema | Detalhe |
|---|---|
| **Inicialização aleatória** | Rodadas diferentes podem dar resultados diferentes (mínimos locais). Fixe `np.random.seed(...)` para reproduzir e/ou rode várias vezes e compare a inércia |
| **Escala das variáveis** | Distância euclidiana é dominada pela variável de maior escala (ex.: R\$ vs. nº de compras). **Normalize/padronize** quando as escalas diferem muito |
| **Escolha de $k$** | **Método do cotovelo:** rode $k=1,2,3,\dots$, plote a inércia × $k$ e escolha o ponto em que a queda "dobra" (retornos decrescentes). Outro critério: **silhueta** (quanto maior, melhor separados e coesos os clusters) |
| **Formato dos clusters** | Funciona bem com grupos compactos, mais ou menos **esféricos** e de tamanho parecido; vai mal com formatos alongados/aninhados (aí a ideia é usar outro algoritmo) |
| **Outliers** | Distorcem a média (centróide) |
| **Cluster vazio** | Se nenhum ponto cair em um centróide, o código da aula **mantém o centróide antigo** (`if clusters[i]:`) |
| **Rótulos do cluster** | Os números 0, 1, 2 são **arbitrários** — "cluster 0" numa rodada pode ser "cluster 2" em outra. O significado vem da **interpretação** dos centróides |
| **Avaliação** | Sem rótulo, não existe "acurácia". Avalia-se por inércia, silhueta e **interpretação de negócio** |

### 3.4 Exemplo resolvido passo a passo

Pontos: $P_1(1,1),\ P_2(1{,}5;2),\ P_3(3,4),\ P_4(5,7),\ P_5(3{,}5;5),\ P_6(4{,}5;5)$. $k=2$,
centróides iniciais $c_1=P_1=(1,1)$ e $c_2=P_4=(5,7)$.

**Iteração 1 — associação (distâncias euclidianas):**

| Ponto | $d(\cdot, c_1)$ | $d(\cdot, c_2)$ | Cluster |
|---|---|---|---|
| $P_1(1,1)$ | 0 | 7,21 | 1 |
| $P_2(1{,}5;2)$ | 1,12 | 6,10 | 1 |
| $P_3(3,4)$ | 3,61 | 3,61 | 1 (empate → o `argmin` fica com o **primeiro**) |
| $P_4(5,7)$ | 7,21 | 0 | 2 |
| $P_5(3{,}5;5)$ | 4,72 | 2,50 | 2 |
| $P_6(4{,}5;5)$ | 5,32 | 2,06 | 2 |

*(Exemplo de conta: $d(P_5,c_1)=\sqrt{(3{,}5-1)^2+(5-1)^2}=\sqrt{6{,}25+16}=4{,}72$.)*

**Iteração 1 — atualização:**

- $c_1 = \text{média}(P_1,P_2,P_3) = \left(\dfrac{1+1{,}5+3}{3},\ \dfrac{1+2+4}{3}\right) = (1{,}83;\ 2{,}33)$
- $c_2 = \text{média}(P_4,P_5,P_6) = \left(\dfrac{5+3{,}5+4{,}5}{3},\ \dfrac{7+5+5}{3}\right) = (4{,}33;\ 5{,}67)$

**Iteração 2 — associação com os novos centróides:**

| Ponto | $d(\cdot, c_1)$ | $d(\cdot, c_2)$ | Cluster |
|---|---|---|---|
| $P_1$ | 1,57 | 5,73 | 1 |
| $P_2$ | 0,47 | 4,63 | 1 |
| $P_3$ | 2,03 | 2,13 | 1 |
| $P_4$ | 5,64 | 1,49 | 2 |
| $P_5$ | 3,14 | 1,07 | 2 |
| $P_6$ | 3,77 | 0,69 | 2 |

As associações **não mudaram** → os centróides continuam $(1{,}83;\ 2{,}33)$ e $(4{,}33;\ 5{,}67)$
→ **convergiu** (`np.allclose(centroids, old_centroids)` retorna `True` e o laço para).

**Inércia final:** $J = 10{,}67$ (cluster 1: $6{,}83$; cluster 2: $3{,}83$).

### 3.5 Código (estilo usado em aula — Numpy + Matplotlib)

Arquivo: `aula 12/kmean.ipynb` (dataset `dataset_clientes_kmeans.csv`: `frequencia_compras`,
`ticket_medio`).

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. ler dataset
X = np.genfromtxt("dataset_clientes_kmeans.csv", delimiter=",", skip_header=1)

# 2. distância euclidiana: de UM ponto "a" para TODOS os centróides "b"
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

# 3. K-means
def kmeans(X, k=3, max_iter=100):
    idx = np.random.choice(len(X), k, replace=False)   # k pontos aleatórios
    centroids = X[idx]                                  # centróides iniciais

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        labels = np.zeros(len(X))

        for i, point in enumerate(X):                   # associação
            distances = euclidean_distance(point, centroids)
            label = np.argmin(distances)                # centróide mais próximo
            clusters[label].append(point)
            labels[i] = label

        old_centroids = centroids.copy()                # guarda para comparar

        for i in range(k):                              # atualização = média
            if clusters[i]:                             # evita média de cluster vazio
                centroids[i] = np.mean(clusters[i], axis=0)

        if np.allclose(centroids, old_centroids):       # convergiu?
            break

    return centroids, labels

# 4. executar
centroids, labels = kmeans(X, k=3)

# 5. plotar
plt.figure(figsize=(10, 6))
for cluster_id in np.unique(labels):
    plt.scatter(X[labels == cluster_id, 0], X[labels == cluster_id, 1],
                alpha=0.7, label=f"Cluster {int(cluster_id)}")
plt.scatter(centroids[:, 0], centroids[:, 1], s=180, c="black", marker="X", label="Centróides")
plt.title("Segmentação de Clientes com K-Means")
plt.legend(); plt.grid(True); plt.show()
```

**Detalhes de código que costumam gerar dúvida:**

- `X[idx]` com `idx` array de índices cria uma **cópia** (indexação "fancy"), então alterar
  `centroids` não altera `X`. Mas ao guardar o estado anterior é necessário usar
  `centroids.copy()`, senão `old_centroids` e `centroids` seriam o mesmo objeto e o teste de
  parada sempre daria `True` na primeira iteração.
- `np.mean(clusters[i], axis=0)` calcula a média **por coluna** (cada feature), dando um ponto.
- `labels` é um vetor float (por causa de `np.zeros`); por isso o `int(cluster_id)` nos rótulos.
- `np.random.choice(..., replace=False)` garante centróides iniciais distintos. Sem
  `np.random.seed(...)`, o resultado muda a cada execução.
- Se o dataset tem **colunas de rótulo** (não é o caso do K-means, mas acontece em CSVs
  reaproveitados), exclua-as de `X` antes de agrupar.

**Extensão útil para escolher $k$ (método do cotovelo)** — não está nos slides, mas só usa o que
você já tem; adapte e teste:

```python
def inertia(X, centroids, labels):
    return sum(np.sum((X[labels == j] - centroids[j]) ** 2) for j in range(len(centroids)))

ks = range(1, 9)
inertias = []
for k in ks:
    c, l = kmeans(X, k=k)
    inertias.append(inertia(X, c, l))
plt.plot(list(ks), inertias, marker="o"); plt.xlabel("k"); plt.ylabel("Inércia"); plt.show()
```

### 3.6 Exercícios (com gabarito)

#### Exercício 1 — K-means à mão (3 iterações)

Pontos $A(1,1),\ B(2,1),\ C(4,3),\ D(5,4)$. $k=2$, centróides iniciais $c_1=A$ e $c_2=B$.
Execute o K-means até convergir, mostrando clusters e centróides a cada iteração.

<details>
<summary><b>Gabarito</b></summary>

**Iteração 1:** distâncias a $c_1=(1,1)$ e $c_2=(2,1)$:
$A$: 0 / 1 → 1 · $B$: 1 / 0 → 2 · $C$: 3,61 / 2,83 → 2 · $D$: 5 / 4,24 → 2.
Clusters: {A} e {B, C, D}. Novos centróides: $c_1=(1,1)$, $c_2=(3{,}67;\ 2{,}67)$.

**Iteração 2:** $A$: 0 / 3,15 → 1 · $B$: 1 / 2,36 → **1** · $C$: 3,61 / 0,47 → 2 · $D$: 5 / 1,89 → 2.
B **mudou** de cluster. Clusters: {A, B} e {C, D}. Novos: $c_1=(1{,}5;\ 1)$, $c_2=(4{,}5;\ 3{,}5)$.

**Iteração 3:** $A$: 0,5 / 4,30 → 1 · $B$: 0,5 / 3,54 → 1 · $C$: 3,20 / 0,71 → 2 · $D$: 4,61 / 0,71 → 2.
Nada muda → **convergiu** com $c_1=(1{,}5;\ 1)$ e $c_2=(4{,}5;\ 3{,}5)$.
Inércia final: $0{,}25+0{,}25+0{,}5+0{,}5 = 1{,}5$.

*Lição:* centróides iniciais ruins (A e B, muito próximos) foram "consertados" pelas iterações.
</details>

#### Exercício 2 — Escala das variáveis

Três clientes: $A=(2\ \text{compras},\ R\$100)$, $B=(3,\ R\$140)$, $C=(2,\ R\$60)$.
a) Calcule $d(A,B)$ e $d(A,C)$. Quem parece "mais parecido" com A?
b) O que as contas revelam sobre o problema e qual a correção?

<details>
<summary><b>Gabarito</b></summary>

a) $d(A,B)=\sqrt{1^2+40^2}=40{,}01$ ; $d(A,C)=\sqrt{0^2+40^2}=40{,}00$ → praticamente
**empatados**, ou seja, a diferença de 1 compra (B tem 50 % mais compras que A!) quase não pesa.

b) O ticket médio (ordem de grandeza maior) **domina** a distância e "abafa" a frequência de
compras. Solução: **padronizar** as colunas (`(X - X.mean(axis=0)) / X.std(axis=0)`) ou
normalizar para [0,1] antes de rodar o K-means.
</details>

#### Exercício 3 — Escolhendo k pelo cotovelo

Inércias obtidas: $k=1:\ 950$ · $k=2:\ 480$ · $k=3:\ 190$ · $k=4:\ 170$ · $k=5:\ 155$ · $k=6:\ 145$.
Qual $k$ você escolheria? Justifique.

<details>
<summary><b>Gabarito</b></summary>

**$k=3$.** As quedas até 3 são grandes (950→480→190); depois a curva "achata" (190→170→155→145:
ganhos pequenos). O "cotovelo" está em $k=3$. Usar $k$ maior só reduz a inércia por construção
(com $k=n$ ela seria 0), mas não traz grupos mais interpretáveis.
</details>

#### Exercício 4 — Verdadeiro ou falso

a) O K-means garante o mesmo resultado em qualquer execução.
b) O K-means precisa de rótulos para treinar.
c) Os centróides são sempre pontos do dataset.
d) O K-means pode terminar com um cluster vazio.

<details>
<summary><b>Gabarito</b></summary>

a) **Falso** — depende da inicialização aleatória (mínimos locais).
b) **Falso** — é não supervisionado.
c) **Falso** — só os **iniciais** são pontos do dataset; depois viram médias.
d) **Verdadeiro** — pode ocorrer (por isso o `if clusters[i]:` do código da aula).
</details>

---

## 4. Guia de Decisão — Perceptron × MLP × K-means

Na prova, o enunciado diz "escolha o algoritmo mais adequado". Siga as perguntas **em ordem**.

### Passo 1 — Existe uma coluna com a resposta certa (rótulo/classe)?

- **Não** (você quer *descobrir* grupos) → **K-means** (não supervisionado). Fim.
- **Sim** → é classificação supervisionada → Passo 2.

### Passo 2 — Quantas classes existem?

- **Mais de 2 (multiclasse)** → **MLP** (saída com K neurônios). O Perceptron da aula é binário.
- **2 classes** → Passo 3.

### Passo 3 — Uma reta (ou hiperplano) separa as classes?

- **Sim / "linha clara de separação" / "modelo mais simples e eficiente"** → **Perceptron**.
- **Não / "fronteira curva ou complexa" / "uma reta não separa" / "padrões não lineares"** → **MLP**.
- **Em dúvida?** Plote os dados. Treine o Perceptron: se o erro não chega a (quase) zero e a
  fronteira corta as nuvens, passe para a MLP e **compare** — essa comparação vira argumento na
  sua conclusão.

### Tabela de palavras-chave do enunciado

| O enunciado diz... | Pense em |
|---|---|
| "perfis", "grupos", "segmentos", "não conhecemos as categorias", "sem rótulo" | K-means |
| "descobrir padrões", "agrupar automaticamente", "direcionar campanhas por grupo" | K-means |
| "separa linearmente", "linha clara", "mais simples possível", "eficiência computacional" | Perceptron |
| "classificar em 2 categorias" + dados bem separados | Perceptron |
| "círculo ao redor", "uma reta não consegue", "fronteira curva/complexa", "não linear" | MLP |
| "3 (ou mais) classes", "multiclasse", "suportar múltiplas saídas" | MLP |
| "camadas ocultas", "dropout", "otimizador", "arquitetura" | MLP |

### Comparação resumida

| | Perceptron | MLP | K-means |
|---|---|---|---|
| Tipo | Supervisionado | Supervisionado | **Não** supervisionado |
| Tarefa | Classificação **binária** | Classificação binária/**multiclasse** (e não linear) | Agrupamento |
| Fronteira | Reta/hiperplano | Curva/complexa | — (regiões de Voronoi em torno dos centróides) |
| Hiperparâmetros | $\eta$, épocas, inicialização | Arquitetura, ativação, otimizador, lr, regularização, batch, épocas | $k$, `max_iter`, inicialização |
| Como avaliar | Acurácia, curva de erro, fronteira | Acurácia, curvas de loss/acurácia, fronteira, matriz de confusão | Inércia, cotovelo, silhueta, interpretação |
| Risco | Não converge se não separável | Overfitting, ajuste de hiperparâmetros | $k$ mal escolhido, escala, inicialização |
| Escala dos dados | Útil normalizar | **Importante** normalizar | **Importante** normalizar |

### Mini-exercícios — "identifique o algoritmo" (com gabarito)

1. *Uma loja quer separar clientes em perfis de compra, mas nunca rotulou ninguém.*
2. *Um hospital tem exames de dois tipos e o rótulo benigno/maligno; a análise gráfica mostra
   duas nuvens bem separadas por uma reta. Pede-se o modelo mais leve.*
3. *Uma rede de sensores classifica peças em 4 tipos de defeito usando 6 medições.*
4. *Dados de duas classes formam duas espirais entrelaçadas.*
5. *Um app de música quer agrupar faixas parecidas para sugerir playlists, sem gêneros prévios.*
6. *Prever se um e-mail é spam (sim/não) usando duas métricas simples quando as classes são
   claramente separáveis.*

<details>
<summary><b>Gabarito</b></summary>

1. **K-means** (sem rótulos, quer perfis).
2. **Perceptron** (binário, separável por reta, pede o mais simples).
3. **MLP** (multiclasse, 6 features).
4. **MLP** (fronteira não linear; Perceptron nunca converge).
5. **K-means** (agrupamento sem rótulos).
6. **Perceptron** (binário, linearmente separável).
</details>

---

## 5. Resolvendo a P2 2025 — orientação sem respostas

> **Como esta seção funciona.** Ela **não resolve** as questões. Para cada uma há:
> **(a)** como ler o enunciado, **(b)** perguntas para você responder, **(c)** um checklist de
> passos, **(d)** erros comuns e **(e)** *dicas progressivas* escondidas — abra só até o nível de
> que precisar. Tente sempre responder as perguntas antes de abrir uma dica.
>
> Arquivos: `p2-2025/Avaliação P2 - ... Prova A.pdf` e os 4 datasets. Os notebooks
> `questao1.ipynb` ... `questao4.ipynb` estão **vazios** — são seus para preencher.

### 5.0 O que a prova pede (comum às 4 questões)

- Para cada situação-problema, **escolher** o algoritmo (Perceptron, MLP ou K-means), **aplicar**
  e **explicar com base nos resultados e gráficos** (a justificativa vale tanto quanto o código).
- Entrega: **um Jupyter Notebook** com código e resultados + **os CSVs usados**, compactados em
  um `.zip` com o seu nome completo.
- Prova **offline**, 1h35 (07h45–09h20), consulta apenas ao **seu material** → deixe as classes
  `Perceptron`, `MLPClassifier` e a função `kmeans` prontas e testadas.

**Estratégia sugerida de tempo (aprox.):** ~10 min lendo/decidindo algoritmos, ~20 min por questão,
~10 min finais para conclusões e conferir o zip. Faça primeiro a que você tiver mais segurança.

**Checklist universal (aplicar em toda questão):**

1. [ ] Abri o CSV: quantas linhas/colunas? Tem coluna de rótulo? Que valores ela assume?
2. [ ] Plotei os dados (scatter) antes de escolher o modelo?
3. [ ] Decidi o algoritmo com um argumento baseado no enunciado **e** no gráfico?
4. [ ] Os rótulos estão no formato que o meu código exige?
5. [ ] Separei treino/teste (ou treino/validação/teste) — o CSV da prova vem **inteiro**?
6. [ ] Gerei os gráficos pedidos (fronteira, curva de erro/loss, clusters)?
7. [ ] Escrevi a conclusão **com números e figuras** que apareceram no notebook?
8. [ ] O notebook roda do início ao fim (`Restart & Run All`) e os CSVs estão no zip?

**Como separar treino/teste só com Numpy** (os datasets da prova vêm sem divisão):

```python
np.random.seed(42)                      # reprodutibilidade
idx = np.random.permutation(len(X))     # embaralha os índices
corte = int(0.8 * len(X))               # 80 % treino
treino, teste = idx[:corte], idx[corte:]
X_train, X_test = X[treino], X[teste]
y_train, y_test = y[treino], y[teste]
```
(Pergunta para pensar: por que **embaralhar** antes de cortar? Os CSVs podem estar ordenados por
classe? Confira com `y[:20]`.)

---

### Questão 1 — Segmentação de clientes (`dataset_marketing.csv`)

**Resumo do enunciado:** 300 clientes, 2 variáveis (Renda Anual em milhares e Score de Gastos).
Ninguém sabe as categorias; acredita-se haver perfis distintos. Identificar os grupos
**automaticamente** para campanhas.

**Perguntas para você responder:**

1. O arquivo tem coluna de rótulo? O que isso diz sobre a **família** do algoritmo (supervisionado
   ou não)?
2. O enunciado diz quantos grupos existem? Se não, como você pode **escolher** esse número a
   partir dos dados? (Releia a Seção 3.3.)
3. As duas variáveis estão em escalas parecidas? Isso importa para a distância euclidiana?
   Confira com `X.min(axis=0)`, `X.max(axis=0)`.
4. O resultado muda se você rodar duas vezes? Como estabilizar/comparar? (Seção 3.3: inicialização.)
5. O que cada centróide **significa** em linguagem de marketing (renda alta/gasto alto etc.)?

**Checklist de passos:**

1. [ ] Carregar com `np.genfromtxt(..., skip_header=1)` e conferir o `shape`.
2. [ ] Scatter simples dos dados **antes** de agrupar — quantas "nuvens" você enxerga?
3. [ ] Testar vários valores de $k$ e comparar a **inércia** (cotovelo).
4. [ ] Rodar o K-means com o $k$ escolhido; plotar clusters + centróides (como na aula).
5. [ ] Interpretar cada centróide (renda e gasto médios) e dar um nome/perfil a cada grupo.
6. [ ] Sugerir uma ação de campanha para cada perfil.

**Erros comuns:**

- Usar um algoritmo supervisionado (não há alvo).
- Escolher $k$ "no chute" sem justificar.
- Concluir só com "formou 3 grupos" sem descrever **quem são** os grupos.
- Esquecer `skip_header=1` e quebrar o código com a linha de cabeçalho em texto.

**O que sua conclusão deve conter (critérios, não texto pronto):** o algoritmo escolhido e por
quê; como o $k$ foi definido (com evidência); a descrição dos perfis a partir dos centróides; uma
ressalva (sensibilidade à inicialização, escala etc.).

<details>
<summary><b>Dica nível 1 — só uma pista</b></summary>

Dados **sem** coluna-alvo e o objetivo é "descobrir grupos": releie o Passo 1 do Guia de Decisão
(Seção 4). Para escolher o número de grupos, use o gráfico da inércia em função de $k$.
</details>

<details>
<summary><b>Dica nível 2 — como proceder</b></summary>

Use a função `kmeans` da aula 12 dentro de um `for k in range(...)`, calcule a inércia de cada
rodada (extensão da Seção 3.5) e plote. Depois rode o `kmeans` final com o $k$ do cotovelo e
reutilize o código de plot da aula. Para ler os centróides, imprima `centroids` e relacione a
coluna 0 (renda) e a coluna 1 (score).
</details>

<details>
<summary><b>Dica nível 3 — como validar o seu resultado (sem dar o resultado)</b></summary>

Rode o seu `kmeans` 5 vezes com sementes diferentes para o mesmo $k$: os centróides devem ser
praticamente os mesmos (só a ordem dos números dos clusters muda). Se mudarem muito, aumente
`max_iter`, revise a escolha de $k$ ou rode várias vezes e fique com a menor inércia. Confira
também se a soma dos tamanhos dos clusters é 300.
</details>

---

### Questão 2 — Aprovação de cartão de crédito (`dataset_credito.csv`)

**Resumo do enunciado:** variáveis Histórico de Crédito e Dívida Atual; coluna `Aprovado` (0/1)
= bom pagador. Há uma análise prévia dizendo que existe uma **linha clara** que separa os grupos.
Pede o **modelo mais simples possível** com **eficiência computacional**.

**Perguntas para você responder:**

1. Existe rótulo? Quantas classes? (Passos 1 e 2 do Guia de Decisão.)
2. O que "existe uma linha clara" e "modelo mais simples" sugerem sobre a **forma da fronteira**?
3. Quais valores a coluna `Aprovado` tem? O seu código de treino espera quais valores? (Releia a
   Seção 1.3 — "Rótulos".)
4. Como você **prova** (com um gráfico) que a escolha foi boa? O que a curva de erro por época
   deve mostrar? E a fronteira de decisão?
5. Por que ter um conjunto de **teste** separado importa mesmo que o treino dê 100 %?

**Checklist de passos:**

1. [ ] Carregar, conferir `shape`, valores únicos da última coluna (`np.unique(y)`).
2. [ ] Scatter colorido pelas classes — a "linha clara" aparece?
3. [ ] Converter os rótulos para o formato exigido pelo modelo.
4. [ ] Separar treino/teste (função da Seção 5.0).
5. [ ] Treinar; plotar **taxa de erro × época** e **fronteira de decisão**; medir acurácia em
   treino **e** teste.
6. [ ] Escrever a conclusão ligando o resultado ao enunciado (simplicidade, custo computacional).

**Erros comuns:**

- Usar os rótulos 0/1 diretamente numa regra pensada para $-1/+1$. Pense no que acontece com
  `self.weights += lr * y[i] * X[i]` quando `y[i] == 0`, e se `np.sign(...)` pode algum dia ser
  igual a 0 de forma "correta" para essas amostras. Como fica a curva de erro?
- Avaliar só no treino.
- Escolher MLP "porque é mais poderosa" e **não justificar** por que o enunciado pede o contrário.
- Pesos e dados em escalas muito diferentes e `learning_rate` mal calibrado — observe a curva.

**O que sua conclusão deve conter:** a escolha do algoritmo com a justificativa ligada ao
enunciado ("linha clara" + "simples" + "eficiência"); evidência (curva de erro, fronteira,
acurácia de teste); o que você concluiria se o erro **não** chegasse a zero; limitações.

<details>
<summary><b>Dica nível 1 — só uma pista</b></summary>

Duas classes + "uma linha clara separa" + "o mais simples possível" apontam para o modelo da
**Aula 10**. O cuidado principal é com o **formato dos rótulos**.
</details>

<details>
<summary><b>Dica nível 2 — como proceder</b></summary>

Reaproveite a classe `Perceptron` e as funções `plot_error` / `plot_comparison` /
`plot_decision_boundary` do `aula10/perceptron.ipynb`. Converta `Aprovado` de {0,1} para
{−1,+1} (uma linha com `np.where`). Teste alguns valores de `learning_rate` e de `n_epochs`
e compare as curvas de erro, em vez de fixar um valor sem olhar.
</details>

<details>
<summary><b>Dica nível 3 — como validar o seu resultado (sem dar o resultado)</b></summary>

Se o modelo for adequado, a taxa de erro deve cair rapidamente e a fronteira deve passar **entre**
as duas nuvens. Compare acurácia de treino e de teste: valores próximos indicam boa
generalização. Se a curva de erro ficar oscilando, pergunte-se: os dados são mesmo separáveis,
ou há alguns pontos sobrepostos (ruído)? Um pequeno resíduo de erro **não** invalida a escolha —
mas deve ser comentado.
</details>

---

### Questão 3 — Detecção de fraude (`dataset_fraude.csv`)

**Resumo do enunciado:** variáveis Valor e Distância (normalizadas); `Fraude` = rótulo. As
fraudes formam um **círculo ao redor** das transações legítimas (ou vice-versa). Testes anteriores
mostraram que **uma única reta não separa**. Escolher um algoritmo capaz de aprender fronteiras
**curvas ou complexas** e treinar um classificador.

**Perguntas para você responder:**

1. Há rótulo? Quantas classes? Existe separação linear (o que o próprio enunciado diz)?
2. Que característica do modelo da Aula 11 permite fronteiras curvas? (Releia 2.2 — o papel da
   ativação e das camadas ocultas.)
3. Como você **demonstra** (e não apenas afirma) que o modelo mais simples falha? Dica de
   método: *treine-o e olhe o resultado*.
4. Seus rótulos já estão no formato certo para a `CrossEntropyLoss`? (Seção 2.4 e 2.10 #1.)
5. Como escolher arquitetura, ativação, otimizador e regularização sem "chutar"? (Seção 2.7.)
6. Como você sabe que não houve overfitting? Quais curvas olhar?

**Checklist de passos:**

1. [ ] Scatter dos dados — confira visualmente o "círculo".
2. [ ] Verificar valores de `Fraude`; converter para inteiros se vierem como `float`.
3. [ ] Separar treino/validação/teste (ou, no mínimo, treino/teste + `validation` no `fit`).
4. [ ] (Opcional, mas forte na argumentação) Treinar o modelo linear como **baseline** e registrar
   a acurácia.
5. [ ] Treinar a MLP; escolher a configuração com base em **validação**, não no teste.
6. [ ] Plotar `plot_training_history()` e `plot_decision_boundary()` (aqui são 2 features, então a
   fronteira aparece!).
7. [ ] Avaliar **no teste uma vez**; comparar com o baseline.
8. [ ] Conclusão: a fronteira realmente ficou curva? Acurácia? Houve sinais de overfitting?

**Erros comuns:**

- Tentar o Perceptron e "forçar" a conclusão de que serve só porque a acurácia deu "razoável"
  (compare a fronteira no gráfico!).
- Usar o teste para escolher hiperparâmetros.
- Rodar `fit` passando validação e cair no bug do `patiente_counter` (Seção 2.10 #2).
- Arquitetura muito pequena/poucas épocas e concluir que "não funciona" — olhe a curva de loss:
  ainda estava caindo?
- Ativação/otimizador escolhidos sem comparar com outra opção.

**O que sua conclusão deve conter:** por que o modelo linear não serve (com evidência); por que a
MLP serve; configuração final e **por que** (cenários comparados); métricas de treino/validação/
teste; análise das curvas (overfitting?) e do gráfico da fronteira.

<details>
<summary><b>Dica nível 1 — só uma pista</b></summary>

"Fronteira curva/complexa" + "uma reta não separa" + classificação supervisionada: pense no
modelo da **Aula 11**. Para este dataset o rótulo já é 0/1.
</details>

<details>
<summary><b>Dica nível 2 — como proceder</b></summary>

Use `MLPClassifier` do `aula 11/MLP.ipynb` com `hidden_layers` com mais de uma camada e
`activation='relu'` ou `'tanh'` (teste as duas). Use a tabela do Protocolo Experimental como
inspiração: varie **uma** dimensão por vez (ex.: Adam vs. SGD; com e sem dropout) e anote as
acurácias de validação numa tabela. Corrija o typo `patiente_counter` antes de usar validação.
</details>

<details>
<summary><b>Dica nível 3 — como validar o seu resultado (sem dar o resultado)</b></summary>

Se o modelo aprendeu o padrão circular, a fronteira plotada **envolve** uma classe em vez de ser
uma linha reta, e a acurácia de teste fica **claramente acima** do baseline linear. Se a
acurácia de treino for muito maior que a de validação/teste, desconfie de overfitting e teste
dropout/L2. Se as duas forem baixas e a loss ainda estiver caindo, treine por mais épocas ou
aumente a capacidade da rede.
</details>

---

### Questão 4 — Robô advisor (`dataset_investimentos.csv`)

**Resumo do enunciado:** 4 características (Volatilidade, Retorno Esperado, Liquidez, Tempo de
Maturação) e 3 classes de risco: 0 (Baixo), 1 (Médio) e 2 (Alto). O modelo deve receber as 4
informações e prever a categoria; "como a saída não é binária... seu modelo precisa suportar
**múltiplas saídas**".

**Perguntas para você responder:**

1. Quantas classes? Qual dos três algoritmos aceita mais de 2 classes **do jeito que foi
   implementado na aula**? (Releia 1.1 e 2.1.)
2. Quantos neurônios terá a camada de saída? Quem a cria — você ou o código da aula?
3. As 4 features estão em escalas parecidas? O que isso pode causar no treino? (Confira
   `X.min(axis=0)` / `X.max(axis=0)` / `X.std(axis=0)`.)
4. Dá para plotar a fronteira de decisão como na Q3? Por quê? O que usar no lugar para provar
   que o modelo funciona?
5. Acurácia geral esconde algo? Que ferramenta da **Aula 6 do guia anterior** mostra *em que
   classes* o modelo erra?
6. As classes são balanceadas? (Cheque `np.unique(y, return_counts=True)`.)

**Checklist de passos:**

1. [ ] Conferir `shape` (400 × 5), classes e contagem por classe.
2. [ ] Decidir se normaliza as features — e **ajustar a normalização só no treino**, aplicando os
   mesmos parâmetros em validação/teste (por quê? vazamento de informação).
3. [ ] Separar treino/validação/teste (embaralhado).
4. [ ] Treinar a MLP; comparar pelo menos algumas configurações com a validação.
5. [ ] Plotar curvas de loss/acurácia (treino × validação).
6. [ ] No teste: acurácia **e** matriz de confusão 3×3 (monte com Numpy: contagem
   `real × previsto`).
7. [ ] Interpretar: que classe é mais difícil? Que tipo de erro seria mais grave para um robô
   advisor (prever "baixo" para um ativo de risco "alto")?

**Erros comuns:**

- Tentar usar o Perceptron "porque é binário, vou treinar 3 deles" sem justificar (é possível em
  teoria — *um contra todos* — mas fora do que a aula implementou e do que o enunciado sugere).
- Chamar `plot_decision_boundary` com 4 features (só imprime um aviso — e então ficar sem
  figura de apoio).
- Passar `y` como `float`: `torch.LongTensor(y)` precisa de inteiros → use `y.astype(int)`.
- Normalizar usando média/desvio do **dataset todo** (vaza informação do teste).
- Reportar só a acurácia e não falar das classes individualmente.

**O que sua conclusão deve conter:** por que o modelo suporta 3 classes; arquitetura/ativação/
otimizador finais e justificativa; acurácia de teste; matriz de confusão e análise por classe; uma
reflexão sobre o custo de errar em cada direção (risco).

<details>
<summary><b>Dica nível 1 — só uma pista</b></summary>

3 classes → o modelo que a Aula 11 implementa com **K neurônios de saída** e `CrossEntropyLoss`.
As classes já estão como 0, 1 e 2 — mas veja o tipo do array.
</details>

<details>
<summary><b>Dica nível 2 — como proceder</b></summary>

Reaproveite o `MLPClassifier` (ele define sozinho `output_dim = nº de classes distintas`).
Padronize as features usando média/desvio **do treino**. Treine algumas configurações, use a
validação para escolher e só então avalie no teste. Para a matriz de confusão, crie uma matriz
`np.zeros((3,3), int)` e incremente `cm[real, previsto]` em um laço.
</details>

<details>
<summary><b>Dica nível 3 — como validar o seu resultado (sem dar o resultado)</b></summary>

A soma da matriz de confusão deve ser igual ao tamanho do conjunto de teste, e a **diagonal
dividida pela soma** deve coincidir com a acurácia que o `evaluate` imprimiu — se não bater, há
bug (rótulos desalinhados, por exemplo). Compare a acurácia com a de um "chute na classe mais
frequente" para saber se o modelo realmente aprendeu algo.
</details>

---

### 5.5 Autoavaliação final da P2

Antes de entregar, responda sem consultar:

1. Por que a Q1 **não** usa acurácia?
2. Que evidência (gráfico) prova que o Perceptron foi uma escolha adequada na Q2?
3. O que você mostraria para provar que a Q3 exige um modelo não linear?
4. Por que a fronteira de decisão não aparece na Q4?
5. Em qual questão o `random_state`/semente importa mais para a reprodutibilidade? Em quais
   questões você precisa de **validação**? Em quais precisa de **teste**?

<details>
<summary><b>Respostas de conferência (conceituais, sobre o conteúdo — não sobre os resultados dos datasets)</b></summary>

1. Porque não há rótulo para comparar; usa-se inércia, cotovelo e interpretação.
2. A curva de erro convergindo e a reta de fronteira passando entre as classes, com boa acurácia
   de teste.
3. O baseline linear com fronteira reta cortando as classes + acurácia insuficiente, contraposto à
   fronteira curva da MLP.
4. Porque há 4 features (o gráfico 2D da aula só funciona com 2).
5. Semente: em todas que envolvem inicialização/embaralhamento aleatório (K-means, Perceptron com
   `random`, MLP, divisão treino/teste). Validação: nas que comparam configurações da MLP (Q3 e
   Q4). Teste: Q2, Q3, Q4 (supervisionadas).
</details>

---

## 6. Cola Final — fórmulas e hiperparâmetros

| Tópico | Fórmula / regra |
|---|---|
| Perceptron — saída linear | $z = w\cdot x + b$ |
| Perceptron — predição | $\hat y = \text{sign}(z) \in \{-1, 0, +1\}$ |
| Perceptron — atualização (só se $\hat y \neq y$) | $w \leftarrow w + \eta\,y\,x$ ; $b \leftarrow b + \eta\,y$ |
| Perceptron — taxa de erro da época | $\text{erros}/n$ |
| Perceptron — fronteira | $w\cdot x + b = 0$ (reta em 2D) |
| Perceptron — rótulos | $\{-1,+1\}$ (converter 0/1) |
| MLP — camada | $h = f(Wx + b)$ |
| MLP — ReLU / Tanh | $\max(0,z)$ / $\dfrac{e^z-e^{-z}}{e^z+e^{-z}}$ |
| MLP — softmax | $p_k = \dfrac{e^{z_k}}{\sum_j e^{z_j}}$ |
| MLP — entropia cruzada | $\mathcal{L} = -\log p_{\text{classe correta}}$ |
| MLP — parâmetros por camada | $\text{entradas}\times\text{saídas}+\text{saídas}$ |
| MLP — SGD | $w \leftarrow w - \eta\nabla\mathcal{L}$ |
| MLP — L2 | $\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda\sum w^2$ |
| MLP — rótulos | inteiros $0,\dots,K-1$ |
| MLP — Protocolo Experimental | 2 otimizadores × 3 arquiteturas × 3 regularizações × 2 ativações = 36 |
| K-means — distância | $\sqrt{\sum_i(a_i-b_i)^2}$ |
| K-means — associação | cluster = $\arg\min_j\ d(x,\mu_j)$ |
| K-means — atualização | $\mu_j = \text{média dos pontos do cluster } j$ |
| K-means — inércia | $J = \sum_j\sum_{x\in C_j}\lVert x-\mu_j\rVert^2$ |
| K-means — escolha de $k$ | cotovelo (inércia × $k$) / silhueta |
| Treino / validação / teste | treino aprende; validação escolhe hiperparâmetros; teste avalia **uma vez** |
| Overfitting | treino ↑ e validação ↓ (ou estagnada); gap grande treino × teste |

### Fluxo de decisão resumido

```
Há uma coluna de rótulo (classe) para PREVER?
│
├── NÃO ──────────────────────────► K-means  (escolha k pelo cotovelo; normalize; interprete centróides)
│
└── SIM ──► Classificação supervisionada
        │
        ├── Mais de 2 classes (multiclasse) ───────────────► MLP  (K neurônios de saída + CrossEntropy)
        │
        └── 2 classes
              ├── Reta separa / "o mais simples" ──────────► Perceptron (rótulos −1/+1)
              └── Fronteira curva / não linear ────────────► MLP

Depois de treinar:
  Perceptron → curva de erro × época + fronteira (reta) + acurácia treino/teste
  MLP        → curvas loss/acurácia (treino × validação) + fronteira (se 2D) + matriz de confusão
  K-means    → inércia/cotovelo + gráfico dos clusters + interpretação dos centróides
```
