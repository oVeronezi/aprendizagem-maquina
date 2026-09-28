# Guia de Estudos para a Prova — Aprendizagem de Máquina

> Material de consulta consolidando o conteúdo das Aulas 2 a 7 (fundamentos de Python,
> Regressão Linear Simples, Regressão Linear Múltipla, Regressão Logística, Matriz de Confusão
> e KNN), com explicações detalhadas, exemplos numéricos resolvidos passo a passo, exercícios
> com gabarito, e um **guia de decisão** para identificar qual algoritmo usar em cada tipo de
> exercício de prova.

## Como usar este guia

Cada seção de algoritmo segue a mesma estrutura: **quando usar → fórmulas → exemplo resolvido
passo a passo → código no estilo usado em aula (Numpy puro) → exercícios com gabarito**. Se você
tem pouco tempo antes da prova, vá direto para a **Seção 7 (Guia de Decisão)** e para a
**Seção 8 (Cola Final)** — elas resumem tudo. Depois volte às seções de cada algoritmo para
revisar os pontos que ficaram menos claros.

---

## Sumário

1. [Aula 2 — Fundamentos de Python, POO, Numpy e Matplotlib](#1-aula-2--fundamentos-de-python-poo-numpy-e-matplotlib)
2. [Aula 3 — Regressão Linear Simples](#2-aula-3--regressão-linear-simples)
3. [Aula 4 — Regressão Linear Múltipla](#3-aula-4--regressão-linear-múltipla)
4. [Aula 5 — Regressão Logística](#4-aula-5--regressão-logística)
5. [Aula 6 — Matriz de Confusão e Métricas de Classificação](#5-aula-6--matriz-de-confusão-e-métricas-de-classificação)
6. [Aula 7 — KNN (K-Nearest Neighbors)](#6-aula-7--knn-k-nearest-neighbors)
7. [Guia de Decisão — Qual algoritmo usar em cada exercício](#7-guia-de-decisão--qual-algoritmo-usar-em-cada-exercício)
8. [Cola Final — Todas as fórmulas em uma tabela](#8-cola-final--todas-as-fórmulas-em-uma-tabela)

---

## 1. Aula 2 — Fundamentos de Python, POO, Numpy e Matplotlib

Esta aula não é um "algoritmo de ML" em si, mas é a base de código usada em **todas** as outras
aulas. Vale revisar rapidamente porque a prova pode pedir para ler/completar um trecho de código.

### 1.1 Estruturas de dados

| Estrutura | Mutável? | Uso típico |
|---|---|---|
| **Lista** `[1, 2, 3]` | Sim | Coleção ordenada que muda de tamanho (ex.: `frutas.append("uva")`) |
| **Tupla** `(10, "Vini")` | Não | Agrupar valores fixos (ex.: coordenadas) |
| **Dicionário** `{"nome": "Tiago", "nota": 9.5}` | Sim | Pares chave-valor, acesso por chave (`aluno["nota"]`) |

**List comprehension** (muito cobrado): `quadrados = [x**2 for x in numeros]` é equivalente a um
`for` que faz `.append()` em cada iteração — mas em uma linha só.

### 1.2 Funções e Lambda

```python
def calcular_bonus(salario, percentual=10):
    return salario + (salario * percentual / 100)

dobro = lambda x: x * 2   # função anônima, uma linha, sem "def"
```
- Parâmetro com valor padrão (`percentual=10`) é opcional na chamada.
- `lambda` é usada para funções curtas e descartáveis (ex.: dentro de `sorted(key=lambda x: x[0])`,
  que é exatamente como o KNN ordena distâncias — veja a Seção 6).

### 1.3 Programação Orientada a Objetos (POO)

| Conceito | O que é | Exemplo do curso |
|---|---|---|
| **Classe / Objeto** | Classe é o "molde"; objeto é a instância criada a partir dele | `class Pessoa: ...` / `p1 = Pessoa("Tiago", 35)` |
| **Encapsulamento** | Atributos privados (prefixo `__`) protegem o dado interno | `self.__saldo` em `ContaBancaria`, só acessível via métodos |
| **Herança** | Uma classe filha reaproveita atributos/métodos da classe mãe com `super()` | `class Funcionario(Pessoa): super().__init__(...)` |
| **Polimorfismo** | Classes diferentes respondem ao mesmo método de forma diferente | `Cachorro.falar()` e `Gato.falar()` retornam sons diferentes |

```python
class Pessoa:
    def __init__(self, _nome, _idade):
        self.nome = _nome
        self.idade = _idade
    def saudar(self):
        return f"Olá, meu nome é {self.nome} e tenho {self.idade} anos."

class Funcionario(Pessoa):          # herança
    def __init__(self, _nome, _idade, _cargo):
        super().__init__(_nome, _idade)   # chama o construtor da classe mãe
        self.cargo = _cargo
    def saudar(self):                # polimorfismo: sobrescreve o método da mãe
        return f"Olá, eu sou {self.nome}, atuo como {self.cargo}"
```

### 1.4 Numpy — o essencial

```python
import numpy as np

a = np.array([1, 2, 3])
zeros = np.zeros((2, 3))        # matriz 2x3 de zeros
faixa = np.arange(0, 11, 2)     # [0, 2, 4, 6, 8, 10]

# operações são vetorizadas (aplicadas elemento a elemento, sem loop)
soma = np.array([1,2,3]) + np.array([4,5,6])          # [5, 7, 9]
mult = np.array([1,2,3]) * np.array([4,5,6])          # [4, 10, 18]

X.T                # transposta
A @ B              # multiplicação de matrizes (também pode ser A.dot(B))
np.linalg.inv(M)   # inversa de uma matriz
np.sqrt(x), np.sum(x), np.mean(x)
```
Todos os algoritmos das Aulas 3 a 7 neste curso são implementados **só com Numpy** (sem
scikit-learn) — por isso essas operações (`@`, `.T`, `np.linalg.inv`) aparecem em toda prova.

### 1.5 Matplotlib — qual gráfico usar quando

| Gráfico | Função | Quando usar |
|---|---|---|
| Linha | `plt.plot(x, y)` | Evolução/tendência ao longo de uma sequência (ex.: vendas por ano) |
| Barra | `plt.bar(categorias, valores)` | Comparar quantidades entre categorias distintas |
| Dispersão (scatter) | `plt.scatter(x, y)` | Ver a relação entre duas variáveis numéricas — **é o gráfico usado em toda a regressão e no KNN** |
| Histograma | `plt.hist(dados, bins=30)` | Ver a distribuição/frequência de uma variável numérica |
| Pizza | `plt.pie(valores, labels=categorias)` | Mostrar proporção (%) de um total entre poucas categorias |
| Subplots | `plt.subplots(2, 2)` | Vários gráficos lado a lado na mesma figura |

### 1.6 Como carregar dados de um CSV corretamente

O que mais muda de exercício pra exercício não é o algoritmo — é isto: qual arquivo carregar,
quais colunas usar como features/target, e como tratar o conteúdo da coluna. Olhe o cabeçalho do
CSV e responda 3 perguntas antes de escrever o código:

1. Qual coluna é o **y** (o que eu quero prever)?
2. Quais colunas são o **X** (features)?
3. As colunas têm só números limpos, números entre aspas, ou texto/valores faltando?

**Perguntas 1 e 2 — identificando y e X:** o enunciado sempre nomeia o alvo ("prever/estimar o
preço", "saber se sobreviveu" → isso é o `y`, sempre uma única coluna). As features são as
variáveis citadas como "a partir de / com base em / usando" — use só as que o enunciado pede,
mesmo que o CSV tenha muitas outras colunas disponíveis. Confira o nome/posição real no arquivo
antes de codificar (nunca adivinhe):
```python
with open("arquivo.csv") as f:
    print(f.readline())   # mostra o cabeçalho com os nomes das colunas, na ordem
```
Nunca use como feature uma coluna de identificação (`id`) nem a própria coluna do `y`.

**Exemplo real do curso** (`kc_house_data.csv`: `id`(0), `date`(1), `price`(2), `bedrooms`(3),
`bathrooms`(4), `sqft_living`(5), ...):

| Exercício | y (target) | X (features) |
|---|---|---|
| Regressão Simples (Aula 3) | `price` (col. 2) | `sqft_living` (col. 5) |
| Regressão Múltipla (Aula 4) | `price` (col. 2) | `bedrooms`(3), `bathrooms`(4), `sqft_living`(5) |
| Regressão Logística (Aula 5) | `survived` | `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare` |

Repare: o `y` é o mesmo em Simples e Múltipla — o que muda é só quantas/quais features entram em `usecols`.

**Pergunta 3 — qual forma de carregamento usar:**

| Situação do CSV | Como carregar |
|---|---|
| Só números, sem aspas, sem valor faltando (`tumores.csv`) | `np.loadtxt` |
| Números entre aspas `"123"` (`kc_house_data.csv`) | `np.genfromtxt` + limpeza de aspas |
| Texto/categoria e/ou valor faltando (`titanic.csv`) | `csv.DictReader` + tratamento manual |

**Opção 1 — `np.loadtxt` (mais simples):**
```python
import numpy as np
# skiprows pula o cabeçalho (2 se houver linha de comentário extra); usecols escolhe as colunas
dados = np.loadtxt("arquivo.csv", delimiter=',', skiprows=1, usecols=(5, 2))
x, y = dados[:, 0], dados[:, 1]
```

**Opção 2 — `np.genfromtxt` (números entre aspas):**
```python
import numpy as np
dados = np.genfromtxt("arquivo.csv", delimiter=',', skip_header=1, usecols=(2, 3, 4, 5), dtype=str)
dados = np.char.strip(dados, '""').astype(float)   # remove as aspas e converte para número
y, X_features = dados[:, 0], dados[:, 1:]
```

**Opção 3 — `csv.DictReader` (texto/categoria, valores faltando):**
```python
import csv
import numpy as np

def load_data(filename):
    X, y = [], []
    with open(filename, "r", encoding='utf-8') as f:
        for row in csv.DictReader(f):        # cada linha vira um dict, acessível por nome
            if row["survived"] == "":
                continue
            y.append(int(row["survived"]))
            sex = 1 if row["sex"] == "female" else 0           # texto -> número
            age = float(row["age"]) if row["age"] != "" else -1  # marcador de faltante
            X.append([float(row["pclass"]), sex, age, float(row["fare"])])
    return np.array(X), np.array(y).reshape(-1, 1)
```
Depois da Opção 3, ainda falta substituir o marcador de faltante pela média (`fill_missing_age`)
e normalizar as colunas (`normalize`, Seção 4.5) antes de treinar.

---

## 2. Aula 3 — Regressão Linear Simples

### 2.1 Quando usar

Use quando você quer **prever um valor numérico contínuo (y)** a partir de **uma única
variável numérica (x)**, assumindo que a relação entre elas é aproximadamente uma reta.

Exemplos: prever preço de casa a partir da área; prever salário a partir dos anos de experiência.

### 2.2 A fórmula

$$y = mx + b$$

- `m` (coeficiente angular/slope): quanto `y` muda para cada unidade a mais de `x`.
- `b` (intercepto): valor previsto de `y` quando `x = 0`.

O modelo é ajustado pelo **método dos mínimos quadrados**, que encontra o `m` e o `b` que
minimizam a soma dos erros ao quadrado entre o valor real e o previsto:

$$m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} \qquad\qquad b = \bar{y} - m \cdot \bar{x}$$

onde $\bar{x}$ e $\bar{y}$ são as médias de x e y.

### 2.3 Avaliando o ajuste: R² (coeficiente de determinação)

$$R^2 = 1 - \frac{SQR}{SQT} \qquad SQR = \sum (y_i - \hat{y}_i)^2 \quad\text{(soma dos resíduos²)} \qquad SQT = \sum (y_i - \bar{y})^2 \quad\text{(soma total²)}$$

- $R^2$ varia de 0 a 1 (pode ser negativo se o modelo for muito ruim).
- $R^2$ próximo de 1 → o modelo explica bem a variação de y. Próximo de 0 → o modelo não explica quase nada.
- Regra usada em aula: $R^2>0.75$ → bom ajuste; $0.5<R^2\le0.75$ → ajuste moderado; $R^2\le0.5$ → ajuste fraco.

### 2.4 Exemplo resolvido passo a passo

Dataset (horas de estudo `x` → nota `y`):

| x (horas) | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| y (nota) | 3 | 5 | 7 | 9 |

**Passo 1 — médias:** $\bar{x} = \frac{1+2+3+4}{4} = 2.5$, $\bar{y} = \frac{3+5+7+9}{4} = 6$

**Passo 2 — desvios:** $(x-\bar x)$ = [-1.5, -0.5, 0.5, 1.5]; $(y-\bar y)$ = [-3, -1, 1, 3]

**Passo 3 — produtos e quadrados:**

| $(x-\bar x)$ | $(y-\bar y)$ | produto | $(x-\bar x)^2$ |
|---|---|---|---|
| -1.5 | -3 | 4.5 | 2.25 |
| -0.5 | -1 | 0.5 | 0.25 |
| 0.5 | 1 | 0.5 | 0.25 |
| 1.5 | 3 | 4.5 | 2.25 |
| **soma** | | **10** | **5** |

**Passo 4 — m e b:** $m = 10/5 = 2$, $b = 6 - 2\times2.5 = 1$

**Modelo final:** $y = 2x + 1$. Nesse exemplo os pontos caem exatamente sobre a reta (dados
perfeitos), então $SQR=0$ e $R^2 = 1$. Na vida real (e nos exercícios abaixo) os pontos quase
nunca são perfeitos — sempre sobra algum resíduo.

### 2.5 Código (estilo usado em aula — só Numpy)

```python
import numpy as np

x = np.array([1, 2, 3, 4]).reshape(-1, 1)
y = np.array([3, 5, 7, 9]).reshape(-1, 1)

x_mean, y_mean = np.mean(x), np.mean(y)

m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b = y_mean - m * x_mean

y_pred = m * x + b
sqr = np.sum((y - y_pred) ** 2)
sqt = np.sum((y - y_mean) ** 2)
r2 = 1 - (sqr / sqt)

print(f"y = {m:.4f}x + {b:.4f}   |   R² = {r2:.4f}")
```

### 2.6 Exercícios (com gabarito)

> Os 3 exercícios a seguir seguem os mesmos enunciados dos slides de "Exercícios de Revisão e
> Fixação" da Aula 3, agora com dados numéricos para praticar o cálculo completo.

#### Exercício 1 — Marketing x Faturamento

Uma empresa quer entender a relação entre investimento em marketing digital (milhares de R$) e
faturamento mensal (milhares de R$):

| Investimento (x) | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|
| Faturamento (y) | 35 | 45 | 75 | 85 | 110 |

a) O modelo consegue representar bem a relação? Justifique.
b) Qual o coeficiente angular e sua interpretação?
c) Se a empresa investir R$ 15 mil, qual o faturamento previsto?
d) Há indícios de que a relação deixa de ser linear para investimentos muito altos?
e) Só o marketing explica o faturamento, ou outros fatores deveriam entrar no modelo?

<details>
<summary><b>Gabarito — Exercício 1</b></summary>

$\bar x = 6$, $\bar y = 70$. Desvios de x: [-4,-2,0,2,4]; desvios de y: [-35,-25,5,15,40].
Produtos: [140, 50, 0, 30, 160] → soma = **380**. Quadrados de x: [16,4,0,4,16] → soma = **40**.

$m = 380/40 = 9.5$, $b = 70 - 9.5\times6 = 13$.

**Modelo: faturamento = 9,5 × investimento + 13**

Previsões: 32, 51, 70, 89, 108 (para x = 2,4,6,8,10) → resíduos: 3, -6, 5, -4, 2.
$SQR = 9+36+25+16+4 = 90$. $SQT = 35^2+25^2+5^2+15^2+40^2 = 3700$.
$R^2 = 1 - 90/3700 \approx \mathbf{0{,}976}$.

a) Sim — $R^2\approx0.976$ é muito próximo de 1, os pontos ficam bem próximos da reta no gráfico.
b) $m=9.5$: a cada R$ 1 mil investido a mais em marketing, o faturamento esperado aumenta em
   R$ 9,5 mil, mantendo tudo mais constante.
c) $9.5\times15+13 = \mathbf{155{,}5}$ mil reais.
d) x=15 está **fora** do intervalo observado (2 a 10) — é extrapolação. Não há garantia de que a
   reta continue válida; é comum que o retorno de marketing sature em investimentos muito altos
   (retornos marginais decrescentes).
e) Resposta aberta esperada: não — fatores como sazonalidade, concorrência, preço, qualidade do
   produto etc. também afetam o faturamento; um modelo mais realista usaria Regressão Múltipla.

**Código:**
```python
import numpy as np

x = np.array([2, 4, 6, 8, 10])
y = np.array([35, 45, 75, 85, 110])

x_mean, y_mean = np.mean(x), np.mean(y)
m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b = y_mean - m * x_mean

y_pred = m * x + b
r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2)

print(f"m = {m}, b = {b}, R² = {r2:.4f}")
print("Previsão para investimento de 15 mil:", m * 15 + b)
```
</details>

#### Exercício 2 — Experiência x Salário

RH quer prever o salário médio (mil R$) a partir dos anos de experiência:

| Experiência (x) | 1 | 3 | 5 | 7 | 9 |
|---|---|---|---|---|---|
| Salário (y) | 3 | 4 | 6 | 7 | 10 |

a) Existe tendência linear clara? b) Interprete o coeficiente angular. c) Qual o salário previsto
para 25 anos de experiência? Faz sentido? d) Há sinais de progressão não linear? e) Que
limitações tem usar só "anos de experiência" como variável?

<details>
<summary><b>Gabarito — Exercício 2</b></summary>

$\bar x=5$, $\bar y=6$. Desvios x: [-4,-2,0,2,4]; desvios y: [-3,-2,0,1,4].
Produtos: [12,4,0,2,16] → soma=**34**. Quadrados de x: [16,4,0,4,16] → soma=**40**.

$m = 34/40 = 0.85$, $b = 6 - 0.85\times5 = 1.75$

**Modelo: salário = 0,85 × experiência + 1,75**

Previsões: 2.6, 4.3, 6.0, 7.7, 9.4 → resíduos: 0.4, -0.3, 0, -0.7, 0.6.
$SQR = 1.10$. $SQT = 9+4+0+1+16=30$. $R^2 = 1-1.10/30 \approx \mathbf{0{,}963}$.

a) Sim, tendência linear forte ($R^2\approx0.963$).
b) A cada ano a mais de experiência, o salário médio previsto sobe R$ 850 (0,85 mil), tudo mais constante.
c) $0.85\times25+1.75 = \mathbf{23{,}0}$ mil reais. Não é muito realista: 25 está bem fora do
   intervalo observado (1 a 9 anos) — extrapolação longa.
d) Na prática, salários costumam **desacelerar** (não crescer linearmente para sempre); um
   modelo linear tende a superestimar salários de profissionais muito experientes.
e) Não considera cargo, empresa, formação, região, negociação individual etc. — pediria Regressão Múltipla.

**Código:**
```python
import numpy as np

x = np.array([1, 3, 5, 7, 9])
y = np.array([3, 4, 6, 7, 10])

x_mean, y_mean = np.mean(x), np.mean(y)
m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b = y_mean - m * x_mean

y_pred = m * x + b
r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2)

print(f"m = {m}, b = {b}, R² = {r2:.4f}")
print("Previsão para 25 anos de experiência:", m * 25 + b)
```
</details>

#### Exercício 3 — Treinamentos x Produtividade

Gestor de RH quer relacionar nº de treinamentos técnicos no ano com a produtividade média mensal
(0 a 100 pontos):

| Treinamentos (x) | 0 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| Produtividade (y) | 50 | 65 | 70 | 80 | 90 |

a) Há relação linear clara? b) Interprete o coeficiente angular. c) Produtividade prevista para
20 treinamentos? d) Produtividade inicial (0 treinamentos) prevista — é realista? e) O ganho
continua indefinidamente ou tende a se estabilizar? f) Diferença de produtividade entre 5 e 15
treinamentos — justifica o investimento? g) Que outras variáveis poderiam influenciar?

<details>
<summary><b>Gabarito — Exercício 3</b></summary>

$\bar x=10$, $\bar y=71$. Desvios x: [-10,-5,0,5,10]; desvios y: [-21,-6,-1,9,19].
Produtos: [210,30,0,45,190] → soma=**475**. Quadrados x: [100,25,0,25,100] → soma=**250**.

$m = 475/250 = 1.9$, $b = 71 - 1.9\times10 = 52$

**Modelo: produtividade = 1,9 × treinamentos + 52**

Previsões: 52, 61.5, 71, 80.5, 90 → resíduos: -2, 3.5, -1, -0.5, 0.
$SQR=17.5$. $SQT=441+36+1+81+361=920$. $R^2=1-17.5/920 \approx \mathbf{0{,}981}$.

a) Sim, relação linear muito forte ($R^2\approx0.981$).
b) Cada treinamento adicional aumenta a produtividade média prevista em 1,9 ponto.
c) $1.9\times20+52=\mathbf{90}$ pontos.
d) Intercepto $b=52$: produtividade prevista sem nenhum treinamento é 52 pontos — plausível
   (funcionário já tem alguma produtividade base sem treinamento).
e) O modelo linear, por construção, diz que o ganho é constante e infinito — mas na prática é
   comum esse ganho se estabilizar (retornos marginais decrescentes); dentro do intervalo
   observado (0–20) o ajuste linear é bom, extrapolar para muito além disso é arriscado.
f) Em x=5 → 61.5; em x=15 → 80.5 → diferença de **19 pontos** de produtividade. Se o custo dos
   10 treinamentos extras for menor que o valor gerado por 19 pontos a mais de produtividade,
   compensa (resposta depende do contexto de custo/benefício da empresa).
g) Motivação, tipo/qualidade do treinamento, experiência prévia, ferramentas disponíveis, carga
   de trabalho, etc.

**Código:**
```python
import numpy as np

x = np.array([0, 5, 10, 15, 20])
y = np.array([50, 65, 70, 80, 90])

x_mean, y_mean = np.mean(x), np.mean(y)
m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
b = y_mean - m * x_mean

y_pred = m * x + b
r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2)

print(f"m = {m}, b = {b}, R² = {r2:.4f}")
print("Previsão para 20 treinamentos:", m * 20 + b)
print("Diferença entre 15 e 5 treinamentos:", (m * 15 + b) - (m * 5 + b))
```
</details>

---

## 3. Aula 4 — Regressão Linear Múltipla

### 3.1 Quando usar

Igual à regressão simples, mas quando você tem **duas ou mais variáveis explicativas (features)**
para prever um único valor numérico contínuo. Visualmente: 1 feature = reta em 2D; 2 features =
plano em 3D; 3+ features = hiperplano (não dá mais para visualizar diretamente).

### 3.2 A fórmula

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon$$

- $\beta_0$: intercepto.
- $\beta_i$: quanto `y` muda para cada unidade a mais de $x_i$, **mantendo todas as outras
  variáveis constantes** (é a interpretação-chave que a prova cobra).
- $\epsilon$: erro/resíduo — o que o modelo não consegue explicar.

Forma matricial: $\hat{y} = X\cdot\beta$, e a solução que minimiza o erro quadrático (mínimos
quadrados) é a **equação normal**:

$$\beta = (X^T X)^{-1} X^T y$$

onde `X` é a matriz de features **com uma coluna de 1's à esquerda** (para representar $\beta_0$).

### 3.3 Por que não dá sempre para resolver $X\beta = y$ diretamente?

Normalmente há **mais exemplos do que parâmetros** — o sistema é "sobredeterminado", sem solução
exata — e a equação normal usa mínimos quadrados para achar o $\beta$ que **minimiza o erro
total**. Quando o número de exemplos é **igual** ao de parâmetros (como no exemplo abaixo: 3
pontos, 3 parâmetros), há uma solução **exata** que passa por todos os pontos (resíduo zero) —
bom para praticar a conta na mão, mas **não representa um problema real**, onde se usa bem mais
dados do que parâmetros para generalizar.

### 3.4 Exemplo resolvido passo a passo (completando o exemplo das slides)

Dataset: área (x₁), quartos (x₂) → preço (y):

| área | quartos | preço |
|---|---|---|
| 50 | 2 | 200 |
| 80 | 3 | 300 |
| 100 | 4 | 400 |

**Passo 1 — montar X (com coluna de 1's) e y:**

$$X = \begin{bmatrix}1 & 50 & 2\\ 1 & 80 & 3\\ 1 & 100 & 4\end{bmatrix} \qquad y = \begin{bmatrix}200\\300\\400\end{bmatrix}$$

**Passo 2 — calcular $X^TX$:**

$$X^TX = \begin{bmatrix}3 & 230 & 9\\ 230 & 18900 & 740\\ 9 & 740 & 29\end{bmatrix}$$

**Passo 3 — calcular $X^Ty$:**

$$X^Ty = \begin{bmatrix}200+300+400\\ 50\cdot200+80\cdot300+100\cdot400\\ 2\cdot200+3\cdot300+4\cdot400\end{bmatrix} = \begin{bmatrix}900\\74000\\2900\end{bmatrix}$$

**Passo 4 — resolver $X^TX\,\beta = X^Ty$:** montando e resolvendo o sistema de 3 equações a 3
incógnitas (pode ser feito por substituição ou invertendo a matriz), chega-se a:

$$\beta_0 = 0 \qquad \beta_1 = 0 \qquad \beta_2 = 100$$

**Modelo final: preço = 100 × quartos** (a área não entrou no modelo, $\beta_1=0$!). Confira:
quartos=2 → 200 ✓; quartos=3 → 300 ✓; quartos=4 → 400 ✓ — a reta/plano passa exatamente pelos 3
pontos, como esperado (3 exemplos, 3 parâmetros → resíduo zero).

> **Lição:** isso não quer dizer que "área nunca importa" — com só 3 exemplos não há dados
> suficientes para separar o efeito de área e quartos (aqui, quartos sozinho já explica tudo).
> Em problemas reais, use sempre muito mais exemplos do que parâmetros.

### 3.5 Código (estilo usado em aula)

```python
import numpy as np

X_features = np.array([[50, 2], [80, 3], [100, 4]])
y = np.array([200, 300, 400])

X = np.c_[np.ones(X_features.shape[0]), X_features]   # adiciona coluna de 1's (bias)
beta = np.linalg.inv(X.T @ X) @ X.T @ y

print("Coeficientes:", beta)     # [beta0, beta1, beta2]

# avaliação com R² (mesma fórmula da regressão simples)
y_pred = X @ beta
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
```

### 3.6 Exercícios (com gabarito)

#### Exercício 1 — Nota final a partir de horas de estudo e faltas

Uma escola quer prever a nota final de um aluno (0 a 10) a partir das horas de estudo semanais
($x_1$) e do número de faltas no mês ($x_2$):

| Aluno | Horas de estudo ($x_1$) | Faltas ($x_2$) | Nota final (y) |
|---|---|---|---|
| A | 10 | 1 | 7 |
| B | 6 | 3 | 3 |
| C | 8 | 0 | 7 |

a) Monte a matriz $X$ (com a coluna de 1's) e o vetor $y$.
b) Calcule $\beta$ e escreva a equação do modelo.
c) Qual a nota prevista para um aluno com 12 horas de estudo e 2 faltas?
d) Interprete o sinal de $\beta_1$ e de $\beta_2$.
e) Por que usar Regressão Múltipla aqui, e não Simples?

<details>
<summary><b>Gabarito — Exercício 1</b></summary>

$$X=\begin{bmatrix}1&10&1\\1&6&3\\1&8&0\end{bmatrix}\quad y=\begin{bmatrix}7\\3\\7\end{bmatrix}$$

Resolvendo o sistema $X\beta=y$ (3 equações, 3 incógnitas):
das linhas A e C elimina-se $\beta_0$ e chega-se a $\beta_1=0.5$ e $\beta_2=-1$, e então
$\beta_0 = 7 - 8(0.5) = 3$.

**Modelo: nota = 3 + 0,5 × horas − 1 × faltas**

(Confira: A → $3+5-1=7$ ✓; B → $3+3-3=3$ ✓; C → $3+4-0=7$ ✓.)

c) $3 + 0.5\times12 - 1\times2 = 3+6-2=\mathbf{7}$.
d) $\beta_1=0.5>0$: mais horas de estudo aumentam a nota (esperado). $\beta_2=-1<0$: cada falta
   reduz a nota prevista em 1 ponto (esperado).
e) Porque há **duas** variáveis explicativas (horas e faltas) influenciando a nota ao mesmo
   tempo — Regressão Simples só suporta uma variável.

**Código:**
```python
import numpy as np

X_features = np.array([[10, 1], [6, 3], [8, 0]])   # horas, faltas
y = np.array([7, 3, 7])

X = np.c_[np.ones(X_features.shape[0]), X_features]   # adiciona coluna de 1's (bias)
beta = np.linalg.inv(X.T @ X) @ X.T @ y

print("beta =", beta)   # [beta0, beta1, beta2]

novo_aluno = np.array([1, 12, 2])   # bias, horas=12, faltas=2
print("Nota prevista:", novo_aluno @ beta)
```
</details>

#### Exercício 2 — Aluguel a partir de área e distância ao centro

Uma imobiliária quer prever o aluguel (R$) a partir da área (m², $x_1$) e da distância ao centro
(km, $x_2$):

| Imóvel | Área ($x_1$) | Distância ($x_2$) | Aluguel (y) |
|---|---|---|---|
| A | 40 | 5 | 750 |
| B | 60 | 2 | 1080 |
| C | 30 | 10 | 550 |

a) Calcule $\beta_0, \beta_1, \beta_2$. b) Escreva o modelo. c) Qual o aluguel previsto para um
imóvel de 50 m² a 4 km do centro? d) O sinal de $\beta_2$ faz sentido?

<details>
<summary><b>Gabarito — Exercício 2</b></summary>

Resolvendo o sistema: $\beta_0=200$, $\beta_1=15$, $\beta_2=-10$.

**Modelo: aluguel = 200 + 15 × área − 10 × distância**

(Confira: A → $200+600-50=750$ ✓; B → $200+900-20=1080$ ✓; C → $200+450-100=550$ ✓.)

c) $200 + 15\times50 - 10\times4 = 200+750-40=\mathbf{R\$\,910}$.
d) Sim — $\beta_2=-10<0$ significa que, quanto mais longe do centro, menor o aluguel (mantendo a
   área constante), o que é o esperado no mercado imobiliário.

**Código:**
```python
import numpy as np

X_features = np.array([[40, 5], [60, 2], [30, 10]])   # área, distância
y = np.array([750, 1080, 550])

X = np.c_[np.ones(X_features.shape[0]), X_features]
beta = np.linalg.inv(X.T @ X) @ X.T @ y

print("beta =", beta)   # [beta0, beta1, beta2]

novo_imovel = np.array([1, 50, 4])   # bias, área=50, distância=4
print("Aluguel previsto:", novo_imovel @ beta)
```
</details>

---

## 4. Aula 5 — Regressão Logística

### 4.1 Quando usar

Use quando a variável a prever é **categórica/binária** (0/1, sim/não, sobreviveu/não) e você quer
também saber a **probabilidade** de cada classe. **Nunca use regressão linear aqui**: uma reta
pode prever valores fora de [0, 1], o que não faz sentido como probabilidade.

### 4.2 A função sigmoide

A regressão logística aplica a função sigmoide sobre a combinação linear das variáveis, o que
"espreme" qualquer valor para o intervalo (0, 1):

$$z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n \qquad\qquad \sigma(z) = \frac{1}{1+e^{-z}}$$

- $\sigma(z)$ é interpretado como a **probabilidade** de a classe ser 1.
- **Fronteira de decisão:** se $\sigma(z) \ge 0.5$ → classifica como 1; senão → classifica como 0.
  ($\sigma(z)=0.5$ exatamente quando $z=0$.)

Valores de referência úteis (usando $e\approx2{,}71828$):

| z | -3 | -2 | -1 | -0.5 | 0 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|---|---|
| $\sigma(z)$ | 0,047 | 0,119 | 0,269 | 0,378 | **0,5** | 0,622 | 0,731 | 0,881 |

### 4.3 Função de custo (log loss / entropia cruzada)

Não se usa erro quadrático aqui (a superfície de erro ficaria "cheia de vales", dificultando a
otimização). Usa-se a **entropia cruzada binária**:

$$J(\beta) = -\frac{1}{m}\sum_{i=1}^{m}\Big[y_i \ln(h_i) + (1-y_i)\ln(1-h_i)\Big] \qquad h_i=\sigma(z_i)$$

- Quanto mais a probabilidade prevista $h_i$ se aproxima do valor real $y_i$, **menor** o custo.
- O treinamento ajusta $\beta$ por **gradiente descendente**, minimizando $J(\beta)$ a cada época:
  $\beta \leftarrow \beta - \alpha \cdot \frac{1}{m}X^T(h-y)$, onde $\alpha$ é a taxa de aprendizado.

### 4.4 Exemplo resolvido passo a passo

Um modelo já treinado prevê se um cliente compra (1) ou não (0) um produto, usando
$z = -4 + 0.5\cdot\text{tempo no site (min)} + 1\cdot\text{visitas anteriores}$.

| Cliente | tempo | visitas | z | $\sigma(z)$ | Classificação (limiar 0,5) |
|---|---|---|---|---|---|
| 1 | 6 | 2 | $-4+3+2=1$ | 0,731 (73,1%) | **1 — compra** |
| 2 | 2 | 0 | $-4+1+0=-3$ | 0,047 (4,7%) | **0 — não compra** |
| 3 | 4 | 1 | $-4+2+1=-1$ | 0,269 (26,9%) | **0 — não compra** |

### 4.5 Código (estilo usado em aula)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, beta):
    m = len(y)
    h = sigmoid(X @ beta)
    eps = 1e-8
    return -(1/m) * np.sum(y*np.log(h+eps) + (1-y)*np.log(1-h+eps))

def gradient_descent(X, y, beta, lr, epochs):
    m = len(y)
    for i in range(epochs):
        h = sigmoid(X @ beta)
        gradient = (1/m) * (X.T @ (h - y))
        beta = beta - lr * gradient
    return beta

def predict(X, beta):
    return (sigmoid(X @ beta) >= 0.5).astype(int)
```
No pipeline completo do curso (dataset do Titanic), antes de treinar sempre se faz: tratar
valores ausentes (`fill_missing_age`), **normalizar** as features (`(X - média)/desvio`) e
adicionar a coluna de bias — nessa ordem.

### 4.6 Exercícios (com gabarito)

#### Exercício 1 — Classificando clientes

Um banco usa $z = -3 + 0.8\cdot\text{renda (em milhares)} - 0.5\cdot\text{nº de dívidas ativas}$
para prever se aprova (1) ou não (0) um crédito. Calcule, para os clientes abaixo, o $z$, a
probabilidade $\sigma(z)$ (use a tabela da seção 4.2) e a classificação final:

| Cliente | Renda | Dívidas |
|---|---|---|
| P | 5 | 1 |
| Q | 3 | 0 |
| R | 4 | 3 |

<details>
<summary><b>Gabarito — Exercício 1</b></summary>

- P: $z=-3+4-0.5=0.5 \to \sigma(0.5)=0{,}622$ (62,2%) → **1 — crédito aprovado**
- Q: $z=-3+2.4-0=-0.6 \to \sigma(-0.6)\approx0{,}354$ (35,4%) → **0 — crédito negado**
- R: $z=-3+3.2-1.5=-1.3 \to \sigma(-1.3)\approx0{,}214$ (21,4%) → **0 — crédito negado**

(Valores de $\sigma$ para z não tabelados podem ser calculados com $\sigma(z)=1/(1+e^{-z})$,
$e\approx2{,}71828$.)

**Código:**
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

beta0, beta1, beta2 = -3, 0.8, -0.5
clientes = {"P": (5, 1), "Q": (3, 0), "R": (4, 3)}   # (renda, dívidas)

for nome, (renda, dividas) in clientes.items():
    z = beta0 + beta1 * renda + beta2 * dividas
    prob = sigmoid(z)
    classe = 1 if prob >= 0.5 else 0
    print(f"{nome}: z={z:.2f}, prob={prob:.4f}, classe={classe}")
```
</details>

#### Exercício 2 — Comparando o custo (log loss) de dois modelos

Um modelo previu as probabilidades $p=[0.8,\ 0.3,\ 0.6]$ para 3 exemplos cujos valores reais são
$y=[1,\ 0,\ 1]$.

a) Calcule o log loss desse modelo (use $\ln(0.8)=-0.2231$, $\ln(0.7)=-0.3567$, $\ln(0.6)=-0.5108$).
b) Um segundo modelo previu $p=[0.95,\ 0.1,\ 0.9]$ para os mesmos $y$ (use $\ln(0.95)=-0.0513$,
   $\ln(0.9)=-0.1054$). Calcule o log loss dele.
c) Qual modelo é melhor? Por quê?

<details>
<summary><b>Gabarito — Exercício 2</b></summary>

a) $J = -\frac{1}{3}\big[\ln(0.8) + \ln(0.7) + \ln(0.6)\big] = -\frac{1}{3}(-0.2231-0.3567-0.5108) = -\frac{-1.0906}{3} \approx \mathbf{0{,}3635}$

b) $J = -\frac{1}{3}\big[\ln(0.95) + \ln(0.9) + \ln(0.9)\big] = -\frac{1}{3}(-0.0513-0.1054-0.1054) \approx \mathbf{0{,}0874}$

c) O **segundo modelo** é melhor: seu log loss (≈0,087) é bem menor que o do primeiro (≈0,364).
   Quanto menor o custo, mais as probabilidades previstas se aproximam dos valores reais.

**Código:**
```python
import numpy as np

def log_loss(y, p):
    eps = 1e-8
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

y = np.array([1, 0, 1])
p1 = np.array([0.8, 0.3, 0.6])
p2 = np.array([0.95, 0.1, 0.9])

print("Custo modelo 1:", log_loss(y, p1))
print("Custo modelo 2:", log_loss(y, p2))
```
</details>

---

## 5. Aula 6 — Matriz de Confusão e Métricas de Classificação

### 5.1 Quando usar

Não é um algoritmo de predição — é como você **avalia** um modelo de classificação já treinado
(seja Regressão Logística ou KNN). Toda vez que o exercício perguntar "quantos o modelo acertou",
"qual a taxa de falsos positivos", "o modelo é bom?" — é matriz de confusão.

### 5.2 A matriz de confusão

Para um problema binário (classe positiva = 1, negativa = 0):

| | Previsto: Positivo | Previsto: Negativo |
|---|---|---|
| **Real: Positivo** | VP (Verdadeiro Positivo) | FN (Falso Negativo) |
| **Real: Negativo** | FP (Falso Positivo) | VN (Verdadeiro Negativo) |

- **VP**: modelo previu 1, era 1 (acertou). **VN**: previu 0, era 0 (acertou).
- **FP** ("alarme falso"): previu 1, mas era 0. **FN** (o pior tipo de erro em contextos de
  saúde/segurança): previu 0, mas era 1 — o modelo **deixou passar** um caso positivo.

### 5.3 As métricas

$$\text{Acurácia} = \frac{VP+VN}{VP+VN+FP+FN} \qquad \text{Precisão} = \frac{VP}{VP+FP} \qquad \text{Revocação (Recall)} = \frac{VP}{VP+FN}$$

$$\text{Especificidade} = \frac{VN}{VN+FP} \qquad\qquad F_1 = 2\cdot\frac{\text{Precisão}\times\text{Recall}}{\text{Precisão}+\text{Recall}}$$

| Métrica | Responde a pergunta... | Use quando... |
|---|---|---|
| **Acurácia** | "De tudo, quanto o modelo acertou?" | Classes equilibradas (~50/50) |
| **Precisão** | "Dos que o modelo disse que eram positivos, quantos realmente eram?" | O custo de um **falso positivo** é alto (ex.: acusar um e-mail bom de spam) |
| **Revocação (Recall)** | "De todos os positivos reais, quantos o modelo conseguiu achar?" | O custo de um **falso negativo** é alto (ex.: deixar passar uma doença grave) |
| **Especificidade** | "Dos negativos reais, quantos o modelo identificou corretamente?" | Importante avaliar junto com a recall em diagnósticos |
| **F1-score** | Equilíbrio entre precisão e recall | Dados desbalanceados, quando as duas importam |

> ⚠️ **Armadilha clássica de prova:** em dados **desbalanceados** (ex.: doença rara, fraude),
> a **acurácia pode ser enganosamente alta** mesmo que o modelo erre quase todos os casos
> positivos. Veja o Exercício 2 abaixo.

### 5.4 Exemplo resolvido passo a passo

Matriz de confusão de um classificador: VP=40, VN=50, FP=5, FN=5 (total = 100).

- Acurácia $=\frac{40+50}{100}=0{,}90$ (90%)
- Precisão $=\frac{40}{40+5}=\frac{40}{45}\approx0{,}889$ (88,9%)
- Revocação $=\frac{40}{40+5}=\frac{40}{45}\approx0{,}889$ (88,9%)
- Especificidade $=\frac{50}{50+5}=\frac{50}{55}\approx0{,}909$ (90,9%)
- $F_1 = 2\cdot\frac{0.889\times0.889}{0.889+0.889}=0{,}889$ (como precisão = recall aqui, F1 = os dois)

### 5.5 Código (estilo usado em aula)

```python
def confusion_matrix(y_true, y_pred):
    y_t, y_p = y_true.flatten(), y_pred.flatten()
    VP = np.sum((y_t == 1) & (y_p == 1))
    VN = np.sum((y_t == 0) & (y_p == 0))
    FP = np.sum((y_t == 0) & (y_p == 1))
    FN = np.sum((y_t == 1) & (y_p == 0))
    return VP, VN, FP, FN

def calculate_metrics(VP, VN, FP, FN):
    acc = (VP + VN) / (VP + VN + FP + FN)
    prec = VP / (VP + FP)
    rec = VP / (VP + FN)
    spec = VN / (VN + FP)
    f1 = 2 * (prec * rec) / (prec + rec)
    return acc, prec, rec, spec, f1
```

### 5.6 Exercícios (com gabarito)

#### Exercício 1 — Métricas básicas

Um modelo de classificação de spam gerou VP=40, VN=50, FP=5, FN=5. Calcule acurácia, precisão,
recall, especificidade e F1 (você pode conferir com os valores da Seção 5.4).

<details>
<summary><b>Gabarito — Exercício 1</b></summary>

Acurácia = 0,90 · Precisão = 0,889 · Recall = 0,889 · Especificidade = 0,909 · F1 = 0,889
(mesmos cálculos do exemplo resolvido acima).

**Código:**
```python
VP, VN, FP, FN = 40, 50, 5, 5

acc = (VP + VN) / (VP + VN + FP + FN)
prec = VP / (VP + FP)
rec = VP / (VP + FN)
spec = VN / (VN + FP)
f1 = 2 * (prec * rec) / (prec + rec)

print(f"Acurácia={acc:.4f}  Precisão={prec:.4f}  Recall={rec:.4f}  Especificidade={spec:.4f}  F1={f1:.4f}")
```
</details>

#### Exercício 2 — A armadilha da acurácia (dados desbalanceados)

Um exame para detectar uma doença rara foi aplicado em 200 pessoas. Das 18 pessoas realmente
doentes, o exame identificou corretamente 8 (VP=8) e errou 10 (FN=10). Das 182 pessoas
saudáveis, o exame acertou 180 (VN=180) e errou 2 (FP=2).

a) Monte a matriz de confusão e calcule acurácia, precisão, recall, especificidade e F1.
b) A acurácia de ~94% indica que o exame é excelente? O que a recall revela?
c) Nesse contexto (doença rara), qual métrica importa mais: precisão ou recall? Por quê?

<details>
<summary><b>Gabarito — Exercício 2</b></summary>

a) Acurácia $=\frac{8+180}{200}=0{,}94$ (94%). Precisão $=\frac{8}{8+2}=0{,}80$ (80%).
   Recall $=\frac{8}{8+10}=\frac{8}{18}\approx0{,}444$ (44,4%). Especificidade
   $=\frac{180}{182}\approx0{,}989$ (98,9%). $F_1=2\cdot\frac{0.8\times0.444}{0.8+0.444}\approx0{,}571$.

b) **Não.** A acurácia de 94% parece ótima só porque a maioria das pessoas é saudável (a classe
   negativa domina o dataset). A recall de apenas 44,4% revela que o exame **deixa passar mais
   da metade dos doentes reais** (10 de 18) — um resultado péssimo para um exame médico.

c) **Recall** é a métrica mais importante aqui: em doenças, o custo de um falso negativo (deixar
   um doente sem diagnóstico) é muito mais grave que o de um falso positivo (mandar uma pessoa
   saudável fazer um exame extra de confirmação).

**Código:**
```python
VP, VN, FP, FN = 8, 180, 2, 10

acc = (VP + VN) / (VP + VN + FP + FN)
prec = VP / (VP + FP)
rec = VP / (VP + FN)
spec = VN / (VN + FP)
f1 = 2 * (prec * rec) / (prec + rec)

print(f"Acurácia={acc:.4f}  Precisão={prec:.4f}  Recall={rec:.4f}  Especificidade={spec:.4f}  F1={f1:.4f}")
```
</details>

---

## 6. Aula 7 — KNN (K-Nearest Neighbors)

### 6.1 Quando usar

KNN serve tanto para **classificação** quanto para **regressão**, e é uma boa escolha quando:
- Você não quer assumir uma fórmula/fronteira fixa entre as variáveis (é **não-paramétrico**) —
  útil para relações complexas/não-lineares.
- A lógica natural do problema é "olhar para os casos mais parecidos".
- O dataset não é gigante — KNN recalcula a distância a **todos** os pontos a cada predição, o
  que fica lento em bases grandes.

É chamado de **"aprendizado preguiçoso" (lazy learning)**: não constrói modelo no treino, só
guarda os dados — todo o trabalho acontece na hora de prever.

### 6.2 O algoritmo passo a passo

1. **Escolher K** (nº de vizinhos a consultar).
2. **Calcular a distância** do novo ponto a todos os pontos de treino.
3. **Selecionar os K pontos mais próximos** (menores distâncias).
4. **Decidir a saída**:
   - Classificação → **voto majoritário** da classe entre os K vizinhos.
   - Regressão → **média** dos valores dos K vizinhos.

### 6.3 Métricas de distância

$$\text{Euclidiana: } d(p,q)=\sqrt{\sum_{i=1}^n (p_i-q_i)^2} \qquad\qquad \text{Manhattan: } d(p,q)=\sum_{i=1}^n |p_i-q_i|$$

- Euclidiana: distância "em linha reta" — a mais comum.
- Manhattan: soma das diferenças absolutas — útil em espaços de muitas dimensões.

### 6.4 Como escolher K

- **K pequeno** → modelo muito sensível a ruído/outliers → risco de **overfitting**.
- **K grande** → fronteira de decisão muito suavizada → risco de **underfitting** (perde padrões).
- Normalmente se escolhe **K ímpar** (em classificação binária) para evitar empates na votação.
- O K ideal costuma ser encontrado testando vários valores com **validação cruzada**.

### 6.5 Vantagens e desvantagens

| Vantagens | Desvantagens |
|---|---|
| Simples e intuitivo | Custo computacional alto (recalcula tudo a cada predição) |
| Serve para classificação e regressão | Sensível a atributos irrelevantes |
| Não-paramétrico (sem suposição sobre a distribuição dos dados) | "Maldição da dimensionalidade" (perde eficácia com muitas features) |
| Fácil adicionar novos dados (não precisa retreinar) | **Precisa normalizar os dados antes** (senão uma variável em escala maior domina a distância) |

### 6.6 Exemplo resolvido passo a passo — classificação

Dataset (tamanho do tumor, textura → classe: 0=benigno, 1=maligno):

| Ponto | tamanho | textura | classe |
|---|---|---|---|
| P1 | 2 | 3 | 0 |
| P2 | 3 | 2 | 0 |
| P3 | 8 | 8 | 1 |
| P4 | 7 | 9 | 1 |
| P5 | 1 | 1 | 0 |
| P6 | 9 | 7 | 1 |

Novo tumor: (6, 7). Classificar com **K=3** (distância Euclidiana).

| Ponto | cálculo | distância |
|---|---|---|
| P1 | $\sqrt{4^2+4^2}=\sqrt{32}$ | 5,657 |
| P2 | $\sqrt{3^2+5^2}=\sqrt{34}$ | 5,831 |
| P3 | $\sqrt{2^2+1^2}=\sqrt{5}$ | **2,236** |
| P4 | $\sqrt{1^2+2^2}=\sqrt{5}$ | **2,236** |
| P5 | $\sqrt{5^2+6^2}=\sqrt{61}$ | 7,810 |
| P6 | $\sqrt{3^2+0^2}=\sqrt{9}$ | **3,0** |

Os 3 vizinhos mais próximos são **P3, P4, P6** — todos de classe **1**. Votação: 3 votos para 1,
0 votos para 0 → **novo tumor classificado como maligno (1)**.

*(Se K fosse 5, entrariam também P1 e P2 (classe 0), dando 3 votos para 1 e 2 votos para 0 —
a classificação continuaria maligno, mostrando que essa previsão é razoavelmente estável a K.)*

### 6.7 Exemplo resolvido passo a passo — regressão

KNN também prevê números: preço de imóveis (área → preço), K=3, novo imóvel com área=58 m²:

| Área | 50 | 55 | 60 | 85 | 90 |
|---|---|---|---|---|---|
| Preço | 200 | 210 | 220 | 380 | 400 |

Distâncias (1D, valor absoluto) até área=58: |58-50|=8, |58-55|=3, |58-60|=2, |58-85|=27, |58-90|=32.
Os 3 mais próximos são área=60 (d=2, preço 220), área=55 (d=3, preço 210), área=50 (d=8, preço 200).

**Previsão (média dos 3 vizinhos) = (220+210+200)/3 = 210.**

### 6.8 Código (estilo usado em aula)

```python
import numpy as np
from collections import Counter

def calcular_distancia_euclidiana(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def encontrar_vizinhos(X_train, y_train, ponto_teste, k):
    distancias = [(calcular_distancia_euclidiana(p, ponto_teste), y_train[i])
                  for i, p in enumerate(X_train)]
    distancias.sort(key=lambda x: x[0])
    return [d[1] for d in distancias[:k]]

def prever_classificacao(vizinhos):
    return Counter(vizinhos).most_common(1)[0][0]     # classificação: voto majoritário
    # para regressão, seria: return np.mean(vizinhos)
```

### 6.9 Exercícios (com gabarito)

#### Exercício 1 — Classificação com K=3 e K=5

Use o mesmo dataset da Seção 6.6, mas classifique o novo ponto **(4, 4)**.

a) Calcule a distância Euclidiana de (4,4) a cada um dos 6 pontos.
b) Quais são os 3 vizinhos mais próximos (K=3)? Qual a classificação?
c) E com K=5? A classificação muda?
d) Por que normalizar os dados antes de aplicar KNN costuma ser importante?

<details>
<summary><b>Gabarito — Exercício 1</b></summary>

a) P1(2,3): $\sqrt{2^2+1^2}=\sqrt5\approx2{,}236$. P2(3,2): $\sqrt{1^2+2^2}=\sqrt5\approx2{,}236$.
   P3(8,8): $\sqrt{4^2+4^2}=\sqrt{32}\approx5{,}657$. P4(7,9): $\sqrt{3^2+5^2}=\sqrt{34}\approx5{,}831$.
   P5(1,1): $\sqrt{3^2+3^2}=\sqrt{18}\approx4{,}243$. P6(9,7): $\sqrt{5^2+3^2}=\sqrt{34}\approx5{,}831$.

b) Ordenando: P1(2,236), P2(2,236), P5(4,243) são os 3 mais próximos → classes 0, 0, 0 →
   **classificado como benigno (0)** (votação unânime).

c) Com K=5, entram também P3 e P4 (classe 1 cada) → votos: 3 para classe 0, 2 para classe 1 →
   **continua benigno (0)**, a classificação não muda (mas a margem fica mais apertada).

d) Porque o cálculo de distância trata todas as variáveis igualmente em valor numérico — se uma
   variável tiver uma escala muito maior que a outra (ex.: uma em milhares e outra entre 0 e 1),
   ela dominaria a distância mesmo sem ser mais importante. Normalizar coloca todas as variáveis
   na mesma escala antes de medir a distância.

**Código:**
```python
import numpy as np
from collections import Counter

X_train = np.array([[2, 3], [3, 2], [8, 8], [7, 9], [1, 1], [9, 7]])
y_train = np.array([0, 0, 1, 1, 0, 1])

def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))

novo_ponto = np.array([4, 4])
distancias = sorted(
    [(dist(p, novo_ponto), y_train[i]) for i, p in enumerate(X_train)],
    key=lambda t: t[0]
)

for k in [3, 5]:
    vizinhos = [classe for _, classe in distancias[:k]]
    classificacao = Counter(vizinhos).most_common(1)[0][0]
    print(f"K={k}: vizinhos={vizinhos} -> classificação={classificacao}")
```
</details>

#### Exercício 2 — KNN regressão

Usando a tabela da Seção 6.7, preveja o preço de um imóvel de **72 m²** com K=3.

a) Calcule as distâncias. b) Quais os 3 vizinhos mais próximos? c) Qual o preço previsto?
d) Esse mesmo problema também poderia ser resolvido com Regressão Linear Simples — nesse caso,
qual você escolheria? Justifique.

<details>
<summary><b>Gabarito — Exercício 2</b></summary>

a) |72-50|=22, |72-55|=17, |72-60|=12, |72-85|=13, |72-90|=18.

b) Os 3 menores são: área=60 (d=12, preço 220), área=85 (d=13, preço 380), área=55 (d=17, preço 210).

c) Previsão $=(220+380+210)/3 = 810/3 = \mathbf{270}$.

d) Resposta aberta, mas a esperada: como a relação entre área e preço parece ser bem linear e
   crescente (padrão visível nos dados), a **Regressão Linear Simples** tende a generalizar
   melhor e é mais simples de interpretar (dá um coeficiente único de "R$ por m²"); o KNN, aqui,
   é mais sensível a como os pontos vizinhos estão distribuídos (repare que a previsão do KNN
   "pulou" de 210 para 270 entre imóveis de 58 e 72 m², um comportamento mais irregular que uma
   reta contínua).

**Código:**
```python
import numpy as np

areas = np.array([50, 55, 60, 85, 90])
precos = np.array([200, 210, 220, 380, 400])

novo = 72
distancias = sorted(zip(np.abs(areas - novo), precos), key=lambda t: t[0])

k = 3
vizinhos = [preco for _, preco in distancias[:k]]
print("Vizinhos:", vizinhos)
print("Preço previsto:", np.mean(vizinhos))
```
</details>

---

## 7. Guia de Decisão — Qual algoritmo usar em cada exercício

Esta é a parte mais importante para a prova: a maioria dos enunciados **não diz** qual algoritmo
usar — você precisa identificar pelas pistas do problema. Siga este fluxo:

### Passo 1 — O que você quer prever (a variável alvo/y) é um número contínuo ou uma classe/categoria?

- **Número contínuo** (preço, salário, nota, tempo, quantidade...) → é um problema de
  **Regressão** → vá para o **Passo 2**.
- **Classe/categoria** (sim/não, 0/1, aprovado/reprovado, benigno/maligno, spam/não-spam...) →
  é um problema de **Classificação** → vá para o **Passo 3**.

### Passo 2 — (Regressão) Quantas variáveis explicativas (features) o problema fornece?

- **Apenas 1 variável** → **Regressão Linear Simples**.
- **2 ou mais variáveis** → **Regressão Linear Múltipla**.
- Se o enunciado disser algo como "com base nos casos/imóveis/alunos **mais parecidos**" (em vez
  de pedir uma fórmula/equação) → considere **KNN (regressão)** em vez de regressão linear.

### Passo 3 — (Classificação) Como o problema quer que a decisão seja tomada?

- O enunciado fala em **"probabilidade de..."**, pede para "interpretar o peso/coeficiente de
  cada variável", ou o contexto sugere uma fronteira relativamente **linear** entre as classes →
  **Regressão Logística**.
- O enunciado fala em comparar com **"os casos mais parecidos/vizinhos"**, ou não há necessidade
  de interpretar coeficientes/pesos, apenas comparar por similaridade → **KNN (classificação)**.

### Passo 4 — O enunciado pergunta sobre o desempenho do modelo?

Se a pergunta é do tipo "quantos o modelo acertou/errou", "qual a taxa de falso positivo/falso
negativo", "o modelo é bom?", "compare dois modelos" → isso **não é** uma técnica de predição, é
avaliação → **Matriz de Confusão e métricas** (Acurácia, Precisão, Recall, Especificidade, F1).
Essa etapa vem **depois** de qualquer classificador (Logística ou KNN) já ter feito as previsões.

### Tabela de palavras-chave do enunciado

| Se o enunciado menciona... | Provável algoritmo |
|---|---|
| "prever o valor/preço/salário/faturamento" (número) + 1 variável | Regressão Linear Simples |
| "prever o valor/preço/salário" + 2 ou mais variáveis | Regressão Linear Múltipla |
| "sim ou não", "aprovado ou reprovado", "sobreviveu ou não", "0 ou 1" | Classificação (Logística ou KNN) |
| "qual a probabilidade de..." | Regressão Logística |
| "interpretar o peso/coeficiente de cada variável" | Regressão Logística (ou Múltipla, se o alvo for numérico) |
| "com base nos K vizinhos/casos mais parecidos" | KNN |
| "classificar um novo caso comparando com o histórico" | KNN |
| "quantos o modelo acertou", "falso positivo/negativo", "taxa de erro/acerto" | Matriz de Confusão |
| "o modelo é bom mesmo com dados desbalanceados?" | Matriz de Confusão (recall/precisão/F1, não acurácia) |

### Mini-exercícios — "identifique o algoritmo" (com gabarito)

Para cada enunciado, diga **qual algoritmo/técnica** usar e por quê.

1. "Uma seguradora quer prever se um cliente vai cancelar o plano (sim/não) com base na idade,
   tempo de contrato e nº de reclamações, e também quer saber a probabilidade de cada cliente
   cancelar."
2. "Uma imobiliária quer estimar o valor de venda de um imóvel (R$) usando apenas a área construída."
3. "A mesma imobiliária agora quer usar área, nº de quartos e idade do imóvel para estimar o valor de venda."
4. "Um hospital quer classificar um novo paciente em 'grupo de risco A' ou 'B' comparando seus
   exames com os exames dos pacientes anteriores mais parecidos."
5. "Depois de rodar o modelo de diagnóstico, o hospital quer saber quantos pacientes doentes o
   modelo detectou corretamente e quantos ele deixou passar."
6. "Uma escola de idiomas quer prever a nota de proficiência (0–100) de um novo aluno com base
   na nota dos 5 alunos mais parecidos em perfil (idade, nível anterior, horas de estudo)."
7. "Um banco quer saber a probabilidade de um cliente pagar ou não um empréstimo."
8. "Uma fábrica quer comparar dois modelos de classificação de peças defeituosas e decidir qual
   deles erra menos ao classificar peças realmente defeituosas como boas."

<details>
<summary><b>Gabarito comentado</b></summary>

1. **Regressão Logística** — alvo binário (cancelar sim/não) e o enunciado pede explicitamente
   "probabilidade".
2. **Regressão Linear Simples** — alvo numérico contínuo (preço), apenas 1 variável (área).
3. **Regressão Linear Múltipla** — mesmo alvo numérico, agora com 3 variáveis.
4. **KNN (classificação)** — a palavra-chave é "comparando com os... mais parecidos": decisão
   por vizinhança, não por fórmula/coeficientes.
5. **Matriz de Confusão** — "detectou corretamente" (VP) e "deixou passar" (FN) são exatamente os
   conceitos da matriz de confusão; aqui a métrica mais relevante seria a **Revocação (Recall)**.
6. **KNN (regressão)** — alvo numérico (nota 0–100), mas a técnica pedida é claramente baseada em
   vizinhança ("5 alunos mais parecidos" = K=5).
7. **Regressão Logística** — alvo binário (pagar ou não) + palavra "probabilidade".
8. **Matriz de Confusão** — "erra menos ao classificar peças defeituosas como boas" é um falso
   negativo (peça ruim classificada como boa) → comparar modelos pela **Revocação (Recall)**.
</details>

---

## 8. Cola Final — Todas as fórmulas em uma tabela

| Tema | Fórmula-chave |
|---|---|
| Regressão Linear Simples | $y=mx+b$; $\ m=\dfrac{\sum(x-\bar x)(y-\bar y)}{\sum(x-\bar x)^2}$; $\ b=\bar y - m\bar x$ |
| R² (qualquer regressão) | $R^2=1-\dfrac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}$ |
| Regressão Linear Múltipla | $y=\beta_0+\beta_1x_1+\dots+\beta_nx_n$; $\ \beta=(X^TX)^{-1}X^Ty$ |
| Regressão Logística — sigmoide | $\sigma(z)=\dfrac{1}{1+e^{-z}}$, com $z=\beta_0+\beta_1x_1+\dots$; classifica 1 se $\sigma(z)\ge0{,}5$ |
| Regressão Logística — custo | $J=-\dfrac{1}{m}\sum\big[y\ln(h)+(1-y)\ln(1-h)\big]$ |
| Matriz de Confusão — Acurácia | $\dfrac{VP+VN}{VP+VN+FP+FN}$ |
| Matriz de Confusão — Precisão | $\dfrac{VP}{VP+FP}$ |
| Matriz de Confusão — Recall | $\dfrac{VP}{VP+FN}$ |
| Matriz de Confusão — Especificidade | $\dfrac{VN}{VN+FP}$ |
| Matriz de Confusão — F1 | $2\cdot\dfrac{\text{Precisão}\times\text{Recall}}{\text{Precisão}+\text{Recall}}$ |
| KNN — Distância Euclidiana | $\sqrt{\sum_i(p_i-q_i)^2}$ |
| KNN — Distância Manhattan | $\sum_i \lvert p_i-q_i\rvert$ |
| KNN — predição | Classificação = voto majoritário dos K vizinhos; Regressão = média dos K vizinhos |

### Fluxo de decisão resumido

```
O que quero PREVER é um número contínuo ou uma classe?
│
├── NÚMERO CONTÍNUO ──────► Regressão
│      ├── 1 variável explicativa   → Regressão Linear Simples
│      ├── 2+ variáveis explicativas → Regressão Linear Múltipla
│      └── "baseado nos casos mais parecidos" → KNN (regressão)
│
└── CLASSE / CATEGORIA ───► Classificação
       ├── pede probabilidade / coeficientes interpretáveis → Regressão Logística
       └── "baseado nos vizinhos/casos mais parecidos"      → KNN (classificação)

Depois de classificar → pergunta é sobre ACERTOS/ERROS do modelo?
   → Matriz de Confusão (Acurácia, Precisão, Recall, Especificidade, F1)
   → Dados desbalanceados? Desconfie da Acurácia, priorize Recall/Precisão/F1.
```
