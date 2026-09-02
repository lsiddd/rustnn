# rustnn: convolução e atenção em Rust para o CIFAR-100, sem bibliotecas de álgebra linear

## Resumo

Implementamos duas redes para o CIFAR-100 inteiramente em Rust, sem BLAS, sem oneDNN e
sem bibliotecas de aprendizado de máquina. Escrevemos um micro-kernel GEMM de 6×16 em
AVX2/FMA e fundimos as transformações im2col e col2im ao empacotamento dos operandos, de
modo que a matriz de colunas nunca ocupa memória. Sobre esse núcleo construímos uma rede
residual convolucional e um híbrido hierárquico de convolução e atenção, que chamamos
rustvit. Medimos 61 GFLOP/s em uma thread e entre 228 e 255 GFLOP/s em catorze threads,
em um processador de 15 W. Uma rodada de otimização elevou a vazão de treino do rustvit
de 63,9 para 102,6 imagens por segundo, com resultados idênticos bit a bit. O rustvit
atinge 73,59% de acurácia no conjunto de teste. Verificamos o gradiente analítico contra
a derivada direcional em cinquenta checagens. A contribuição principal deste trabalho é
um núcleo de CPU único que serve tanto à convolução quanto à atenção.

## 1. Introdução

Bibliotecas de aprendizado profundo executam a convolução como uma multiplicação de
matrizes, e a transformação im2col é o caminho usual para colocá-la nessa forma. O
obstáculo dessa formulação é o custo de memória. A matriz de colunas é `k²` vezes
maior que a imagem de entrada, e o caminho ingênuo a escreve por inteiro, lê tudo de
volta para reorganizar no layout que o kernel espera e lê uma terceira vez dentro do
próprio kernel. O desejo de eliminar esse tráfego sem abandonar a forma GEMM é a
principal motivação deste trabalho.

A atenção exige do mesmo núcleo um segundo regime de tamanho. Ela é composta de produtos
matriciais pequenos, um por par de imagem e cabeça, e um driver dimensionado para as
matrizes grandes da convolução não os atende bem. Mostramos que um único micro-kernel
serve aos dois regimes, desde que o driver que o chama seja separado em duas versões,
uma paralela para as matrizes grandes e uma sequencial para as pequenas.

As contribuições são as seguintes.

- **C1.** A convolução obtém a forma GEMM sem materializar a matriz im2col, e o
  backward calcula o gradiente da entrada sem materializar a matriz `dcol`. A
  construção está na seção 4.2.
- **C2.** Cada um dos três produtos matriciais da convolução recebe um layout próprio,
  escolhido pelo padrão de acesso e não pela contagem de operações. A seção 4.3
  descreve os três, e a tabela 3 mede o efeito de cada decisão.
- **C3.** A implementação atinge 61 GFLOP/s em uma thread e de 228 a 255 GFLOP/s em
  catorze threads. As medidas estão na tabela 1.
- **C4.** O rustvit combina um tronco convolucional com atenção global em baixa
  resolução e agrega quatro mecanismos: uma memória de alta resolução lida por atenção
  cruzada, o otimizador Muon, autodestilação a partir de saídas auxiliares e uma cabeça
  taxonômica que usa a superclasse do CIFAR-100. A seção 3.2 descreve a arquitetura, e a
  tabela 6 traz a acurácia medida.
- **C5.** Uma rodada de otimização elevou a vazão de treino do rustvit em 60,6%, sem
  alterar nenhum resultado numérico. A atribuição por mudança está na tabela 4, e o
  efeito sobre cada forma de GEMM está na tabela 5.
- **C6.** O gradiente analítico concorda com a derivada direcional em cinquenta
  checagens, que cobrem toda camada isolada, as duas redes completas e o driver de GEMM.
  O procedimento está na seção 6.

O restante deste documento está organizado como segue. A seção 2 descreve o uso do
programa. A seção 3 define as duas arquiteturas. A seção 4 descreve a implementação. A
seção 5 apresenta as medidas de desempenho e de acurácia. A seção 6 descreve a
verificação do backward. A seção 7 reúne as limitações conhecidas, e a seção 8 mapeia
os arquivos do repositório.

## 2. Uso

```bash
# dataset (versão binária, ~161 MB)
mkdir -p data && curl -L -o /tmp/c100.tar.gz \
  https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-binary.tar.gz
tar xzf /tmp/c100.tar.gz -C data

cargo build --release

./target/release/rustnn                      # treina o rustvit, 100 épocas
./target/release/rustnn --arch resnet        # treina a ResNet de referência
./target/release/rustnn --gradcheck          # as cinquenta checagens de gradiente
./target/release/rustnn --bench 15           # imagens por segundo e GFLOP/s
./target/release/rustnn --gemmbench          # GEMM por forma usada pelo modelo
./target/release/rustnn --help               # todas as opções
```

Note que a compilação precisa ser feita em modo `--release`. O arquivo
`.cargo/config.toml` habilita `-C target-cpu=native`, e o `Cargo.toml` usa LTO fat com
`codegen-units = 1`. Sem essas opções, os números da seção 5 não se reproduzem.

## 3. Arquiteturas

### 3.1 A ResNet de referência

A primeira rede é uma ResNet no estilo CIFAR. Ela consiste em uma convolução 3×3 de
entrada, três estágios de blocos residuais com larguras `W`, `2W` e `4W`, uma redução
espacial por média global e uma camada linear de saída. As transições entre estágios
usam stride 2. A opção `--depth n` produz uma rede de `6n+2` camadas, e a configuração
padrão `--depth 2 --width 32` é uma ResNet-14 com 708 mil parâmetros e 105,8 MMAC por
imagem.

Um bloco residual é simplesmente a composição `conv3x3 → BN → ReLU → conv3x3 → BN →
(+ atalho) → ReLU`. Podemos pensar no bloco como uma correção aditiva aplicada à sua
entrada. O parâmetro γ do segundo BatchNorm é inicializado em zero, de modo que a
correção começa nula e o bloco parte da identidade. É fácil ver que, com essa
inicialização, a rede se comporta no primeiro passo como uma rede rasa, o que
estabiliza o início do treino.

### 3.2 rustvit

A segunda rede combina convolução e atenção em uma hierarquia de quatro estágios.
Podemos pensá-la como uma resposta a uma restrição de custo. O custo da atenção cresce
com o quadrado do número de tokens, e o custo da convolução cresce linearmente com a
área. Aplicamos convolução onde a resolução é alta e os tokens seriam muitos, e atenção
global onde os tokens já são poucos. Em 8×8 e em 4×4 a atenção global custa pouco, e por
isso não há janelas nem máscara de deslocamento em nenhum ponto da rede.

| estágio | resolução | operação | tokens | largura |
|---|---|---|---|---|
| stem + C1 | 32×32 | conv 3×3 e bloco residual | — | 32 |
| C2 | 16×16 | bloco residual com stride 2 | — | 96 |
| S3 | 8×8 | cinco blocos de atenção | 64 | 256 |
| S4 | 4×4 | dois blocos de atenção | 16 | 384 |

*Tabela 0. Os quatro estágios do rustvit na configuração padrão, com 6,37 milhões de
parâmetros e 318,6 MMAC por imagem.*

Um bloco de atenção é a composição

```
h = LN₁(x)
x = x + λ₁ ⊙ [ atenção(h) + γ · conv 3×3 depthwise(h) ]
x = x + λ_m ⊙ atenção cruzada(h → memória)          (apenas nos blocos leitores)
x = x + λ₂ ⊙ SwiGLU(LN₂(x))
```

em que cada λ é um vetor de LayerScale inicializado em 1e-4. Recorde que o γ zerado do
BatchNorm da seção 3.1 cumpre a mesma função: o bloco parte da identidade. A convolução
depthwise 3×3 corre no layout de tokens `[n, t, c]`, com o canal como dimensão mais
rápida, e por isso vetoriza sobre nove taps sem passar por im2col.

A atenção de S3 usa duas modificações que a literatura de transformers em conjuntos
pequenos recomenda. A diagonal da matriz de pontuações recebe `-inf`, o que obriga cada
token a se explicar pelos vizinhos, e cada cabeça tem uma temperatura própria treinável,
inicializada em `1/√d_h`. A posição entra por um viés relativo bidimensional, com uma
tabela de `(2·t_h − 1)²` entradas por cabeça. A saída não usa token `[CLS]`, e sim uma
redução por atenção sobre os tokens.

**Memória de alta resolução.** A saída de C2, em 16×16, guarda detalhe que a redução
para 8×8 descarta. Projetamos essa saída por uma convolução 1×1 para 512 canais, o que
produz 256 tokens de memória, e deixamos que os blocos 1 e 3 de S3 a leiam por atenção
cruzada. As consultas são os 64 tokens de S3, e as chaves e os valores vêm da memória,
de modo que o custo do produto é `64 × 256`. O viés relativo da atenção cruzada usa
unidades de meio token, `2·q_y − m_y + (m_h − 1)`, porque
cada token de S3 cobre dois tokens de memória em cada eixo.

**Muon.** As matrizes 2D ocultas são atualizadas pelo Muon, que substitui o gradiente
pela matriz ortogonal mais próxima do momento. A ortogonalização usa três iterações do
polinômio quíntico `X ← aX + b(XXᵀ)X + c(XXᵀ)²X`, com `(a, b, c) = (3,4445; −4,7750;
2,0315)`, e a atualização é `W ← W − lr · 0,2 · √max(linhas, colunas) · NS(momento)`. Os
demais parâmetros usam AdamW. É importante notar por que essa escolha se encaixa em uma
CPU: cada iteração do polinômio é um produto de matriz por matriz nas dimensões da
própria matriz de pesos, que são pequenas, e o mesmo micro-kernel da seção 4.1 as
executa. O custo medido é de 106 ms em um passo de 2495 ms, ou 4,2%.

**Autodestilação por saídas auxiliares.** Duas saídas auxiliares, uma sobre a saída de
C2 e outra sobre a saída de S3, são treinadas contra os logits da saída final por
divergência de Kullback-Leibler com temperatura 3, e as suas representações recebem um
termo de casamento com a representação final. O professor é destacado do grafo, de modo
que nada retorna pela saída final.

**Cabeça taxonômica.** O CIFAR-100 armazena dois rótulos por imagem, a classe fina e a
superclasse de vinte grupos, e a segunda costuma ser descartada. Usamos as duas. Os
logits finais são `z_f[i] = z_c[pai(i)] + r[i]`, ou seja, o logit da superclasse mais uma
correção por classe. O label smoothing também segue a taxonomia: 70% da massa suavizada
vai para os quatro irmãos de superclasse e 30% para as outras noventa e cinco classes.

### 3.3 Receita de treinamento

A ResNet usa SGD com momentum de Nesterov, weight decay somado ao gradiente e não
aplicado aos parâmetros de BatchNorm, aquecimento linear seguido de decaimento por
cosseno, e label smoothing. As imagens passam por recorte aleatório com preenchimento,
espelhamento horizontal e cutout.

O rustvit usa clip de norma global em 1,0, média exponencial dos pesos, e um currículo de
augmentation em quatro fases: as primeiras 10% das épocas sem augmentation, uma rampa até
40%, força total até 90%, e uma fase final com magnitude reduzida a 0,3 e mixup
desligado. As transformações são RandAugment com treze operações, mixup, cutmix e random
erasing. A avaliação por espelhamento horizontal e a avaliação dos pesos crus rodam a
cada dez épocas, e a média exponencial é avaliada a cada época.

## 4. Implementação

### 4.1 O micro-kernel

O núcleo aritmético é um GEMM de 6×16 escrito em AVX2/FMA. Os doze acumuladores `ymm`,
somados a dois registradores para o painel B e um para o broadcast de A, ocupam
exatamente os dezesseis registradores vetoriais que a ISA oferece. O código gerado não
contém nenhum spill para a pilha. Cada passo da dimensão `k` executa doze operações de
FMA contra oito carregamentos, o que coloca o laço sob o limite das unidades de FMA e
não sob o limite da porta de load. A presença de AVX2 é detectada em tempo de
execução, e há um caminho escalar de fallback.

### 4.2 Construção do painel B sem a matriz im2col

O painel B é simplesmente um bloco de `256 × 16` valores, ou 16 KB, que cabe na cache
L1. Podemos pensar nele como a fatia da matriz im2col que o micro-kernel consome de
uma vez. Em vez de escrever a matriz inteira e depois recortá-la, montamos cada painel
diretamente a partir dos pixels da imagem, deixamos que todos os painéis de peso o
consumam, e o descartamos em seguida. No primeiro estágio da rede, a matriz de colunas
ocuparia 1,2 MB por imagem, e a construção fundida remove cerca de 2,4 MB de tráfego
por convolução por imagem.

O col2im do backward desaparece pela mesma razão. O micro-kernel produz um bloco de
6×16 que acumulamos diretamente sobre o gradiente da entrada, e nenhuma matriz `dcol`
chega a existir na memória.

O caso `stride = 1` é o dominante nesta rede e recebe um caminho dedicado na leitura e
na escrita. Nele, cada trecho contíguo se reduz a um `copy_from_slice` ou a um acúmulo
que o compilador vetoriza.

### 4.3 Layout de dados por produto matricial

O forward, o gradiente da entrada `dX` e o gradiente dos pesos `dW` têm a mesma
contagem de operações. Acontece que os três têm padrões de acesso à memória distintos,
e conduzir os três pelo mesmo caminho reduz o desempenho.

O cálculo de `dX` usa `Wᵀ` pré-empacotado e distribui a saída do kernel diretamente
sobre a imagem, conforme descrito na seção 4.2.

O cálculo de `dW = dY · im2col(X)ᵀ` requer o painel de im2col transposto em relação ao
que o forward consome. Em um laço direto, isso se torna uma leitura escalar com uma
verificação de borda por elemento, ao custo aproximado de nove instruções por valor
produzido. Em vez disso, lemos as dezesseis linhas de forma contígua e transpomos o
bloco inteiro dentro dos registradores, com uma transposição 8×8 em AVX2. O custo cai
para cerca de meia instrução por elemento, e a linha 1 da tabela 3 mede o efeito.

O operando `dY` também entra empacotado. Cada painel é reaproveitado `ckk/NR` vezes,
o que corresponde a setenta e duas utilizações na camada mais larga. O empacotamento
tem um segundo efeito, de magnitude maior. Sem ele, o kernel lê seis fluxos separados
por `hw` floats, e quando `hw` é potência de dois esses seis endereços caem no mesmo
conjunto da L1 e se expulsam mutuamente.

### 4.4 Aritmética redundante no laço interno

Decompor uma linha da matriz im2col em `(canal, kh, kw)` requer quatro divisões
inteiras. Note que o laço interno visita a mesma linha uma vez para cada painel B, o
que resulta em centenas de milhares de divisões por imagem em cada camada. Como os
divisores só são conhecidos em tempo de execução, o compilador não os converte em
multiplicações. Construímos a tabela de decomposição uma única vez por camada, com
`ckk` entradas, que cabem na L1. A decomposição do índice linear do pixel de saída em
`(oy, ox)` foi retirada do laço pela mesma razão.

### 4.5 Ativações compartilhadas

Uma mesma ativação costuma servir de entrada a mais de uma camada. No bloco residual,
a convolução do ramo principal e a do atalho recebem o mesmo tensor. Na versão
inicial, cada camada guardava a sua própria cópia da entrada para uso no backward, o
que somava cerca de 380 MB por passo em cópias de entrada e cerca de 150 MB em clones
de saída. As ativações passaram a ser compartilhadas por `Arc`, e o que cada camada
retém para o backward é uma referência. A linha 2 da tabela 3 mede o efeito.

Três fusões acompanham essa mudança. O BatchNorm normaliza dentro do próprio tensor
que recebe, o que é seguro porque o backward da convolução precisa de `X` e não de
`Y`. A ReLU é aplicada na mesma passagem do BatchNorm. A soma residual e a ReLU final
do bloco também ocorrem em uma única passagem.

### 4.6 Paralelismo na convolução

A paralelização é feita por imagem dentro do lote, com os buffers de trabalho
reaproveitados por thread. No cálculo de `dW`, em que todas as imagens contribuem para
o mesmo gradiente, cada worker acumula em um buffer privado e a redução ocorre ao
final, o que dispensa operações atômicas no caminho quente.

Testamos a hipótese de que limitar o número desses buffers privados, por meio de
`with_min_len`, reduziria o tempo, uma vez que economiza alocações e encurta a
redução. A hipótese não se confirma. A tabela 2 mostra que a divisão livre é a mais
rápida das três configurações medidas. Intuitivamente, em uma CPU híbrida o
balanceamento que o work-stealing alcança vale mais do que as alocações economizadas.

### 4.7 O driver de GEMM em duas versões

O rustvit exige do micro-kernel dois regimes de tamanho. As camadas densas produzem
matrizes de dezesseis mil linhas por várias centenas de colunas, e a atenção produz
matrizes de 64×64 por cabeça, uma por par de imagem e cabeça. Um único driver não serve
aos dois. A versão paralela divide as linhas de C em painéis de seis e as distribui pelo
rayon, com o empacotamento dos dois operandos em armazenamento local de thread. A versão
sequencial recebe o espaço de trabalho do chamador e não toca no rayon, porque roda
dentro de uma região já paralela.

O empacotamento nunca reduz o seu buffer. Uma implementação que redimensiona nos dois
sentidos escreve dezenas de megabytes de zeros a cada chamada, e a única condição de
crescimento remove esse custo.

Acima de 2¹⁸ elementos o próprio empacotamento é paralelo. Note que o operando A da
atenção é pequeno e cai abaixo desse limiar, o que preserva a versão sequencial dentro
da região paralela.

### 4.8 Tokens em `[n, t, c]`

O tensor de tokens tem o canal como dimensão mais rápida. Essa escolha tem três
consequências. Em primeiro lugar, as matrizes de peso viram um GEMM único de `[n·t, c]`,
sem bordas e sem laço externo. Em segundo lugar, a convolução depthwise vetoriza sobre o
canal em nove taps. Em terceiro lugar, a LayerNorm percorre um trecho contíguo por token,
com acumulação em `f64` para a média e a variância.

### 4.9 A atenção pelo micro-kernel

Na primeira versão a atenção rodava em laços escalares, e custava um quarto do passo
inteiro. Cada par de imagem e cabeça reúne agora `Q`, `K` e `V` em buffers contíguos, e
os quatro produtos por cabeça, `QKᵀ` e `PV` no forward e `PᵀdO` e `dOVᵀ` no backward,
passam pelo driver sequencial da seção 4.7. A reunião dos operandos custa uma cópia por
cabeça e coloca os quatro produtos dentro do micro-kernel. As linhas 3 e 4 da tabela 4
medem o efeito.

### 4.10 Alocador que recicla os blocos grandes

Cada camada aloca a própria saída, e com lote 256 isso significa dezenas de blocos de 20
a 50 MB por passo. Registramos um alocador global que guarda os blocos acima de 1 MB em
uma tabela de tamanho fixo de 96 posições e os devolve na próxima requisição de mesmo
tamanho e alinhamento. A tabela tem tamanho fixo. Um `Vec` no lugar dela reentraria no
alocador durante um `push` dentro do lock, segurando o próprio mutex. O ganho medido é
de 2 a 3%, e não maior, porque o alocador do sistema já eleva o seu limiar de `mmap` de
forma dinâmica até 32 MB.

### 4.11 Buffers sem inicialização

Os buffers que o GEMM sobrescreve por inteiro são alocados sem serem zerados. A economia
é maior do que o normal aqui, porque o alocador da seção 4.10 devolve blocos sujos, e com
isso o truque de página zerada do `calloc` deixa de se aplicar. Medimos, em rodadas
alternadas, 87 imagens por segundo com os buffers não inicializados contra 68 com
`vec![0.0; n]`, ou 21% do passo inteiro. A função que os aloca é `unsafe`, e o seu
contrato é que o chamador escreva os `n` elementos antes de ler qualquer um deles.

## 5. Resultados

**Protocolo.** Todas as medidas de vazão usam um Core Ultra 7 265U com envelope de 15 W,
dois P-cores a 5,4 GHz, oito E-cores a 4,3 GHz e dois núcleos LP-E a 2,4 GHz, com catorze
threads do rayon. Cada configuração foi executada em rodadas alternadas dentro de uma
mesma sessão, e reportamos o melhor resultado de cada uma. A alternância neutraliza a
redução progressiva de frequência por temperatura, que nesta máquina é a maior fonte de
variação entre medidas. A vazão em imagens por segundo é a medida primária, e os GFLOP/s
são derivados da contagem analítica de operações do forward e do backward. As medidas da
ResNet usam lotes de 128 imagens e `--bench 20`; as do rustvit usam lotes de 256 e
`--bench 15`. As medidas da seção 5.1 foram tomadas antes da adição do rustvit, sobre o
caminho de código convolucional, que o trabalho posterior não alterou.

### 5.1 Vazão da ResNet

Observamos as vazões da tabela 1. A configuração de uma thread foi fixada em um P-core
com `taskset`.

| configuração | imagens/s | GFLOP/s (forward + backward) |
|---|---|---|
| 1 thread, fixada em um P-core | 96 | 61 |
| 14 threads | 360 a 400 | 228 a 255 |

*Tabela 1. Vazão de treino da ResNet-14 w32 sob o protocolo acima. A faixa da segunda
linha reflete a variação entre rodadas alternadas.*

O ganho de escala é de aproximadamente 4× para catorze threads. Note que o
processador tem doze núcleos físicos de três tipos, com frequências máximas que
diferem por um fator de 2,25. Sob carga total as frequências caem para respeitar o
envelope de 15 W, o que explica a distância entre o ganho observado e a contagem de
núcleos.

Observamos as vazões da tabela 2, que testam a hipótese descrita na seção 4.6.

| partição do laço de `dW` | imagens/s |
|---|---|
| divisão livre pelo rayon | 363 |
| `with_min_len(4)` | 346 |
| `with_min_len(2)` | 338 |

*Tabela 2. Efeito de restringir a granularidade da paralelização no cálculo de `dW`. A
divisão livre é a configuração mais rápida.*

Comparando o binário ao fim dessa rodada com a primeira versão funcional do mesmo
código, observamos 312,4 contra 387,8 imagens por segundo, um ganho de 24,1%. Observamos
os ganhos individuais da tabela 3, cada linha medida isoladamente contra o estado
imediatamente anterior a ela.

| mudança | seção | ganho |
|---|---|---|
| transposição vetorizada do painel de `dW` | 4.3 | +15% |
| ativações compartilhadas por `Arc` | 4.5 | +7,5% |
| BatchNorm no lugar com ReLU fundida | 4.5 | +4,7% |

*Tabela 3. Atribuição do ganho da ResNet por mudança. As medidas individuais carregam
uma incerteza aproximada de ±5%, e por isso a soma das linhas não reproduz o ganho
acumulado; a comparação acumulada é a medida de referência.*

### 5.2 Vazão do rustvit

Observamos as vazões da tabela 4. Cada linha foi medida contra o estado imediatamente
anterior a ela, e todas produzem resultados idênticos bit a bit à primeira versão
funcional.

| mudança | seção | forward | backward | otimizador | imagens/s |
|---|---|---|---|---|---|
| primeira versão funcional | — | 1164 ms | 2693 ms | 152 ms | 63,9 |
| empacotamento sem realocar | 4.7 | 1164 | 2693 | 152 | 65,0 |
| atenção pelo micro-kernel | 4.9 | 907 | 1963 | 153 | 84,7 |
| `dW` paralelo sobre o lote | 4.7 | 907 | 1673 | 155 | 93,6 |
| buffers sem zerar nem copiar | 4.11 | 869 | 1651 | 156 | 95,7 |
| alocador que recicla | 4.10 | — | — | — | 97,5 |
| empacotamento paralelo | 4.7 | 802 | 1602 | 155 | 100,0 |
| Muon paralelo por parâmetro | 3.2 | 803 | 1595 | 102 | 101,9 |
| `dW` empacota o lado menor | 4.7 | 798 | 1591 | 106 | 102,6 |

*Tabela 4. Atribuição do ganho do rustvit por mudança, de 63,9 para 102,6 imagens por
segundo, ou 60,6%. As colunas de tempo são por passo de treino com lote 256.*

Observamos as taxas por forma de GEMM da tabela 5, medidas com `--gemmbench` nas formas
exatas que o modelo usa.

| forma (M × N × K) | onde | antes | depois |
|---|---|---|---|
| 16384 × 768 × 256 | projeção QKV, forward | 435,2 | 434,2 |
| 16384 × 256 × 768 | projeção QKV, `dX` | 199,7 | 366,2 |
| 4096 × 384 × 1024 | fusão de tokens S3 para S4 | 216,0 | 344,2 |
| 16384 × 256 × 352 | segunda matriz do SwiGLU | 195,8 | 305,3 |
| 768 × 256 × 16384 | projeção QKV, `dW` | 177,0 | 167,0 |

*Tabela 5. GFLOP/s por forma de GEMM, antes e depois da rodada de otimização da tabela
4. As três formas do meio ganham com o empacotamento paralelo da seção 4.7.*

A linha de `dW` permanece na mesma taxa. O gargalo ali é a redução encadeada de 64
acumuladores privados de 786 KB, e não o empacotamento. Uma redução em árvore está entre
os itens da seção 7.

**Onde o tempo é gasto.** Na ResNet, aproximadamente metade do tempo está dentro do
micro-kernel, que opera no limite descrito na seção 4.1. O restante se divide entre o
tráfego de memória do im2col, que é inerente ao método, e a ociosidade das threads.
Observamos cerca de nove das catorze threads ocupadas em média, o que é consistente com
as aproximadamente quarenta regiões paralelas por passo, cada uma com a sua barreira e a
sua cauda de desbalanceamento. No rustvit, o otimizador responde por 4,2% do passo e a
avaliação de teste respondia por 136 s de uma época de 730 s antes de passar a rodar em
regime reduzido fora das épocas múltiplas de dez.

### 5.3 Acurácia do rustvit

**Protocolo.** A configuração é a padrão do programa, com uma exceção: o treino foi
lançado com `--ema 0.9998` explícito, e a versão atual do código deriva esse decaimento
do orçamento de passos quando a opção é omitida. O comando é

```bash
./target/release/rustnn --epochs 100 --ema 0.9998 --save checkpoint_rustvit.bin
```

O treino foi interrompido na época 73 das 100 programadas, após 12 h 25 min. A acurácia
reportada é a do conjunto de teste de 10 mil imagens, com média sobre o espelhamento
horizontal, e o checkpoint salvo é o da média exponencial dos pesos. A acurácia de treino
que o programa imprime é contada contra o rótulo dominante do lote após o mixup.

Observamos a trajetória da tabela 6.

| época | acurácia de teste | perda de teste |
|---|---|---|
| 10 | 44,18% | 2,2065 |
| 20 | 56,90% | 1,6450 |
| 30 | 63,74% | 1,4216 |
| 40 | 68,02% | 1,2656 |
| 50 | 70,53% | 1,1670 |
| 60 | 73,05% | 1,0679 |
| 70 | **73,59%** | 1,0569 |

*Tabela 6. Acurácia do rustvit no conjunto de teste do CIFAR-100 a cada dez épocas, sob o
protocolo acima. As épocas múltiplas de dez avaliam os pesos crus e a média exponencial;
as demais avaliam apenas a média exponencial.*

O checkpoint da época 70 reproduz 73,59% com

```bash
./target/release/rustnn --resume checkpoint_rustvit.bin --eval
```

O decaimento por cosseno da taxa de aprendizado não chegou ao seu valor final, e a fase
de augmentation reduzida, que começa na época 90, não foi executada. A taxa de ganho por
dez épocas cai de 2,52 pontos no trecho 50 a 60 para 0,54 pontos no trecho 60 a 70.

Note que a receita de regularização é a de um cronograma de 300 épocas aplicada a um de
100. Weight decay de 0,06, RandAugment, mixup, cutmix, random erasing, label smoothing e
stochastic depth se somam à autodestilação, e o efeito da regularização pesada aparece ao
longo de muitas épocas. O currículo da seção 3.3 atenua isso no começo e no fim, e mantém
força total entre as épocas 40 e 90, metade do treino. Uma configuração mais leve, com
RandAugment em 0,15, mixup em 0,4, cutmix em 0,5 e a janela de força total encurtada para
as épocas 50 a 80, é o próximo experimento indicado.

## 6. Verificação do backward

A opção `--gradcheck` compara o gradiente analítico com a derivada direcional
`(L(θ + εv) − L(θ − εv)) / 2ε`, tomada em uma direção aleatória `v`. A comparação é
feita camada por camada e, em seguida, sobre todos os parâmetros de cada rede
simultaneamente.

A escolha da derivada direcional, em lugar da perturbação de pesos individuais, é o
ponto central do procedimento. A derivada direcional agrega o gradiente inteiro em um
único número, e com isso o sinal medido fica bem acima do ruído de arredondamento do
`f32`. Recorde que, ao perturbar uma componente de cada vez, as componentes próximas de
zero produzem erro relativo elevado mesmo na ausência de qualquer erro de implementação,
e o teste perde poder discriminativo.

As cinquenta checagens cobrem quatro grupos. As convoluções, o BatchNorm, a camada
linear, os blocos residuais e as três perdas formam o primeiro. O driver de GEMM contra
uma multiplicação ingênua, nas quatro combinações de transposição e em três formas,
forma o segundo. A LayerNorm, o SwiGLU, a convolução depthwise, a redução por atenção,
a auto-atenção, a atenção cruzada e o bloco completo formam o terceiro. As duas redes
completas e a banda de valores singulares do Newton-Schulz formam o quarto.

Observamos um erro relativo da ordem de 1e-4 por camada, de 2,9e-3 no rustvit completo e
de 1,3e-3 na ResNet completa. As cinquenta checagens passam.

O caso do Newton-Schulz merece uma nota. O polinômio quíntico do Muon não converge para
valores singulares unitários, e sim para uma faixa em torno de 1, o que é deliberado na
formulação original. A checagem correspondente usa uma tolerância de 0,5, e não a de 5e-3
das demais.

## 7. Limitações conhecidas

O caminho vetorizado é específico de x86-64 com AVX2 e FMA. Em outras arquiteturas o
programa usa o caminho escalar, que produz os mesmos resultados a uma fração da
velocidade medida na seção 5.

O treino da seção 5.3 foi interrompido na época 73 de 100, e o cronograma configurado
não foi concluído.

O laço de treino descarta o último lote incompleto de cada época.

O checkpoint armazena os pesos e as estatísticas acumuladas do BatchNorm. Ele não
armazena o momentum do otimizador, a época, o contador de passos nem o estado do
gerador aleatório. A opção `--resume` restaura os pesos e as estatísticas, de modo
que o treino retomado difere do treino contínuo.

A função `scratch_vec` da seção 4.11 é `unsafe`, e a sua correção depende de o chamador
escrever todo o buffer antes de o ler. Reaproveitar os buffers entre passos, em lugar de
alocá-los a cada chamada, removeria essa condição e seria mais rápido, ao custo de
converter as camadas para escrever em um buffer fornecido pelo chamador.

Quatro otimizações do rustvit foram identificadas e não implementadas: fundir a SiLU ao
epílogo do GEMM da primeira matriz do SwiGLU, fundir a soma residual à LayerNorm,
escrever a saída da LayerNorm já no layout empacotado de A, e reduzir em árvore os
acumuladores de `dW`, que é o gargalo da última linha da tabela 5.

O mecanismo de talking heads foi projetado e retirado desta versão. O ganho estimado é de
0,3 ponto, ao custo de 16,8 MB de cache por camada e de uma segunda redução no backward.

## 8. Estrutura do repositório

| arquivo | conteúdo |
|---|---|
| `src/gemm.rs` | micro-kernel AVX2/FMA 6×16, empacotamento dos dois operandos, drivers paralelo e sequencial |
| `src/conv.rs` | convolução com im2col e col2im fundidos ao empacotamento |
| `src/nn.rs` | Conv2d, BatchNorm2d, Linear, pooling, ReLU, softmax e entropia cruzada |
| `src/model.rs` | blocos residuais, a ResNet completa e os checkpoints |
| `src/tok.rs` | tensor de tokens em layout `[n, t, c]` |
| `src/linear.rs` | camada densa sobre o driver de GEMM |
| `src/norm.rs` | LayerNorm, SwiGLU, LayerScale, depthwise 3×3 e redução por atenção |
| `src/attn.rs` | auto-atenção e atenção cruzada, com viés relativo bidimensional |
| `src/vit.rs` | o rustvit completo, as saídas auxiliares e a cabeça taxonômica |
| `src/optim.rs` | Muon com Newton-Schulz, AdamW, clip de norma e média exponencial |
| `src/loss.rs` | alvos com taxonomia, entropia cruzada, destilação e hint |
| `src/pool.rs` | alocador global que recicla os blocos grandes |
| `src/data.rs` | leitor do CIFAR-100 binário, taxonomia e augmentation |
| `src/rng.rs` | gerador xoshiro256++ |
| `src/check.rs` | as cinquenta checagens de gradiente |
| `src/main.rs` | interface de linha de comando, laços de treino, avaliação e benchmarks |
