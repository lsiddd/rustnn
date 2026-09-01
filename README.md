# rustnn: uma rede convolucional residual em Rust, sem bibliotecas de álgebra linear

## Resumo

Implementamos uma rede residual convolucional para o CIFAR-100 inteiramente em Rust, sem
BLAS, sem oneDNN e sem bibliotecas de aprendizado de máquina. Escrevemos um micro-kernel
GEMM de 6×16 em AVX2/FMA e fundimos as transformações im2col e col2im ao empacotamento
dos operandos, de modo que a matriz de colunas nunca ocupa memória. Descrevemos o layout
adotado para cada um dos três produtos matriciais da convolução, que têm a mesma
contagem de operações e padrões de acesso distintos. Medimos 61 GFLOP/s em uma thread e
entre 228 e 255 GFLOP/s em catorze threads, em um processador de 15 W. Comparamos a
versão final com a primeira versão funcional e observamos um ganho de 24,1% em imagens
por segundo. Verificamos o gradiente analítico contra a derivada direcional, com erro
relativo de 1,3e-3 na rede completa. A contribuição principal deste trabalho é uma
convolução de CPU que obtém a forma GEMM sem o tráfego de memória que a formulação usual
exige.

## 1. Introdução

Bibliotecas de aprendizado profundo executam a convolução como uma multiplicação de
matrizes, e a transformação im2col é o caminho usual para colocá-la nessa forma. O
obstáculo dessa formulação é o custo de memória. A matriz de colunas é `k²` vezes
maior que a imagem de entrada, e o caminho ingênuo a escreve por inteiro, lê tudo de
volta para reorganizar no layout que o kernel espera e lê uma terceira vez dentro do
próprio kernel. O desejo de eliminar esse tráfego sem abandonar a forma GEMM é a
principal motivação deste trabalho.

As contribuições são as seguintes.

- **C1.** A convolução obtém a forma GEMM sem materializar a matriz im2col, e o
  backward calcula o gradiente da entrada sem materializar a matriz `dcol`. A
  construção está na seção 4.2.
- **C2.** Cada um dos três produtos matriciais da convolução recebe um layout próprio,
  escolhido pelo padrão de acesso e não pela contagem de operações. A seção 4.3
  descreve os três, e a tabela 3 mede o efeito de cada decisão.
- **C3.** A implementação atinge 61 GFLOP/s em uma thread e de 228 a 255 GFLOP/s em
  catorze threads. As medidas estão na tabela 1.
- **C4.** O gradiente analítico concorda com a derivada direcional dentro de 1,3e-3 na
  rede completa. O procedimento está na seção 6.

O restante deste documento está organizado como segue. A seção 2 descreve o uso do
programa. A seção 3 define a arquitetura da rede. A seção 4 descreve a implementação
da convolução. A seção 5 apresenta as medidas de desempenho. A seção 6 descreve a
verificação do backward. A seção 7 reúne as limitações conhecidas, e a seção 8 mapeia
os arquivos do repositório.

## 2. Uso

```bash
# dataset (versão binária, ~161 MB)
mkdir -p data && curl -L -o /tmp/c100.tar.gz \
  https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-binary.tar.gz
tar xzf /tmp/c100.tar.gz -C data

cargo build --release

./target/release/rustnn                      # treino com os parâmetros padrão
./target/release/rustnn --width 16           # rede menor, treino mais rápido
./target/release/rustnn --gradcheck          # verifica o backward numericamente
./target/release/rustnn --bench 20           # imagens por segundo e GFLOP/s
./target/release/rustnn --help               # todas as opções
```

Note que a compilação precisa ser feita em modo `--release`. O arquivo
`.cargo/config.toml` habilita `-C target-cpu=native`, e o `Cargo.toml` usa LTO fat com
`codegen-units = 1`. Sem essas opções, os números da seção 5 não se reproduzem.

## 3. Arquitetura da rede

A rede é uma ResNet no estilo CIFAR. Ela consiste em uma convolução 3×3 de entrada,
três estágios de blocos residuais com larguras `W`, `2W` e `4W`, uma redução espacial
por média global e uma camada linear de saída. As transições entre estágios usam
stride 2. A opção `--depth n` produz uma rede de `6n+2` camadas, e a configuração
padrão `--depth 2 --width 32` é uma ResNet-14 com 708 mil parâmetros.

Um bloco residual é simplesmente a composição `conv3x3 → BN → ReLU → conv3x3 → BN →
(+ atalho) → ReLU`. Podemos pensar no bloco como uma correção aditiva aplicada à sua
entrada. O parâmetro γ do segundo BatchNorm é inicializado em zero, de modo que a
correção começa nula e o bloco parte da identidade. É fácil ver que, com essa
inicialização, a rede se comporta no primeiro passo como uma rede rasa, o que
estabiliza o início do treino.

A receita de treinamento é convencional. Usamos SGD com momentum de Nesterov, weight
decay somado ao gradiente e não aplicado aos parâmetros de BatchNorm, aquecimento
linear da taxa de aprendizado seguido de decaimento por cosseno, e label smoothing. As
imagens de treino passam por recorte aleatório com preenchimento, espelhamento
horizontal e cutout.

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

### 4.6 Paralelismo

A paralelização é feita por imagem dentro do lote, com os buffers de trabalho
reaproveitados por thread. No cálculo de `dW`, em que todas as imagens contribuem para
o mesmo gradiente, cada worker acumula em um buffer privado e a redução ocorre ao
final, o que dispensa operações atômicas no caminho quente.

Testamos a hipótese de que limitar o número desses buffers privados, por meio de
`with_min_len`, reduziria o tempo, uma vez que economiza alocações e encurta a
redução. A hipótese não se confirma. A tabela 2 mostra que a divisão livre é a mais
rápida das três configurações medidas. Intuitivamente, em uma CPU híbrida o
balanceamento que o work-stealing alcança vale mais do que as alocações economizadas.

## 5. Resultados

**Protocolo.** Todas as medidas desta seção usam o modelo padrão da seção 3, lotes de
128 imagens e a opção `--bench 20`, em um Core Ultra 7 265U com envelope de 15 W, dois
P-cores a 5,4 GHz, oito E-cores a 4,3 GHz e dois núcleos LP-E a 2,4 GHz. Cada
configuração foi executada em rodadas alternadas dentro de uma mesma sessão, e
reportamos o melhor resultado de cada uma. A alternância neutraliza a redução
progressiva de frequência por temperatura. A vazão em imagens por segundo é a medida
primária, e os GFLOP/s são derivados da contagem analítica de operações do forward e
do backward.

**Vazão e escalabilidade.** Observamos as vazões da tabela 1. A configuração de uma
thread foi fixada em um P-core com `taskset`.

| configuração | imagens/s | GFLOP/s (forward + backward) |
|---|---|---|
| 1 thread, fixada em um P-core | 96 | 61 |
| 14 threads | 360 a 400 | 228 a 255 |

*Tabela 1. Vazão de treino do modelo padrão sob o protocolo acima. A faixa da segunda
linha reflete a variação entre rodadas alternadas.*

O ganho de escala é de aproximadamente 4× para catorze threads. Note que o
processador tem doze núcleos físicos de três tipos, com frequências máximas que
diferem por um fator de 2,25. Sob carga total as frequências caem para respeitar o
envelope de 15 W, o que explica a distância entre o ganho observado e a contagem de
núcleos.

**Granularidade do paralelismo.** Observamos as vazões da tabela 2, que testam a
hipótese descrita na seção 4.6.

| partição do laço de `dW` | imagens/s |
|---|---|
| divisão livre pelo rayon | 363 |
| `with_min_len(4)` | 346 |
| `with_min_len(2)` | 338 |

*Tabela 2. Efeito de restringir a granularidade da paralelização no cálculo de `dW`. A
divisão livre é a configuração mais rápida.*

**Efeito acumulado das otimizações.** Comparamos o binário atual com a primeira versão
funcional do mesmo código, sob o protocolo acima:

```
BASELINE 312.4 img/s | OTIMIZADO 387.8 img/s | ganho 24.1%
```

**Atribuição por mudança.** Observamos os ganhos individuais da tabela 3. Cada linha
foi medida isoladamente contra o estado imediatamente anterior a ela.

| mudança | seção | ganho |
|---|---|---|
| transposição vetorizada do painel de `dW` | 4.3 | +15% |
| ativações compartilhadas por `Arc` | 4.5 | +7,5% |
| BatchNorm no lugar com ReLU fundida | 4.5 | +4,7% |

*Tabela 3. Atribuição do ganho da seção 5 por mudança. As medidas individuais carregam
uma incerteza aproximada de ±5%, e por isso a soma das linhas não reproduz o ganho
acumulado; a comparação acumulada é a medida de referência.*

**Onde o tempo é gasto.** Aproximadamente metade do tempo de execução está dentro do
micro-kernel, que opera no limite descrito na seção 4.1. O restante se divide entre o
tráfego de memória do im2col, que é inerente ao método, e a ociosidade das threads.
Observamos cerca de nove das catorze threads ocupadas em média, o que é consistente
com as aproximadamente quarenta regiões paralelas por passo, cada uma com a sua
barreira e a sua cauda de desbalanceamento.

## 6. Verificação do backward

A opção `--gradcheck` compara o gradiente analítico com a derivada direcional
`(L(θ + εv) − L(θ − εv)) / 2ε`, tomada em uma direção aleatória `v`. A comparação é
feita camada por camada e, em seguida, sobre todos os parâmetros da rede
simultaneamente.

A escolha da derivada direcional, em lugar da perturbação de pesos individuais, é o
ponto central do procedimento. A derivada direcional agrega o gradiente inteiro em um
único número, e com isso o sinal medido fica bem acima do ruído de arredondamento do
`f32`. Recorde que, ao perturbar uma componente de cada vez, as componentes
próximas de zero produzem erro relativo elevado mesmo na ausência de qualquer
erro de implementação, e o teste perde poder discriminativo.

Observamos um erro relativo da ordem de 1e-4 por camada e de 1,3e-3 na rede completa.
Os dezessete testes passam.

## 7. Limitações conhecidas

O caminho vetorizado é específico de x86-64 com AVX2 e FMA. Em outras arquiteturas o
programa usa o caminho escalar, que produz os mesmos resultados a uma fração da
velocidade medida na seção 5.

O laço de treino descarta o último lote incompleto de cada época.

O checkpoint armazena os pesos e as estatísticas acumuladas do BatchNorm. Ele não
armazena o momentum do otimizador, a época, o contador de passos nem o estado do
gerador aleatório. A opção `--resume` restaura os pesos e as estatísticas, de modo
que o treino retomado difere do treino contínuo.

A camada `Linear` usa laços escalares. Como ela vem depois da redução por média
global, o seu custo não aparece no perfil desta rede. Em um MLP ou em um transformer,
ela seria o componente dominante.

Não há reaproveitamento de buffers entre camadas, e cada ativação é uma alocação nova.
Uma medida preliminar sugere que um pool de buffers renderia por volta de 9%, mas a
comparação alternada que confirmaria esse número não foi concluída.

## 8. Estrutura do repositório

| arquivo | conteúdo |
|---|---|
| `src/gemm.rs` | micro-kernel AVX2/FMA 6×16, empacotamento de A, transposição 8×8 |
| `src/conv.rs` | convolução com im2col e col2im fundidos ao empacotamento |
| `src/nn.rs` | Conv2d, BatchNorm2d, Linear, pooling, ReLU, softmax e entropia cruzada |
| `src/model.rs` | blocos residuais, a rede completa e os checkpoints |
| `src/data.rs` | leitor do CIFAR-100 binário e transformações de treino |
| `src/rng.rs` | gerador xoshiro256++ |
| `src/main.rs` | interface de linha de comando, laço de treino, gradcheck e benchmark |
