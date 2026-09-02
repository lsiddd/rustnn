# ADR-001: complexidade cognitiva nos kernels numéricos

*Estado: aceito. Data: 2026-09-02.*

## Contexto

Dos 27 issues que a primeira análise do SonarQube apontou, 15 continuam de pé
depois que tudo que era refatorável foi refatorado: doze `rust:S3776`
(complexidade cognitiva acima de 15), duas `rust:S2208` (importação com
curinga) e um par de blocos duplicados. Todos vivem no mesmo lugar — o
micro-kernel AVX2 e o empacotamento dos operandos em `gemm.rs`, os spans de
convolução em `conv.rs`, o núcleo da atenção em `attn.rs` e o depthwise 3×3 em
`norm.rs`.

O resto saiu por correção, não por supressão: `main.rs` (cognitiva 80),
`data.rs` (quatro funções, 45/51/32/16), `nn.rs`, `optim.rs`, `check.rs`, o
curinga de escopo de módulo em `model.rs` e a duplicação de `save`/`load` entre
`vit.rs` e `model.rs`. Cada passo manteve o fingerprint do treino bit a bit
idêntico. Este documento cobre só o que sobrou.

## Decisão

Suprimir as quinze, por arquivo, em `sonar-project.properties`. Não são
dívida a pagar depois: são o resultado que o projeto quer.

## Por quê

**A complexidade é do domínio, não do código.** `micro_avx2` tem doze
acumuladores `ymm` vivos ao mesmo tempo, mais dois para o painel de B e um para
o broadcast de A — exatamente os dezesseis registradores vetoriais que a
máquina tem. Uma função extraída no meio desse laço introduz um ponto onde o
alocador de registradores pode derramar, e o derrame de um acumulador custa
mais que todo o ganho de legibilidade. O mesmo vale para o tile 6×16: o número
de ramos é o número de casos de borda do tile, e cada um existe porque o
tamanho da matriz não é múltiplo do tile.

**Extrair muda o código gerado.** `DwConv3::backward` tem complexidade 70
porque os nove taps estão escritos abertos, e é assim que o laço vetoriza sobre
os canais. Enrolá-los num laço de 3×3, que é a refatoração óbvia, troca nove
FMAs independentes por um laço com dependência de acumulador. `gather_span` e
`scatter_span` fundem o im2col e o col2im ao empacotamento; separá-los devolve
a matriz intermediária que a fusão existe para evitar.

**As importações com curinga são a forma idiomática dos intrínsecos.**
`use std::arch::x86_64::*` dentro de uma função `#[target_feature]` traz cerca
de quinze nomes `_mm256_*` usados uma vez cada. Listá-los explicitamente é
ruído sem leitor. O curinga do `model.rs`, que trazia `crate::nn::*` para o
arquivo inteiro, esse sim foi corrigido: era escopo de módulo e escondia de
onde vinha cada tipo.

## Alternativas descartadas

*Marcar como "Won't Fix" na interface do SonarQube.* O estado viveria no banco
do servidor. Um servidor novo perde tudo, e a justificativa não aparece em code
review. Em arquivo versionado, a supressão e o motivo chegam junto com o
diff.

*Baixar a complexidade e medir depois.* Foi o que se fez onde valia, e é a
maior parte do trabalho: 12 dos 27 issues originais foram resolvidos assim. Nos
kernels a medida já existe — `--gemmbench` e `--bench` — e o `regress.sh`
reprova qualquer queda acima de 3%. A questão não é se dá para medir, é que a
refatoração que o Sonar pede é a que a medida rejeita.

## Consequências

O relatório fica em zero, e um issue novo passa a significar código novo fora
do padrão, não ruído acumulado. Em troca, quem mexer num kernel não recebe
aviso do Sonar sobre complexidade — o que o protege é o `--gradcheck`, que
compara o backward com a diferença central, e o `--fingerprint`, que exige o
estado do treino bit a bit idêntico.

Se um desses arquivos passar a ter código que não é kernel, a supressão por
arquivo deixa de ser precisa e vira supressão por linha.
