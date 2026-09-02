//! Linha de comando: a `Config` que todo o resto le, o texto de uso e o
//! parser. Nada aqui roda por imagem.

pub struct Config {
    pub arch: String,
    pub data: String,
    pub epochs: usize,
    pub batch: usize,
    pub lr: f32,
    pub lr_min: f32,
    pub momentum: f32,
    pub wd: f32,
    pub warmup: usize,
    pub depth: usize,
    pub width: usize,
    pub smooth: f32,
    pub cutout: usize,
    pub pad: usize,
    pub seed: u64,
    pub threads: usize,
    pub save: Option<String>,
    pub resume: Option<String>,
    pub eval_only: bool,
    pub subset: usize,
    pub gradcheck: bool,
    pub bench: usize,
    pub gemmbench: bool,
    pub fingerprint: usize,
    // rustvit
    pub dim: usize,
    pub heads: usize,
    pub depth3: usize,
    pub dim4: usize,
    pub heads4: usize,
    pub depth4: usize,
    pub cmid: usize,
    pub droppath: f32,
    pub muon: bool,
    pub ns: usize,
    pub clip: f32,
    pub ema: f32,
    /// deriva o decay do numero de passos em vez de usar o valor fixo
    pub ema_auto: bool,
    pub sib: f32,
    pub coarse_w: f32,
    pub exit_w: (f32, f32),
    pub alpha: f32,
    pub temp: f32,
    pub beta: f32,
    pub randaug: f32,
    pub ra_n: usize,
    pub mixup: f32,
    pub cutmix: f32,
    pub erase: f32,
    pub curriculum: bool,
    pub tta: bool,
    pub no_mem: bool,
    pub no_exits: bool,
    pub no_taxo: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            arch: "rustvit".into(),
            data: "data/cifar-100-binary".into(),
            epochs: 100,
            batch: 256,
            lr: 6e-4,
            lr_min: 1e-5,
            momentum: 0.9,
            wd: 0.06,
            warmup: 10,
            depth: 2,
            width: 32,
            smooth: 0.1,
            cutout: 8,
            pad: 4,
            seed: 1234,
            threads: 0,
            save: None,
            resume: None,
            eval_only: false,
            subset: 0,
            gradcheck: false,
            bench: 0,
            gemmbench: false,
            fingerprint: 0,
            dim: 256,
            heads: 4,
            depth3: 5,
            dim4: 384,
            heads4: 6,
            depth4: 2,
            cmid: 96,
            droppath: 0.1,
            muon: true,
            ns: 3,
            clip: 1.0,
            ema: 0.9998,
            ema_auto: true,
            sib: 0.7,
            coarse_w: 0.4,
            exit_w: (0.3, 0.6),
            alpha: 1.0,
            temp: 3.0,
            beta: 0.03,
            randaug: 0.3,
            ra_n: 2,
            mixup: 0.8,
            cutmix: 1.0,
            erase: 0.25,
            curriculum: true,
            tta: true,
            no_mem: false,
            no_exits: false,
            no_taxo: false,
        }
    }
}

pub fn usage() -> ! {
    eprintln!(
        "rustnn - redes do zero em Rust para CIFAR-100

USO: rustnn [opcoes]
  --arch <a>         rustvit | resnet                  (rustvit)
  --data <dir>       diretorio com train.bin/test.bin  (data/cifar-100-binary)
  --epochs <n>       numero de epocas                  (100)
  --batch <n>        tamanho do batch                  (256)
  --lr <f>           learning rate de pico             (6e-4 / 0.2 no resnet)
  --wd <f>           weight decay                      (0.06)
  --warmup <n>       epocas de warmup                  (10)
  --seed <n>         semente                           (1234)
  --threads <n>      threads rayon (0 = todas)         (0)
  --subset <n>       usa so N imagens de treino        (0)
  --save <arq>       salva os melhores pesos
  --resume <arq>     carrega pesos antes de treinar
  --eval             so avalia (use com --resume)
  --gradcheck        checagem numerica do backward e sai
  --bench <n>        cronometra n steps de treino e sai
  --fingerprint <n>  treina n steps em uma thread e imprime FNV-1a

 arquitetura (rustvit)
  --dim <n>          largura do estagio S3             (256)
  --heads <n>        cabecas em S3                     (4)
  --depth3 <n>       blocos em S3                      (5)
  --dim4 <n>         largura do estagio S4             (384)
  --heads4 <n>       cabecas em S4                     (6)
  --depth4 <n>       blocos em S4                      (2)
  --cmid <n>         canais do tronco convolucional    (96)
  --droppath <f>     stochastic depth no ultimo bloco  (0.1)

 otimizacao
  --muon <0|1>       Muon nas matrizes 2D              (1)
  --ns <n>           iteracoes de Newton-Schulz        (3)
  --clip <f>         clip de norma global              (1.0)
  --ema <f>          decay da media exponencial, 0 off (0.9998)

 perda
  --smooth <f>       label smoothing                   (0.1)
  --sib <f>          fracao do smoothing nos irmaos    (0.7)
  --coarse <f>       peso da perda de superclasse      (0.4)
  --alpha <f>        peso da autodestilacao            (1.0)
  --temp <f>         temperatura da destilacao         (3.0)
  --beta <f>         peso do termo de hint             (0.03)

 augmentation
  --randaug <f>      magnitude do RandAugment, 0 off   (0.3)
  --mixup <f>        alpha do mixup                    (0.8)
  --cutmix <f>       alpha do cutmix                   (1.0)
  --erase <f>        probabilidade do random erasing   (0.25)
  --curriculum <0|1> rampa e corte da augmentation     (1)
  --tta <0|1>        media com o espelho no teste      (1)

 ablacao
  --no-mem           desliga a memoria de alta resolucao
  --no-exits         desliga as saidas auxiliares
  --no-taxo          smoothing uniforme, sem taxonomia
"
    );
    std::process::exit(1)
}

/// Converte o valor de uma flag, ou morre dizendo qual flag e qual valor.
///
/// Um `.parse().unwrap()` aqui daria `called `Result::unwrap()` on an `Err`
/// value: ParseFloatError`, sem dizer que a flag errada foi `--lr`.
fn value<T: std::str::FromStr>(flag: &str, raw: &str) -> T {
    raw.parse().unwrap_or_else(|_| {
        eprintln!("valor invalido para {flag}: '{raw}'");
        std::process::exit(1)
    })
}

pub fn parse() -> Config {
    let mut c = Config::default();
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let mut lr_set = false;
    let mut ema_set = false;
    macro_rules! next {
        () => {{
            let name = a[i].clone();
            i += 1;
            a.get(i).cloned().unwrap_or_else(|| {
                eprintln!("{name} exige um valor");
                usage()
            })
        }};
    }
    /// Valor tipado da flag corrente.
    macro_rules! val {
        () => {{
            let name = a[i].clone();
            value(&name, &next!())
        }};
    }
    /// `--muon 0` desliga, qualquer outro inteiro liga. Antes um valor
    /// invalido virava silenciosamente `true`; agora e erro, porque
    /// `--tta sim` ligar por acidente esconde o erro de digitacao.
    macro_rules! flag {
        () => {{
            let name = a[i].clone();
            value::<i32>(&name, &next!()) != 0
        }};
    }
    while i < a.len() {
        match a[i].as_str() {
            "--arch" => c.arch = next!(),
            "--data" => c.data = next!(),
            "--epochs" => c.epochs = val!(),
            "--batch" => c.batch = val!(),
            "--lr" => {
                c.lr = val!();
                lr_set = true;
            }
            "--wd" => c.wd = val!(),
            "--momentum" => c.momentum = val!(),
            "--warmup" => c.warmup = val!(),
            "--depth" => c.depth = val!(),
            "--width" => c.width = val!(),
            "--smooth" => c.smooth = val!(),
            "--cutout" => c.cutout = val!(),
            "--pad" => c.pad = val!(),
            "--seed" => c.seed = val!(),
            "--threads" => c.threads = val!(),
            "--subset" => c.subset = val!(),
            "--save" => c.save = Some(next!()),
            "--resume" => c.resume = Some(next!()),
            "--eval" => c.eval_only = true,
            "--gradcheck" => c.gradcheck = true,
            "--bench" => c.bench = val!(),
            "--gemmbench" => c.gemmbench = true,
            "--fingerprint" => c.fingerprint = val!(),
            "--dim" => c.dim = val!(),
            "--heads" => c.heads = val!(),
            "--depth3" => c.depth3 = val!(),
            "--dim4" => c.dim4 = val!(),
            "--heads4" => c.heads4 = val!(),
            "--depth4" => c.depth4 = val!(),
            "--cmid" => c.cmid = val!(),
            "--droppath" => c.droppath = val!(),
            "--muon" => c.muon = flag!(),
            "--ns" => c.ns = val!(),
            "--clip" => c.clip = val!(),
            "--ema" => {
                c.ema = val!();
                ema_set = true;
            }
            "--sib" => c.sib = val!(),
            "--coarse" => c.coarse_w = val!(),
            "--alpha" => c.alpha = val!(),
            "--temp" => c.temp = val!(),
            "--beta" => c.beta = val!(),
            "--randaug" => c.randaug = val!(),
            "--mixup" => c.mixup = val!(),
            "--cutmix" => c.cutmix = val!(),
            "--erase" => c.erase = val!(),
            "--curriculum" => c.curriculum = flag!(),
            "--tta" => c.tta = flag!(),
            "--no-mem" => c.no_mem = true,
            "--no-exits" => c.no_exits = true,
            "--no-taxo" => c.no_taxo = true,
            "-h" | "--help" => usage(),
            other => {
                eprintln!("argumento desconhecido: {other}");
                usage()
            }
        }
        i += 1;
    }
    c.ema_auto = !ema_set;
    if c.arch == "resnet" && !lr_set {
        c.lr = 0.2;
        c.wd = 5e-4;
    }
    c
}

/// Checagens que separam um erro de digitacao de horas de treino perdidas.
///
/// Roda antes de ler o dataset e antes de alocar o modelo, e reporta todos os
/// problemas de uma vez: corrigir um por execucao custa uma execucao por erro.
pub fn validate(c: &Config) {
    let mut bad: Vec<String> = Vec::new();
    // `if $cond {} else` e nao `if !($cond)`: negar uma comparacao de float
    // esconde o caso NaN, que aqui cai no ramo do erro de proposito.
    macro_rules! need {
        ($cond:expr, $($arg:tt)*) => {
            if $cond {
            } else {
                bad.push(format!($($arg)*));
            }
        };
    }
    let unit = |x: f32| (0.0..=1.0).contains(&x);

    need!(
        c.arch == "rustvit" || c.arch == "resnet",
        "--arch '{}': use rustvit ou resnet",
        c.arch
    );
    need!(
        c.batch >= 1,
        "--batch {}: o batch divide o numero de imagens por epoca",
        c.batch
    );
    need!(c.epochs >= 1, "--epochs {}: nao ha o que treinar", c.epochs);
    need!(
        c.warmup <= c.epochs,
        "--warmup {} maior que --epochs {}: o lr nunca sairia da rampa",
        c.warmup,
        c.epochs
    );
    need!(
        c.subset == 0 || c.subset >= c.batch,
        "--subset {} menor que --batch {}: a epoca teria zero passos e as metricas sairiam NaN",
        c.subset,
        c.batch
    );

    // dh = c / h em attn.rs. Sem divisibilidade a divisao trunca e as cabecas
    // deixam de cobrir a largura: os canais que sobram somem sem aviso.
    need!(
        c.heads >= 1,
        "--heads {}: precisa de ao menos uma cabeca",
        c.heads
    );
    need!(
        c.heads4 >= 1,
        "--heads4 {}: precisa de ao menos uma cabeca",
        c.heads4
    );
    need!(
        c.heads == 0 || c.dim.is_multiple_of(c.heads),
        "--dim {} nao e divisivel por --heads {}: sobram {} canais que a atencao descartaria em silencio",
        c.dim,
        c.heads,
        c.dim % c.heads.max(1)
    );
    need!(
        c.heads4 == 0 || c.dim4.is_multiple_of(c.heads4),
        "--dim4 {} nao e divisivel por --heads4 {}: sobram {} canais que a atencao descartaria em silencio",
        c.dim4,
        c.heads4,
        c.dim4 % c.heads4.max(1)
    );
    need!(c.dim >= 1 && c.dim4 >= 1, "--dim e --dim4 precisam ser positivos");
    need!(
        c.cmid >= 1,
        "--cmid {}: o tronco convolucional ficaria vazio",
        c.cmid
    );
    need!(c.width >= 1, "--width {}: precisa ser positivo", c.width);
    need!(
        c.ns >= 1,
        "--ns {}: o Muon precisa de ao menos uma iteracao de Newton-Schulz",
        c.ns
    );

    need!(c.lr > 0.0, "--lr {}: precisa ser positivo", c.lr);
    need!(
        c.lr_min >= 0.0 && c.lr_min <= c.lr,
        "--lr {} e o piso {}: o cosseno espera piso entre zero e o pico",
        c.lr,
        c.lr_min
    );
    need!(
        c.temp > 0.0,
        "--temp {}: a destilacao divide pela temperatura",
        c.temp
    );
    need!(c.wd >= 0.0, "--wd {}: nao pode ser negativo", c.wd);
    need!(c.clip >= 0.0, "--clip {}: nao pode ser negativo", c.clip);
    need!(c.alpha >= 0.0, "--alpha {}: nao pode ser negativo", c.alpha);
    need!(c.beta >= 0.0, "--beta {}: nao pode ser negativo", c.beta);
    need!(
        c.coarse_w >= 0.0,
        "--coarse {}: nao pode ser negativo",
        c.coarse_w
    );
    // mixup e cutmix sao alphas de uma Beta, entao podem passar de um.
    need!(c.mixup >= 0.0, "--mixup {}: nao pode ser negativo", c.mixup);
    need!(c.cutmix >= 0.0, "--cutmix {}: nao pode ser negativo", c.cutmix);

    for (flag, x) in [
        ("--smooth", c.smooth),
        ("--sib", c.sib),
        ("--erase", c.erase),
        ("--droppath", c.droppath),
        ("--randaug", c.randaug),
        ("--ema", c.ema),
        ("--momentum", c.momentum),
    ] {
        need!(unit(x), "{flag} {x}: e uma fracao, precisa estar entre 0 e 1");
    }

    if !bad.is_empty() {
        for m in &bad {
            eprintln!("erro: {m}");
        }
        std::process::exit(1);
    }
}
