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

pub fn parse() -> Config {
    let mut c = Config::default();
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let mut lr_set = false;
    let mut ema_set = false;
    macro_rules! next {
        () => {{
            i += 1;
            a.get(i).cloned().unwrap_or_else(|| usage())
        }};
    }
    macro_rules! flag {
        () => {
            next!().parse::<i32>().unwrap_or(1) != 0
        };
    }
    while i < a.len() {
        match a[i].as_str() {
            "--arch" => c.arch = next!(),
            "--data" => c.data = next!(),
            "--epochs" => c.epochs = next!().parse().unwrap(),
            "--batch" => c.batch = next!().parse().unwrap(),
            "--lr" => {
                c.lr = next!().parse().unwrap();
                lr_set = true;
            }
            "--wd" => c.wd = next!().parse().unwrap(),
            "--momentum" => c.momentum = next!().parse().unwrap(),
            "--warmup" => c.warmup = next!().parse().unwrap(),
            "--depth" => c.depth = next!().parse().unwrap(),
            "--width" => c.width = next!().parse().unwrap(),
            "--smooth" => c.smooth = next!().parse().unwrap(),
            "--cutout" => c.cutout = next!().parse().unwrap(),
            "--pad" => c.pad = next!().parse().unwrap(),
            "--seed" => c.seed = next!().parse().unwrap(),
            "--threads" => c.threads = next!().parse().unwrap(),
            "--subset" => c.subset = next!().parse().unwrap(),
            "--save" => c.save = Some(next!()),
            "--resume" => c.resume = Some(next!()),
            "--eval" => c.eval_only = true,
            "--gradcheck" => c.gradcheck = true,
            "--bench" => c.bench = next!().parse().unwrap(),
            "--gemmbench" => c.gemmbench = true,
            "--fingerprint" => c.fingerprint = next!().parse().unwrap(),
            "--dim" => c.dim = next!().parse().unwrap(),
            "--heads" => c.heads = next!().parse().unwrap(),
            "--depth3" => c.depth3 = next!().parse().unwrap(),
            "--dim4" => c.dim4 = next!().parse().unwrap(),
            "--heads4" => c.heads4 = next!().parse().unwrap(),
            "--depth4" => c.depth4 = next!().parse().unwrap(),
            "--cmid" => c.cmid = next!().parse().unwrap(),
            "--droppath" => c.droppath = next!().parse().unwrap(),
            "--muon" => c.muon = flag!(),
            "--ns" => c.ns = next!().parse().unwrap(),
            "--clip" => c.clip = next!().parse().unwrap(),
            "--ema" => {
                c.ema = next!().parse().unwrap();
                ema_set = true;
            }
            "--sib" => c.sib = next!().parse().unwrap(),
            "--coarse" => c.coarse_w = next!().parse().unwrap(),
            "--alpha" => c.alpha = next!().parse().unwrap(),
            "--temp" => c.temp = next!().parse().unwrap(),
            "--beta" => c.beta = next!().parse().unwrap(),
            "--randaug" => c.randaug = next!().parse().unwrap(),
            "--mixup" => c.mixup = next!().parse().unwrap(),
            "--cutmix" => c.cutmix = next!().parse().unwrap(),
            "--erase" => c.erase = next!().parse().unwrap(),
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
