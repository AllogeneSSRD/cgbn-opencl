
SECTION I.  第一部分。

## Introduction  简介

With the growing amount of digital data and increased risks in data security [19], [27], [44], a range of privacy-preserving techniques have come to the fore. These include privacy-preserving machine learning [1], secure auction [15], and collaborative financial analysis [9]. The primary computations in many of these techniques are big integer multiplication, a compute-intensive operation. Many cryptosystems (e.g., RSA, ElGamal, and Paillier) often mandate the use of at least 2048-bit keys for adequate security [5], which amounts to 4096 32-bit integer multiplication instructions for each big integer mul-tiplication. Given that a single cryptographic operation may involve thousands of big integer multiplications, and privacy-preserving applications often handle many data items, there is a significant opportunity for parallel processing, both within individual multiplications and across multiple operations. As a result, GPUs have emerged as advantageous accelerators for these applications due to their high integer arithmetic capabilities. For instance, Nvidia's H100 [38] and RTX4090 [36] GPUs achieve peak throughputs of 24 and 41.3 Tera Operations Per Second (TOPS), respectively.  
随着数字数据量的增加以及数据安全 [19] 风险的增加， [27] [44] 一系列保护隐私的技术逐渐崭露头角。这些技术包括保护隐私的机器学习 [1] 、安全拍卖 [15] 和协作金融分析 [9] 。这些技术中的主要计算是大整数乘法，这是一种计算密集型操作。许多密码系统（如 RSA、ElGamal 和 Paillier）通常要求至少使用 2048 位密钥以保证足够的安全性 [5] ，这意味着每个大整数多倍滴数需执行 4096 条 32 位整数乘法指令。鉴于单个密码操作可能涉及数千个大整数乘法，且保护隐私的应用通常处理大量数据项，因此在单个乘法内以及跨多操作中，并行处理存在显著机会。因此，GPU 因其高整数运算能力，成为这些应用的有利加速器。例如，英伟达的 H100 [38] 和 RTX4090 [36] GPU 分别达到 24 和 41.3 Tera 运算每秒（TOPS）的峰值吞吐量。

The bit lengths of integers used in cryptography are often too large for single-thread processing but not large enough to benefit from the Number Theoretic Transform (NTT) al-gorithm [43]. Consequently, the schoolbook multiplication algorithm emerges as a practical choice, but the computation should be distributed across several threads for parallel processing. Parallel big integer multiplication involves an extensive design space due to the intricate balance required among on-chip resource usage, communication cost, degree of parallelism, workload balance, and memory access patterns [37]. Cryptographic applications are deployed on a variety of platforms, from embedded devices to high-capacity servers, each requiring specific tradeoffs among these factors. More-over, the bit length varies based on the security level and specific cryptosystems, with many multiplications utilizing special variants, such as squaring and multiply-high operations.  
密码学中使用的整数比特长度通常过大，无法用于单线程处理，但又不足以利用数论变换（NTT）al-gorithm [43] 。因此，教科书乘法算法成为实用选择，但计算应分散在多个线程中以实现并行处理。并行大整数乘法需要广泛的设计空间，因为片上资源使用、通信成本、并行程度、工作负载平衡和内存访问模式 [37] 之间需要复杂的平衡。密码学应用部署于多种平台，从嵌入式设备到大容量服务器，每个平台都需要在这些因素之间做出特定的权衡。此外，比特长度取决于安全级别和具体密码系统，许多乘法采用了特殊变体，如平方运算和乘高运算。

This situation poses a significant challenge. Without parallel processing, optimizing big integer multiplications for various high-level parameters is feasible by adjusting loop bounds to accommodate different bit lengths and altering control flows for specific multiplication variants, such as eliminating unnec-essary calculations in squaring operations. On the other hand, focusing on a single parallel big integer multiplication function allows in-depth manual optimization to balance factors that affect performance. However, the real challenge arises when attempting to address parallelization and variations in high-level parameters simultaneously. Specifically, optimizations effective in one context may not perform well in another. Developing customized implementations for each scenario offers benefits, but manually optimizing each variant requires substantial effort, given the many combinations of bit lengths, special cases, and deploying platforms. It highlights the importance of decoupling architecture-specific optimizations from high-level parameters to enable automatic code optimization tailored to each distinct case.  
这种情况带来了重大挑战。在没有并行处理的情况下，可以通过调整环路界限以适应不同比特长度，并调整控制流以适应特定乘法变体（如消除平方运算中的非必要计算）来优化各种高级参数的大整数乘法。另一方面，专注于单一并行大整数乘法函数，可以进行深入的手动优化，以平衡影响性能的因素。然而，真正的挑战在于试图同时处理并行化和高层参数的变化。具体来说，在一种情境下有效的优化在另一种情境下可能表现不佳。为每种场景开发定制实现带来好处，但由于位长、特殊情况和部署平台的多种组合，手动优化每个变体都需要大量精力。这凸显了将架构特定优化与高层参数解耦的重要性，以实现针对不同情况的自动代码优化。

This paper presents _IMCompiler_, a Compiler-like frame-work designed to generate tailored GPU kernels for various Integer Multiplications used in cryptographic applications. As shown in Figure 1, it features a frontend-IR-backend structure similar to a compiler. As its cores, the Intermediate Repre-sentation (IR) is based on a segmented integer multiplication algorithm. The basic unit is a fixed-size standard integer multiplication function, denoted as UNIT _MUL. The integers are divided into fixed-size segments, and the multiplications of various bit lengths and variants are performed via a sequence of UNIT _MUL calls, followed by accumulation. The UNIT_MUL operation is executed in parallel with NT threads, while the remaining operations are processed serially by the same NT threads. Thus, the parallel issues is fully encapsulated within UNIT _MUL, making the IR serial code. By decoupling architecture-specific optimizations from high-level parameters, the frontend can translate various integer multiplication operations into the IR in a straightforward manner, and the backend only needs to implement UNIT_MUL as a parameterized kernel that can be easily adapted for various GPU architectures.  
本文介绍了 _IMCompiler_，一种类似编译器的框架工作，旨在生成用于密码学应用中各种整数乘法的定制 GPU 内核。如所示 Figure 1 ，它具有类似编译器的前端-红外-后端结构。中间预留（IR）作为核心，基于分段整数乘法算法。基本单元是一个固定大小的标准整数乘法函数，记为 UNIT _MUL。整数被划分为固定大小的段，不同比特长度和变体的乘法通过一系列 UNIT _MUL 调用完成，随后进行累积。UNIT_MUL 操作与 NT 线程并行执行，其余操作由同一 NT 线程串行处理。因此，并行问题完全封装在 UNIT _MUL 中，使红外序列代码成为序列代码。通过将架构特定的优化与高层参数解耦，前端可以以直接的方式将各种整数乘法运算转换为 IR，后端只需将 UNIT_MUL 实现为参数化内核，便于适应各种 GPU 架构。

[![Fig. 1: - Overall workflow of imcompiler](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-1-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-1-source-large.gif)

**Fig. 1:   图1：**

Overall workflow of imcompiler  
非编译器的整体工作流程

Show All  全部显示

_IMCompiler_ also introduces optimizations for the parameterized kernel, bridging gaps between current algorithms and GPU architectures. The key to parallelizing big integer multiplication lies in distributing computation across several threads, requiring a sophisticated balance among on-chip resource usage, communication cost, degree of parallelism, workload balance, and memory access patterns. This paper presents computation diagrams, a schematic tool designed to elucidate the parallel integer multiplication process. It aids in analyzing and designing parallelization strategies, and thus inspires a two-dimensional parallelization strategy: each word is collaboratively computed by multiple threads, and each thread also manages the partial products of several words. The two-dimensional approach significantly improves the balance among the aforementioned factors. Additionally, the segmented integer multiplication algorithm decouples the working set from bit length, providing flexibility to trade extra on-chip resource usage for improvements in other factors. As a result, _IMCompiler_ employs a series of optimizations, including caching inputs in registers rather than in shared memory to eliminate irregular memory access, transposing indices to avoid warp divergence at different stages, and adopting lazy carrying to reduce communication-intensive carry propagation.  
_IMCompiler_ 还引入了参数化内核的优化，弥合了当前算法与 GPU 架构之间的差距。大整数乘法并行化的关键在于将计算分配到多个线程，这需要在片上资源使用、通信成本、并行程度、工作负载平衡和内存访问模式之间取得复杂平衡。本文介绍了计算图，这是一种设计用以阐明并行整数乘法过程的示意工具。它有助于分析和设计并行化策略，从而启发了二维并行化策略：每个字由多个线程协同计算，每个线程还管理多个词的部分乘积。二维方法显著改善了上述因素之间的平衡。此外，分段整数乘法算法将工作集与位长解耦，提供灵活性，以牺牲额外的片上资源使用换取其他因素的改进。因此，_IMCompiler_ 采用了一系列优化，包括将输入缓存在寄存器中而非共享内存中以消除不规则内存访问，调整索引以避免不同阶段的曲速发散，并采用懒惰进位以减少通信密集型进位传播。

This work makes the following major contributions:  
本研究做出了以下主要贡献：

1. We identify the unique attributes of integer multipli-cation used in cryptosystems. It inspires _IMCompiler_, a compiler-like framework that automatically generates tailored GPU kernels for various cases.  
    我们识别了密码系统中使用的整数乘法的唯一属性。它启发了 _IMCompiler_，一个类编译器的框架，能够自动生成针对各种情况定制的 GPU 内核。
    
2. We design an IR to decouple architecture-specific optimizations from high-level parameters, allowing the fron-tend to independently exploit optimization opportunities across various high-level parameters.  
    我们设计了一个 IR，将架构特定的优化与高层参数解耦，使 fron-tend 能够独立利用跨多个高层参数的优化机会。
    
3. We propose a set of parallel integer multiplication optimizations to align algorithms with hardware, guided by computation diagrams, including two-dimensional paral-lelization, tailored caching strategy, index transposing, and lazy carrying.  
    我们提出了一套并行整数乘法优化方法，以计算图为指导，使算法与硬件对齐，包括二维并行列化、定制缓存策略、索引转置和懒惰进位。
    

The paper is structured as follows: Section II introduces the background. Section III delves into the design of the compiler-like framework that decouple architecture-specific optimizations from high-level parameters, while Section IV details the optimizations implemented within the parameterized kernel. Section V evaluates _IMCompiler_. Finally, Section VI reviews related work, before we conclude in section VII.  
本文结构如下： Section II 介绍背景。 Section III 深入探讨了将架构特定优化与高层参数解耦的类编译器框架，并 Section IV 详细介绍了参数化内核内实现的优化。 Section V 评估_了 IMCompiler_。最后，回顾 Section VI 相关工作，然后 section VII 在。

SECTION II.  第二部分。

## Background  背景

### A. Privacy-Preserving Computation  
A. 隐私保护计算

Privacy-preserving computation is becoming increasingly pivotal in today's data-driven computational environment. Its principal objective is to provide secure data processing mechanisms that prevent any breach of privacy associated with individual datasets. Relying on the computational complexity of hard mathematical problems, cryptography-based privacy-preserving computation eliminates the need for extra trust assumptions, such as trusted third parties. Thus, it is particularly crucial in scenarios demanding the utmost security levels.  
隐私保护计算在当今数据驱动的计算环境中变得越来越关键。其主要目标是提供安全的数据处理机制，防止与单个数据集相关的隐私泄露。基于复杂计算的复杂性，基于密码学的隐私保护计算消除了额外信任假设的需求，如可信第三方。因此，在要求最高安全级别的场景中尤为关键。

A range of innovative techniques has emerged, such as Secure Multi-Party Computation [8] (MPC), Homomorphic Encryption [6] (HE), and Zero-Knowledge Proof [31] (ZKP). MPC enables distributed parties to collaboratively process data without disclosing individual inputs; HE permits computations to be performed on encrypted data without necessitating prior decryption; ZKP enables one entity to demonstrate to another that it possesses certain knowledge without revealing the actual information. These advancements have led to the development of several fascinating applications, including privacy-preserving machine learning [46], secure auction systems [15], blockchain [45], and collaborative financial analytics [9].  
一系列创新技术已涌现，如安全多方计算 [8] （MPC）、同态加密 [6] （HE）和零知识证明 [31] （ZKP）。MPC 使分散式各方能够协作处理数据而无需披露个人输入;HE 允许对加密数据进行计算而无需事先解密;ZKP 使一个实体能够向另一个实体展示其拥有某些知识，而无需泄露实际信息。这些进步催生了多项引人入胜的应用，包括保护隐私的机器学习 [46] 、安全拍卖系统 [15] 、区块链 [45] 和协作金融分析 [9] 。

### B. Number Theory Problem Based Cryptographic Systems  
B. 基于数论问题的密码系统

The backbones of these privacy-preserving computation techniques are various cryptosystems. Many widely-used cryp-tosystems are founded on the computational complexity of hard mathematical problems in number theory, such as factoring large composite numbers. To resist attacks, these problems necessitate the numbers to be sufficiently large, exceeding the bit length natively supported by most hardware. Thus, the main computation of these cryptosystems is big integer arithmetic.  
这些保护隐私的计算技术的骨干是各种密码系统。许多广泛使用的 cryp-tosystems 基于数论中难数学问题的计算复杂性，比如对大型复合数的分解。为了抵御攻击，这些问题要求数字足够大，超过大多数硬件原生支持的位长。因此，这些密码系统的主要计算是大整数运算。

This subsection delves into three such cryptosystems, with an emphasis on the computation steps and variable lengths, while the aspects of key generation and underlying mathematical principles are not discussed.  
本节深入探讨了三种此类密码系统，重点关注计算步骤和可变长度，而未讨论密钥生成及其基础数学原理。

#### 1) Rsa[41]  1）RSA [41]

The RSA (Rivest-Shamir-Adleman) cryp- tosystem is the first practical public key encryption algorithm that relies on the difficulty of factoring large integers. One of its notable properties is homomorphic multiplication.  
RSA（Rivest-Shamir-Adleman）cryp-tosystem 是第一个实用的公钥加密算法，依赖于大整数分解的困难。其显著性质之一是同态乘法。

1. – Enc: c=memod n-Dec: m=cdmod n-  
    – 恩克： c=memod n -德克： m=cdmod n -
    
2. HMul: m1⋅m2=c1⋅c2mod n  HMul： m1⋅m2=c1⋅c2mod n
    

#### 2) Elgamal[12]  2）埃尔加马尔 [12]

ElGamal encryption is a public-key cryptosystem based on Diffie-Hellman (DH) key exchange, whose security depends upon the discrete logarithm problem.  
ElGamal 加密是一种基于 Diffie-Hellman（DH）密钥交换的公钥密码系统，其安全性依赖于离散对数问题。

1. **- DH Key** Exch: (gamod p)b=(gb mod p)b mod p  
    **- DH 密钥**扩展： (gamod p)b=(gb mod p)b mod p
    
2. – Enc: (c=gkmod p, d=m⋅ykmod p)  —— 恩克： (c=gkmod p, d=m⋅ykmod p)
    
3. – Dec: m=d/(cxmod p)mod p  ——十二月： m=d/(cxmod p)mod p
    
4. **- HMul**: m1⋅m2=(c1⋅c2, d1⋅d2)mod p  
    **- HMul**： m1⋅m2=(c1⋅c2, d1⋅d2)mod p
    

#### 3) Paillier[39]  3）佩利耶 [39]

The Paillier cryptosystem is a public key encryption scheme based on the computational difficulty of the Decisional Composite Residuosity (DCR) problem. It supports homomorphic addition and scalar multiplication.  
Paillier 密码系统是一种基于决策复合剩余（DCR）问题计算难度的公钥加密方案。它支持同态加法和标量乘法。

1. Enc:c=(m⋅n+1)⋅rnmod n2
    
2. – Dec:μ⋅(cλmod n2−1)/nmod n-HAdd: m1+m2mod n=c1⋅c2mod n2-HScale: k⋅m1mod n=ck1mod n2  
    —— Dec:μ⋅(cλmod n2−1)/nmod n 哈德： m1+m2mod n=c1⋅c2mod n2 ——赫斯凯尔： k⋅m1mod n=ck1mod n2
    

The primary computations in these cryptosystems are mod-ular multiplication and modular exponentiation, while the latter operation can be decomposed into a series of modular multiplication operations. The modulus operation necessitates division, which is considerably costly, especially when dealing with big integers. To this end, Montgomery proposes an efficient algorithm for modular multiplication [32]. It replaces division by n with division by r=2m, where the latter can be implemented simply by discarding the least significant m bits. The additional costs encompass a multiply-low operation and a multiply-high operation, both introduced due to multiplication within the residue class modulo space. Consequently, Mont-gomery multiplication converts modular multiplication into a sequence of integer multiplication operations.  
这些密码系统的主要计算是模乘法和模幂运算，后者可以分解为一系列模乘法运算。模数运算需要除法，尤其在处理大整数时代价较高。为此，Montgomery 提出了一种高效的模乘法算法 [32] 。它用除法替换 n 为 r=2m 除法，后者可以通过丢弃最低有效 m 位实现。额外成本包括乘低运算和高乘法运算，这两者都是由于模余数类空间内的乘法引入的。因此，蒙哥马利乘法将模乘法转换为整数乘法序列。

### C. Motivation  C. 动机

Big integer multiplication has a time complexity of O(N2) and a memory complexity of O(N). The computational intensity is O(N), making GPUs an advantageous platform for acceleration. However, designing an efficient parallel integer multiplication algorithm poses significant challenges due to inherent data dependencies, workload imbalances, and irregu-lar memory access patterns, as depicted in Section IV-Band assessed in Section V-E.  
大整数乘法的时间复杂度为 O(N2) ，内存复杂度为 O(N) 。计算强度为 O(N) ，使 GPU 成为加速的有利平台。然而，由于数据依赖性、工作负载不平衡以及无序-拉尔内存访问模式，设计高效的并行整数乘法算法存在重大挑战 Section V-E ，如 Section IV -带评估中所示。

Cryptographic applications vary widely in security require-ments and execution platforms. Figure 2 depicts the correlation between security levels and key sizes across various cryp-tosystems [28], including elliptic curve cryptography (ECC), RSA, DH key exchange, ElGamal, Paillier [29], NTRUEn-crypt, FrodoKEM [2], and Saber [10]. These cryptosystems range from established (e.g., RSA) to emerging (e.g., post-quantum) technologies. Notably, security levels in some cryp-tosystems are influenced not only by key sizes but also by additional factors, such as the polynomial degree in lattice-based cryptography and complex elliptic curve operations in ECC. The minimum security level considered secure today is 112-bit [28]; however, 80-bit may suffice for certain privacy-preserving computations, and 128-bit is recommended for long-term security. In this context, most cryptosystems handle big integers that exceed 2048 bits, extending up to _64K_.  
密码学应用在安全需求和执行平台上差异很大。 Figure 2 描绘了各种 cryp-to 系统 [28] 中安全级别与密钥大小之间的相关性，包括椭圆曲线密码学（ECC）、RSA、DH 密钥交换、ElGamal、Paillier [29] 、NTRUEn-crypt、 [2] FrodoKEM 和 Saber [10] 。这些密码系统涵盖了从成熟（如 RSA）到新兴（如后量子）技术。值得注意的是，某些 cryp-tosystems 的安全级别不仅受密钥大小影响，还受其他因素影响，如基于格点的密码学中的多项式次数和 ECC 中的复杂椭圆曲线运算。目前被认为安全的最低安全级别为 112 位 [28] ;然而，对于某些保护隐私的计算，80 位或许足够，128 位则建议用于长期安全。在这种情况下，大多数密码系统处理的大整数超过 2048 位，最高可达 _64K_。

[![Fig. 2: - Correlation between security level and key size.](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-2-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-2-source-large.gif)

**Fig. 2:   图2：**

Correlation between security level and key size.  
安全性等级与密钥大小之间的相关性。

Show All  全部显示

[![Fig. 3: - Frequency of different variants.](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-3-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-3-source-large.gif)

**Fig. 3:   图3：**

Frequency of different variants.  
不同变体的频率。

Show All  全部显示

Moreover, a single application can involve multiple variants of big integer multiplications. Figure 3 illustrates the frequency distribution of each variant within the cryptosystems discussed. Multiplications that fall into both the multiply-constant and multiply-low/high categories are categorized under the latter, reflecting their higher optimization potential.  
此外，单个应用可能涉及多种大整数乘法变体。 Figure 3 展示了上述密码系统中每种变体的频率分布。既属于乘法常数类和乘法低/高类别的乘法也归入后者，反映了其更高的优化潜力。

Optimizing big integer multiplication for a single setting and device is already complex. Real-world cryptographic applications require handling diverse integer sizes and multi-plication variants across different devices. While customizing implementations for each scenario offers benefits, manual optimization is labor-intensive. This underscores the necessity of decoupling architecture-specific optimizations from high-level parameters, thereby enabling automatic code optimization for each unique scenario.  
针对单一设置和设备优化大整数乘法本身就很复杂。现实世界的密码应用需要处理不同设备的不同整数大小和多重变体。虽然为每种场景定制实现有好处，但手动优化劳动强度较大。这凸显了将架构特定优化与高层参数解耦的必要性，从而实现针对每种独特场景的自动代码优化。

SECTION III.  第三部分。

## Compiler-Like Frontend-Ir-Backend Design  
类编译器的前端-后端设计

To facilitate automatic code generation, _IMCompiler_ employs a compiler-like design, as shown in Figure 4. It features a frontend-IR-backend architecture to decouple architecture-specific optimizations from high-level parameters. The input is the application's source code, where each big integer multipli-cation is defined as a function call, with arguments specifying bit length and variant. This design choice is consistent with the structure of most cryptographic applications, where big integer multiplications are typically implemented through function calls. Furthermore, selecting the optimal variant for each operation is usually straightforward, as most cryptosystems (e.g., [10], [12], [41]) specify this in their documentation.  
为了促进自动代码生成，_IMCompiler_ 采用了类似编译器的设计，如 Figure 4 所示。它采用前端-红外-后端架构，将架构特定的优化与高层参数解耦。输入是应用程序的源代码，每个大整数乘法定义为函数调用，参数指定位长和变体。这种设计选择与大多数密码学应用的结构一致，大整数乘法通常通过函数调用实现。此外，选择每个操作的最优变体通常很简单，因为大多数密码系统（如 [10] ， ， [12] ） [41] 在其文档中都有明确规定。

For each function call, the frontend extracts the high-level parameters, and then generates optimized IR code accordingly. The IR code is subsequently translated by the backend into a parameterized kernel, designed for easy adaptation across various GPU architectures. These optimized kernels are then integrated into the source code, replacing the original big integer multiplication functions. Finally, the standard GPU compiler is invoked to compile the code.  
对于每个函数调用，前端提取高层参数，然后相应生成优化后的 IR 代码。IR 代码随后由后端翻译成参数化内核，便于在不同 GPU 架构间适配。这些优化后的内核随后被集成到源代码中，取代了原有的大整数乘法函数。最后，调用标准 GPU 编译器来编译代码。

[![Fig. 4: - A compiler-like frontend-ir-backend design](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-4-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-4-source-large.gif)

**Fig. 4:   图4：**

A compiler-like frontend-ir-backend design  
类似编译器的前端或后端设计

Show All  全部显示

### A. Intermediate Representation  
A. 中间代表

The IR is designed as a segmented integer multiplication algorithm. As illustrated in Figure 5, each integer is partitioned into Ns-bit segments, where Ns is a hyperparameter determined by the GPU model and independent of bit length and variant. The primary instruction of the IR is UNIT _MUL, a standard Ns-bit parallel integer multiplication function, where two distinct Ns-bit integers are multiplied to produce a 2 Ns-bit integer. Multiplication of varying bit lengths and variants is conducted segment by segment using UNIT_MUL, followed by serial accumulation. _IMCompiler_ profiles various Ns-bit UNIT _MUL configurations to identify the optimal setup, which is conducted once per GPU model and the results are applicable to all future computations on that GPU.  
IR 被设计为分段整数乘法算法。如图所示 Figure 5 ，每个整数被划分为 Ns -bit 段，其中 Ns 是由 GPU 模型决定的超参数，且与位长和变体无关。IR 的主要指令是 UNIT _MUL，一种标准 Ns 的-位并行整数乘法函数，将两个不同的 Ns -位整数相乘得到 2 Ns 位整数。不同比特长度及变体的乘法采用 UNIT_MUL 逐段进行，随后进行串行累积。_IMCompiler_ 对各种 Ns -bit UNIT _MUL 配置进行分析，以确定最优配置，每个 GPU 模型进行一次，结果适用于该 GPU 的所有未来计算。

The detailed procedure is shown in Algorithm 1. Each N-bit integer multiplication is calculated by a thread group consisting of NT threads. This thread group iterates through segment pairs serially with dual loops, with the outer loop traversing the segments of the product, and the inner loop accumulating the partial results calculated by UNIT _MUL. This iteration order avoids costly read-modify-write operations and carry propagation regarding the segments of the product. Within UNIT_MUL, a segment pair (i.e., two Ns-bit integers) is multiplied in parallel by the NT threads. As such, for mN-bit integer multiplications, m×NT threads are initiated, which is a value independent of the bit length N.  
详细过程见 Algorithm 1 。每个 N -位整数乘法由由多个 NT 线程组成的线程群计算。该线程组通过串行迭代段对，带着对偶循环，外环遍历积段，内环累积由 UNIT _MUL 计算的部分结果。这种迭代顺序避免了昂贵的读-修改-写操作，并传递关于乘积段的传播。在 UNIT_MUL 内，线 NT 程并行乘以段对（即两个 Ns -位整数）。因此，对于 mN -bit 整数乘法， m×NT 会启动线程，这个值与位长 N 无关。

[![Fig. 5: - Illustration of segmented integer multiplication](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-5-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-5-source-large.gif)

**Fig. 5:   图5：**

Illustration of segmented integer multiplication  
分段整数乘法的示意图

Show All  全部显示

### Algorithm 1 Segmented Integer Multiplication Algorithm  
算法1 分段整数乘法算法

cooperatively executed by the NT threads of gidth group  
由 gidth 组 NT 的线程协同执行

1:

divide _A, B, C_ into Ns-bit segments _segA, segB, segC_  
将 _A、B、C_ 划分为 Ns -bit 段 _segA、segB、segC_

2:

_carry =_ 0  _进位 =_ 0

3:

_arrHigh_, arrLow=[0]Ns,[0]Ns▹ array of registers  
_arrHigh_， arrLow=[0]Ns,[0]Ns▹ 寄存器数组

4:

**for** n=0 **to** NNs **do**  
**为** n=0 NNs **去做**

5:

_carry, arrHigh_, _arrLow =_ 0, [0, …, _carry], arrHigh_  
_进位，arrHigh，arrLow_ _=_ 0， [0， ...， _进位]，arrHigh_

6:

**for** i=0 to n **do**  
**为** i=0 n **去做**

7:

A′,B′=segA[gid][i],segB[gid][n−i]

8:

arrHigh′,arrLow′=UNIT−MUL(A′,B′)▹ Algo 2

9:

_carry_ ∥ _arrHigh_ ∥arrLow+=arrHigh′∥arrLow′  
_高高_ ∥ ∥arrLow+=arrHigh′∥arrLow′

10:

segC[gid][n]=arrLow

11:

**for** n=NNs **to** 2NNs **do**  
**为** n=NNs 2NNs **去做**

12:

_carry, arrHigh, arrLow =_ 0, [0, …, _carry], arrHigh_  
_进位，arrHigh，arrLow =_ 0， [0， ...， _进位]，arrHigh_

13:

**for** i=n−NNs+1 **to** NNs **do**  
**为** i=n−NNs+1 NNs **去做**

14:

A′,B′=segA[gid][i],SegB[gid][n−i]

15:

arrHigh′,arrLow′=UNIT−MUL(A′,B′)▹ Algo 2

16:

_carry_ ∥ _arrHigh_ ∥arrLow+=arrHigh′∥arrLow′  
_高高_ ∥ ∥arrLow+=arrHigh′∥arrLow′

17:

segC[gid][n]=arrLow

The segmented integer multiplication algorithm is similar to the well-known tiled matrix multiplication approach, where a matrix/vector is divided into tiles/segments. Each tile/segment is multiplied independently before aggregating the results. Although they employ similar methodologies, their underlying design philosophies differ significantly. Tiled matrix multipli-cation primarily aims to enhance data locality by tailoring tile sizes to fit on-chip memory, allowing on-chip processing regardless of the matrix size.  
分段整数乘法算法类似于著名的铺砌矩阵乘法方法，即将矩阵/向量分割成多个图块/段。每个切片/切片/切片在汇总结果前先独立相乘。尽管它们采用了相似的方法，但其底层设计理念有显著差异。切片矩阵乘法主要旨在通过定制切片尺寸以适应片上内存来增强数据局部性，从而实现无论矩阵大小如何都能进行片上处理。

On the other hand, the segmented integer multiplication algorithm not only allows strategic selection of Ns for performance optimization, but also encapsulates all implementation details of parallel integer multiplication within UNIT _MUL. The remaining instructions serve merely as supporting code. This decoupling of parallelization specifics from high-level parameters enables straightforward conversion of integer mul-tiplication across different bit lengths and variants into a series of UNIT _MUL calls. It allows the frontend to independently manage diverse high-level parameters, without engaging with the complexities of parallelization handled by the backend.  
另一方面，分段整数乘法算法不仅允许战略性地选择 Ns 以优化性能，还将并行整数乘法的所有实现细节封装在 UNIT _MUL 中。其余指令仅作为支持代码。这种将并行化细节与高层参数解耦，使得不同比特长度和变体的整数多倍叠换能够直接转换为一系列 UNIT 的_MUL 调用。它允许前端独立管理多样的高级参数，而无需面对后端处理的并行化复杂性。

#### Comparison With Multi-Level IR  
与多级红外的比较

The MLIR project [25] offers a novel, reusable, and extensible compiler infrastructure, designed as a hybrid IR to support diverse requirements. _IM-Compiler_ can be integrated into MLIR as a dialect, with fron-tend and backend optimizations implemented as MLIR passes. The standardized framework of MLIR promotes user-friendly and broad industrial adoption of _IMCompiler_. However, implementing _IMCompiler_ as a standalone framework meets all design objectives effectively: using IR to decouple high-level parameters from architecture-specific details, enabling automated optimization of parallel big integer multiplication.  
MLIR 项目 [25] 提供了一种新颖、可复用且可扩展的编译器基础设施，设计为混合型 IR，以支持多样化需求。_IM-编译器_可以作为方言集成到 MLIR 中，随着 MLIR 的推进，前端和后端优化也得以实现。MLIR 的标准化框架促进_了 IMCompiler_ 在业界的用户友好和广泛应用。然而，作为独立框架实现 _IMCompiler_ 能够有效满足所有设计目标：利用 IR 将高层参数与架构特定细节解耦，实现并行大整数乘法的自动优化。

### B. Frontend  B. 前端

The frontend takes the bit length N and the variant as inputs, then generates optimized IR composed of UNIT_MUL sequences. While the bit length N is managed through loop boundaries, the frontend generates tailored IR code for the following variants of integer multiplication operations, each with its distinct optimization strategy. Adding new variants is straightforward, as the frontend-IR-backend architecture enables the frontend to manage the variants independently.  
前端以位长 N 和变体为输入，生成由 UNIT_MUL 序列组成的优化红外线。位长 N 通过环路边界管理，前端则生成针对以下整数乘法变体的定制红外线代码，每种变体都有其独特的优化策略。添加新变体非常简单，因为前端-红外-后端架构允许前端独立管理变体。

#### Square  方形

Exponentiation is a critical operation in cryptogra- phy, predominantly performed using the square-and-multiply algorithm [16]. This method executes log2N squaring operations for each exponentiation, where N is the bit length of the exponent. Squaring involves multiplying two identi-cal numbers. Since segA[⋯] and segB[⋯] are the same, segA[i]×segB[j] equals segA[j]×segB[i]. By eliminating re-dundant calculations, the computational demands of squaring operations can be reduced by up to 50%.  
指数化是密码学中的关键运算，主要使用平方乘法算法 [16] 完成。该方法对 log2N 每个指数执行平方运算，其中 N 是指数的位长。平方涉及两个等方数相乘。由于 segA[⋯] 和 segB[⋯] 相同， segA[i]×segB[j] 等于 segA[j]×segB[i] 。通过消除重复重复计算，平方运算的计算需求可减少多达 50%。

#### Multiply-Low&multiply-High  
乘低与乘高

Each Montgomery mul- tiplication consists of a multiply-low and a multiply-high operation. The multiply-low operation uses only the lower N bits of the 2N-bit result, allowing Algorithm 1 to bypass Lines 11–17, thereby reducing computational load by approximately 50%. Conversely, the multiply-high operation, which uses only the upper N bits, is more complex due to the need to account for carries from the lower N bits. However, it does not require multiplication of all segment pairs related to the lower N bits. Instead, these segments can be computed in reverse order. Subsequent segments are computed only if the carry produced by segC[n] could potentially cause an overflow in segC[NNs]. The likelihood of performing an additional UNIT _MUL decreases significantly after n reaching NNs−2, which decays by a factor of 2−Ns. Thus, multiply-high also allows up to about 50% computation reduction.  
每个蒙哥马利多重处理包括一个低乘法和高乘运算。乘低运算只使用 -bit 结果的低 N 位，从而绕 Algorithm 1 过第 11–17 行，从而约 50% 的计算负载降低。相反，乘高运算仅使用上 N 位，由于需要考虑低 N 位进位，因此更为复杂。然而，它不需要乘法所有与下 N 位相关的段对。相反，这些段可以逆序计算。只有当 的 segC[n] 进位可能导致 在 中 segC[NNs] 溢出时，才计算后续段。 2N n 达到 NNs−2 后，执行额外单位 _MUL 的概率显著降低，且衰减为 2−Ns 的因子。因此，乘高还允许计算减少约 50%。

#### Multiply-Constant  乘法常数

Many integers in cryptosystems are constant values, such as n and u in the Paillier cryptosystem and r in Montgomery multiplication. Storing these constant values in constant memory or directly embedding them into the kernel code reduces 25% device memory accesses and 50% on-chip resource usage for caching input.  
密码系统中的许多整数是常数值，例如 n 在 Paillier 密码系统 r 中和 Montgomery u 乘法中。将这些常数值存储在常量内存中或直接嵌入内核代码中，可减少 25%的设备内存访问和 50%的片上资源使用，用于缓存输入。

The above optimizations can be implemented in various ways, such as compiler passes. Thanks to the frontend-IR-backend architecture, the frontend merely needs to generate a series of calls to UNIT_MUL based on high-level parameters. Thus, the C++ templates mechanism suffices (and actually offers more flexibility) for all aforementioned optimizations. Notably, _IMCompiler_ handles the special variants at the gran-ularity of UNIT_MUL. For instance, in squaring, _IMCompiler_ eliminates the redundant computation of the same UNIT _MUL instead of optimizing at the granularity of 32-bit multiply instructions. This approach strikes an optimal balance between optimization efforts and kernel performance.  
上述优化可以通过多种方式实现，比如编译器通过。得益于前端-红外-后端架构，前端只需根据高级参数生成一系列调用 UNIT_MUL。因此，C++模板机制对于上述所有优化都足够（实际上提供了更多灵活性）。值得注意的是，_IMCompiler_ 在极度的特殊变体处理 UNIT_MUL。例如，在平方处理中，_IMCompiler_ 消除了同一 UNIT_MUL 的冗余计算，而不是在 32 位乘法指令的粒度下优化。这种方法在优化努力和内核性能之间取得了最佳平衡。

SECTION IV.  第四部分。

## Optimize Backend With Computation Diagram  
用计算图优化后端

Parallelizing integer multiplication on GPUs faces significant challenges due to inherent data dependencies, workload imbalances, and irregular memory access patterns. While optimizing a single factor might seem straightforward, it can exacerbate other issues. Analyzing the complex interplay of these factors through formulaic modeling is unintuitive. To this end, this section introduces the computation diagram to visually depict these interactions.  
在 GPU 上并行化整数乘法面临重大挑战，原因在于固有的数据依赖性、工作负载不平衡和内存访问模式不规则。虽然优化单一因素看似简单，但可能加剧其他问题。通过公式化建模分析这些因素的复杂相互作用并不直观。为此，本节介绍计算图以直观地展示这些交互。

### A. Modeling With Computation Diagram  
A. 使用计算图进行建模

[![Fig. 6: - Example of a computation diagram](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-6-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-6-source-large.gif)

**Fig. 6:   图6：**

Example of a computation diagram  
计算图示例

Show All  全部显示

As shown in Figure 6, each element represents a wide-multiply instruction, and the blue lines denote addition operations. The addition operations along diagonal lines represent the propagation of the upper 32 bits and their associated carry. The enclosed yellow shapes outline the workload per thread.  
如图所示 Figure 6 ，每个元素代表宽乘法指令，蓝线表示加法运算。沿对角线的加法运算表示上方 32 位及其对应进位的传播。包围的黄色图形勾勒出每个线程的工作负荷。

The intersections of yellow and blue lines represent data dependencies among threads, necessitating communication. Communication cost is influenced not only by the number of data exchanges but also their patterns. Regular patterns only require a single warp shuffle per exchange (e.g., odd/even thread pairs swapping data), whereas irregular patterns often necessitate multiple warp shuffles, and some must be conducted via shared memory (e.g., the communication pattern in Figure 6), thus increasing the cost per exchange. Moreover, the communication pattern is strongly correlated with on-chip resource usage. While allocating more resources to cache intermediate results typically reduces communication and synchronization costs, excessive usage can reduce parallelism.  
黄线和蓝线的交点代表线程之间的数据依赖，因此需要通信。通信成本不仅受数据交换次数影响，还受其模式影响。常规模式每次交换只需一次 warp shuff（例如奇偶线程交换数据），而不规则模式通常需要多次 warp shuff，有些必须通过共享内存进行（例如，通信模式）， Figure 6 从而增加了每次交换的成本。此外，通信模式与片上资源使用高度相关。虽然分配更多资源缓存中间结果通常能降低通信和同步成本，但过度使用会降低并行性。

The area of each shape quantifies the workload for each thread. Assigning more workload to a thread reduces paral-lelism, while reducing it increases launch overhead. Notably, execution time does not scale linearly with workload. Over-burdening a thread with instructions introduces a dilemma: although loop unrolling can boost GPU performance [33], it risks exceeding the instruction-cache capacity. This is particularly problematic for kernels executing integer arithmetic, as runtime address calculations in loops lead to contention in integer ALUs (Arithmetic Logic Units).  
每个图形的面积量化了每个线程的工作负载。为线程分配更多工作负载减少了并列拉利主义，同时减少并列运行，增加了启动开销。值得注意的是，执行时间不会随工作负载线性增长。给线程过载指令会带来一个两难问题：虽然循环展开可以提升 GPU 性能 [33] ，但有超出指令缓存容量的风险。这对执行整数运算的内核尤其成问题，因为循环中的运行时地址计算会导致整数算术逻辑单元（ALU）中的争用。

[![Fig. 7: - Parallelization strategy analysis with computation diagrams](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-7-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-7-source-large.gif)

**Fig. 7:   图7：**

Parallelization strategy analysis with computation diagrams  
并行化策略分析与计算图

Show All  全部显示

Variations in shapes, including geometry and orientation, indicate divergence. Warp divergence incurs redundant computations and suboptimal resource use, as resources are allocated according to the most demanding path.  
形状的变化，包括几何形状和方向，表明存在发散。曲速发散会导致冗余计算和资源使用不优，因为资源是按照最严格的路径分配的。

Drawing such a compute diagram at the algorithm design stage offers a clear visual representation of the interactions among these performance-impacting factors. Notably, while initially intended for big integer multiplication, compute di-agrams can also facilitate parallel algorithm design in other domains, potentially inspiring other innovative designs.  
在算法设计阶段绘制此类计算图，可以清晰地直观地展示这些影响性能的因素之间的相互作用。值得注意的是，虽然最初用于大整数乘法，计算二字母星也能促进其他领域的并行算法设计，可能激发更多创新设计。

### B. Computation Diagram of Existing Implementations  
B. 现有实现的计算图

This section analyzes several traditional parallelization strategies using computation diagrams.  
本节分析了几种传统并行化策略，使用计算图。

#### ai×bi

As illustrated in Figure 7a, assigning each thread to a wide-multiply instruction (i.e‘, ai×bi) results in exces-sively high on-chip memory usage and communication cost. Moreover, the workload per thread is insufficient to amortize the thread initialization costs.  
如图所示 Figure 7a ，将每个线程分配到宽乘法指令（即'）， ai×bi) 会导致片上内存使用和通信成本极高。此外，每个线程的工作负载不足以摊销线程初始化成本。

#### ci

Allocating each thread to a word ci reduces commu- nication overhead and ensures a feasible per-thread workload (Figure 7b). However, it incurs significant warp divergence. Some threads execute a single wide-multiply instruction, while others execute Ns wide-multiply instructions.  
将每个线程分配到一个字 ci 可以减少通信开销，并确保每个线程的工作负载（ Figure 7b ）。然而，这会引发显著的曲速偏离。有些线程执行一条宽乘法指令，而有些线程执行 Ns 宽乘法指令。

#### Segc  SECC

Figure 7c illustrates the embarrassingly parallel algorithm, where each thread computes a segC, While this method eliminates communication overhead and warp divergence, it overburdens individual threads, markedly reducing the level of parallelism and prohibiting unrolling.  
Figure 7c 展示了令人尴尬的并行算法，每个线程计算一个 segC ，虽然该方法消除了通信开销和曲速发散，但会让单个线程负载过重，显著降低并行性水平并阻止展开。

#### 1 D

Many prior implementations adopt the one-dimensional parallelization strategy shown in Figure 7d, where each thread handles the computation of one or several lines of the computation diagram. While this approach significantly improves the balance among the aforementioned factors, the computation diagram indicates that it may not be the optimal solution.  
许多早期实现采用了一 Figure 7d 维并行化策略，如图中所示，每个线程负责计算图中的一行或多行。虽然这种方法显著改善了上述因素之间的平衡，但计算图表明它可能不是最优解。

### C. Two-Dimensional Parallelization Strategy  
C. 二维并行化策略

The computation diagram inspires the two-dimensional parallelization strategy (Figure 7e), where the calculation of each word is distributed among several threads, and each thread is also responsible for computing the partial products of several words. It opens up the potential for further optimizing on-chip resource usage, communication cost, parallelism degree, workload balance, and memory access pattern. With rectangle dimensions Nb1 and Nb2, the workload per thread is Nb1×Nb2 wide-multiply instructions, and the communication cost is approximately Nb1+Nb2. The Arithmetic Mean-Geometric Mean (AM-GM) inequality suggests that setting Nb1=Nb2 (a square shape) minimizes communication costs. In contrast, one-dimensional parallelization, where Nb1=N and Nb2=1, also represents a special rectangular case and incurs the highest communication costs as per the AM-GM inequality.  
计算图启发了二维并行化策略（ Figure 7e ），其中每个字的计算分布在多个线程中，每个线程还负责计算多个字的偏积。这为进一步优化片上资源使用、通信成本、并行度、工作负载平衡和内存访问模式提供了潜力。对于矩 Nb1 形尺寸和 Nb2 ，每线程的工作负载为 Nb1×Nb2 宽乘法指令，通信成本约 Nb1+Nb2 为 。算术均几何均值（AM-GM）不等式表明设置 Nb1=Nb2 （方形）可以最小化通信成本。相比之下，一维并行化（其中 Nb1=N 和 Nb2=1 ）也代表一种特殊的矩形情况，并且根据 AM-GM 不等式，通信成本最高。

### D. Parameterized Kernel  D. 参数化核

Algorithm 2 presents a simplified implementation of the parallel integer multiplication GPU kernel, which employs the two-dimensional parallelization strategy. The side length of the square is predetermined as Nb. Accordingly, the number of threads allocated to each side is TPS=Ns/Nb, and a segment pair is multiplied in parallel by NT=TPS×TPS threads. Each thread initially processes its designated workload, which consists of the partial products of N2b words. The iterative order also follows the Comba method to reduce on-chip resource usage and avoid frequent carry propagation. After that, the NT threads within the same thread group exchange data via shared memory, followed by accumulating the partial products of each word to produce the final results.  
Algorithm 2 呈现了 GPU 内核的简化实现，采用二维并行化策略。正方形的边长预先确定为 Nb 。因此，每边分配的线程数为 TPS=Ns/Nb ，并行将一段对线 NT=TPS×TPS 程并行乘以线程。每个线程最初处理其指定的工作负载，即 N2b 字的部分积。迭代顺序也遵循 Comba 方法，以减少片上资源使用并避免频繁的进位传播。之后，同一线程组内的 NT 线程通过共享内存交换数据，然后累积每个字的部分积以产生最终结果。

### E. Software-Hardware Co-Optimization  
E. 软硬件协同优化

#### Index Transposing  指标转置

Each word accumulates a different number of partial products. For example, sum[TPS] accu-mulates a significantly larger number of partial products compared to sum [0]. While the initial task distribution effectively avoids warp divergence during partial product calculation, it fails to do so during global accumulation. To this end, _IMCompiler_ applies index transposing before global accumu-lation. Prior to transposing, a group is formed by NT threads, where each thread is assigned a continuous sequence of Nb data items. After transposing, a group comprises Nb threads, where each thread manages items at strides of Nb. As long as  
每个字累积不同数量的部分积。例如，累 sum[TPS] 积的部分积数量显著多于 sum [0]。虽然初始任务分布在部分积计算中有效避免了曲速发散，但在全局累积时则未能做到这一点。为此，_IMCompiler_ 在全局累积前应用索引转置。在转置之前，由 NT 线程组成一个组，每个线程被分配一个连续 Nb 的数据项序列。转置后，一个组由 Nb 线程组成，每个线程以 的 Nb 步幅管理项目。只要

#### Algorithm 2 Parallel Integer Multiplication (UNIT_MUL)  
算法 2 并行整数乘法（UNIT_MUL）

**Def**: Ns is segment size, Nb is side length  
**Def**： Ns 是段长， Nb 是边长

**Def**: TPS=Ns/Nb ▹# of threads for a segment (one side)  
**定义** ： TPS=Ns/Nb ▹# 一段（单侧）螺纹的定义

**Def**: groupId=tid/TPS2▹ _**tid/TPS2**_ thread id  
**定义** ： groupId=tid/TPS2▹ _**tid/TPS 2**_ 帖子 ID

**Def: _tx, ty_ =** _tid/TPS%TPS, tid%TPS_  
**防御：_tx， ty_ =**_tid/TPS%TPS， tid%TPS_

**Def**: off=0+…+(tx+ty)▹ offset of the tile's partial sum  
**def**： off=0+…+(tx+ty)▹ 该牌部分和的偏移量

1:

cooperatively coalesced-load inputs to shared memory  
协作式合并负载输入到共享内存

2:

Areg[0:Nb]=load(Ashm[groupId][tx][0:Nb])

3:

Breg[0:Nb]=load(Bshm[groupId][ty][0;Nb])

4:

_carry, high, low_, i=0,0,0,0  
_高低，前进，_ i=0,0,0,0

5:

**for** n=0 **to** Nb **do**  
**为** n=0 Nb **去做**

6:

_carry, high_, low=0, _carry, high_  
_carry，high，carry，high，high_ low=0

7:

**for** i=0 to n **do**  
**为** i=0 n **去做**

8:

_carry_ ∥ _high_ ∥low+=regA[i]×regB[n−i]  
_高_ ∥low+=regA[i]×regB[n−i] _举_ ∥

9:

psumshm[groupId][λ(off+tx)][n]=low

10:

**for** n=Nb **to** 2 Nb **do**  
**对于** n=Nb **2** Nb **做**

11:

_carry, high_, low=0, _carry, high_  
_carry，high，carry，high，high_ low=0

12:

**for** i=n+1−Nb to Nb **do**  
**为** i=n+1−Nb Nb **去做**

13:

_carry_ ∥ _high_ ∥ _low_ +=regA[i]×regB[n−i]  
_高举低_ ∥ ∥ +=regA[i]×regB[n−i]

14:

psumshm[groupId][2(off+tx)+1][n−Nb]=low

15:

groupId,elemId=tid/Nb,tid%Nb▹ index transpose  
groupId,elemId=tid/Nb,tid%Nb▹ 指标转调

16:

**for** n=0 **to** _TPS_ **do**  
**对于** n=0 **TPS** **确实如此**

17:

**for** −=0 **to** 2n+1 **do**  
**为** −=0 2n+1 **去做**

18:

c[n]∥sum[n]+=psumshm[groupId][i++][elemId]

19:

**for** n=TPS to _2TPS_ **do**  
**对于** n=TPS _2TPS_ **的**

20:

**for _ = 0 to** 4TPS−2n−1 **do**  
**当_ = 0 时，要** 4TPS−2n−1 **执行**

21:

c[n]∥sum[n]+=psumshm[groupId][i++][elemId]

22:

_prop[0:TPS] =_ warp_shfl(c[0:TPS], _tid+_ 1)  
_prop[0：TPS] =_ warp_shfl（c[0：TPS]， _tid+_ 1）

23:

_res[0:TPS] =_ elemwise_add(sum[0:TPS], _prop[0:TPS]_)  
_res[0：TPS] =_ elemwise_add（sum[0：TPS]， _prop[0：TPS]_）

24:

**if** any thread overflow **then** ▹ rarely happens  
**如果**线程溢出， **那** ▹ 很少发生

25:

carry look-ahead addition  
前置加法

26:

return _res_  返回_水库_

Nb>N2s−−−√3, index transposing can effectively eliminate the warp divergence associated with global accumulation.  
Nb>N2s−−−√3 ，指标转置可以有效消除与全局累积相关的曲速散度。

#### Lazy Carrying  懒散携带

Each act of accumulation potentially gener- ates a carry. To minimize the communication overhead associated with carry propagation, _IMCompiler_ employs a lazy carry mechanism. Instead of instantly propagating the carry after each accumulation, the carries are temporarily stored in indi-vidual registers for each word during the accumulation phase. After accumulation, the carries are added to the corresponding words using a warp data shuffle operation. It is worth noting that the addition of carries, which are typically small values, infrequently results in an additional carry. Thus, _IMCompiler_ proactively verifies if any thread triggers an overflow, and executes a carry look-ahead addition only in such cases.  
每次积累行为都可能产生一次进位。为了最小化进位传播相关的通信开销，_IMCompiler_ 采用了懒惰进位机制。每次累积后不会瞬间传播进位，而是在累积阶段将进位暂时存储在每个字的独立寄存器中。累积后，进位通过变速数据洗牌操作添加到相应的字上。值得注意的是，进位的添加通常为小值，但很少导致额外的进位。因此，_IMCompiler_ 会主动验证任何线程是否触发溢出，并仅在此类情况下执行进位前瞻加法。

#### Tailored Caching Strategy  
定制缓存策略

Caching frequently accessed variables in on-chip resources is a common and well-studied optimization technique, while Algorithm 2 includes several un-conventional choices. First, given the irregular access pattern, inputs segA and segB are cached in registers rather than shared memory. This choice avoids bank conflict and non-coalesced memory access, which could occur if these inputs were stored in shared memory. However, shared memory is still employed as an intermediary when loading data into registers to reduce the number of global memory accesses. Second, the interme-diate results are exchanged through shared memory rather than warp data shuffle. This choice is necessitated by the complex memory access pattern induced by index transposing. Another key consideration is balancing the utilization of different on-chip resources to minimize peak usage. Given the overlap in the live intervals of intermediate results and the cached inputs segA and segB, allocating them to different on-chip resources can increase occupancy. Thus, the intermediate results, which are accessed less frequently, are allocated to shared memory.  
在片上资源中缓存频繁访问的变量是一种常见且研究充分的优化技术，同时 Algorithm 2 包含了若干非常规选择。首先，鉴于访问模式不规则，输入 segA 和 segB 被缓存在寄存器中，而非共享内存。这一选择避免了银行冲突和非合并内存访问，这在这些输入存储在共享内存中可能发生。然而，共享内存仍被用作加载数据到寄存器时的中介，以减少全局内存访问次数。其次，中间数据结果通过共享内存交换，而非曲速数据洗牌。这一选择是由于索引转置引发的复杂内存访问模式。另一个关键考虑是平衡不同片上资源的利用，以最小化峰值使用率。鉴于中间结果与缓存输入 segA segB 的存活区间重叠，将其分配到不同的片上资源可以增加占用率。因此，访问频率较低的中间结果被分配到共享内存。

SECTION V.  第五节。

## Evaluation  评价

In this section, we comprehensively evaluate _IMCompiler_ from two perspectives. First, the overall performance of different algorithms is assessed across several cryptosystems on various GPU platforms. Subsequently, a thorough set of experiments is conducted to evaluate the performance under different bit lengths and parallelism degrees, as well as to break down the effects of the frontend and backend.  
本节将从两个角度全面评估 _IMCompiler_。首先，评估不同算法在不同 GPU 平台上的多个密码系统中的整体性能。随后，进行一系列全面的实验，评估不同比特长度和并行度下的性能，并拆解前端和后端的影响。

We compare _IMCompiler_ with three GPU integer multi-plication implementations: _BASE, CGBN_, and _NTT. BASE_ represents the embarrassingly parallel approach, where each thread processes a distinct high-precision integer multiplication. _CGBN_, a state-of-the-art high-precision integer arithmetic library released by Nvidia [34], includes an efficient integer multiplication implementation built around the cooperative groups. Its computation diagram is similar to that in Figure 7d (i.e., Nb1=N), except that the workloads of several threads may be consolidated into a single one. This approach obviously requires more communication, but the communication pattern can be achieved via warp data shuffle operations. Moreover, with all variables stored in registers, _CGBN_ eagerly propagates the carry after each cooperative wide-multiply to optimize register usage. _NTT_ represents the NTT-based high-precision integer multiplication algorithm. A CPU implemen-tation based on GNU-MP (denoted as _GMP_) is also evaluated, serving as a reference for comparison.  
我们将 _IMCompiler_ 与三种 GPU 整数乘法实现进行比较：_BASE、CGBN_ 和 _NTT。BASE_ 代表了令人尴尬的并行方法，每个线程处理一个独立的高精度整数乘法。_CGBN_ 是 Nvidia 发布的先进高精度整数算术库 [34] ，包含围绕协作组构建的高效整数乘法实现。其计算图类似于（ Figure 7d 即 Nb1=N) ，但多个线程的工作负载可以合并为一个线程。这种方法显然需要更多的通信，但通信模式可以通过 warp 数据洗牌操作实现。此外，所有变量都存储在寄存器中，_CGBN_ 在每次合作式宽乘法后积极传播进位，以优化寄存器的使用率。_NTT_ 代表基于 NTT 的高精度整数乘法算法。还评估了基于 GNU-MP 的 CPU 实现量（记作 _GMP_），作为比较参考。

The experiments are conducted on several platforms, including an Nvidia H100 PCIe 80GB GPU, an Nvidia RTX4090 GPU, an Nvidia Jetson AGX Xavier, and an AMD Radeon 6900XT GPU. For Nvidia GPUs, the operating system is Ubuntu 18.04, with CUDA Driver 530.30.02 and CUDA Runtime 12.1. The AMD Radeon 6900XT operates on Ubuntu 20.04, accompanied by the ROCm Toolkit 5.0 [3]. The CPU implementation is executed on an Intel 13900KF CPU, with 8 performance cores and 16 efficient cores.  
实验在多个平台上进行，包括 Nvidia H100 PCIe 80GB GPU、Nvidia RTX4090 GPU、Nvidia Jetson AGX Xavier 和 AMD Radeon 6900XT GPU。Nvidia GPU 操作系统为 Ubuntu 18.04，CUDA 驱动为 530.30.02，运行时为 12.1。AMD Radeon 6900XT 运行于 Ubuntu 20.04，配备 ROCm Toolkit 5.0 [3] 。CPU 实现在 Intel 13900KF CPU 上运行，配备 8 个性能核心和 16 个高效核心。

### A. Performance of Cryptographic Systems  
答、密码系统的性能

Table I presents the performance of four GPU implemen-tations across three popular cryptosystems: RSA, ElGamal, and Paillier. The key size is 2048 bits, with performance for other key sizes analyzed in subsequent subsections. It reports the throughput of encryption, decryption, and homomorphic functions for each cryptosystem. Experiments are conducted on three different GPUs, except _CGBN_, which is excluded on AMD Radeon 6900XT due to its dependence on the specific features of Nvidia GPUs. For comparison, the table also includes throughput results for _GMP_.  
Table I 展示了三种流行密码系统：RSA、ElGamal 和 Paillier 上四种 GPU 实现的性能。密钥大小为 2048 位，其他密钥大小的性能将在后续子章节中分析。报告每个密码系统的加密、解密和同态函数吞吐量。实验在三款不同 GPU 上进行，除了 _CGBN_，因其依赖 Nvidia GPU 的特定特性，CGBN 在 AMD Radeon 6900XT 上被排除。作为对比，表格还包含_了 GMP_ 的吞吐量结果。

**Table I:** Overall performance of different algorithms on various cryptosystems (operations/second)  
**表 I：** 不同算法在不同密码系统上的整体性能（运算/秒）

[![Table I:- Overall performance of different algorithms on various cryptosystems (operations/second)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-table-1-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-table-1-source-large.gif)

_IMCompiler_ shows a 1.42× average throughput compared with _CGBN_, peaking at 1.49 ×. The advantages of _IMCompiler_ stem from two key aspects: the backend, which ensures an efficient implementation of UNIT_MUL, and the frontend, which leverages optimization opportunities across various high-level parameters. Although in-depth analyses with breakdowns are discussed in subsequent subsections, two key findings are highlighted here to provide insights derived from the overall performance. The algorithms exhibit varying throughputs on different GPUs due to diverse hardware specifications. In-terestingly, the relative speedup also differs among different GPUs. The relative performance between _IMCompiler_ and _CGBN_ is 1.39× on Nvidia H100 GPUs and 1.46× on Nvidia RTX 4090 GPUs. Breakdown analysis reveals that the primary bottleneck for _IMCompiler_ is integer ALUs, while _CGBN_ is mainly blocked by communication instructions. Consequently, _IMCompiler_ exhibits higher speedups on GPUs with more integer ALUs. Additionally, the breakdowns suggest that _IM-Compiler_ will achieve even greater speedups for larger bit lengths, owing to the frontend's optimization. As computational power continues to evolve and bit lengths increase to ensure security, we anticipate that _IMCompiler_ will deliver even superior performance in the future.  
_IMCompiler_ 的平均吞吐量相比 _CGBN_ 为 1.42×，峰值为 1.49×。_IMCompiler_ 的优势来自两个关键方面：后端确保 UNIT_MUL 的高效实现，以及前端，利用各种高级参数的优化机会。虽然后续子章节会讨论深入分析和分解，但这里重点介绍两个关键发现，以提供整体性能的见解。由于硬件规格不同，算法在不同 GPU 上的吞吐量存在差异。有趣的是，不同 GPU 的相对加速也存在差异。_IMCompiler_ 与 _CGBN_ 之间的相对性能在 Nvidia H100 GPU 上为 1.39×在 Nvidia RTX 4090 GPU 上为 1.46×。拆解分析显示，_IMCompiler_ 的主要瓶颈是整数算术逻辑单元（ALU），而 _CGBN_ 主要被通信指令阻挡。因此，_IMCompiler_ 在拥有更多整数算术逻辑单元的 GPU 上表现出更高的加速。此外，分析结果表明，由于前端优化，_IM-Compiler_ 在更长的比特长度下将实现更大的加速。随着计算能力的不断发展和位长的增加以确保安全性，我们预计 _IMCompiler_ 未来将带来更卓越的性能。

Moreover, the baseline GPU implementation outperforms _GMP_ by a factor of 294.3 ×, supporting the argument that GPUs serve as effective platforms for accelerating high-precision integer multiplication operations used in cryptosystems. Meanwhile, both _IMCompiler_ and _CGBN_ demonstrate markedly higher throughput than _BASE_ and _NTT_, indicating that the bit lengths in the field of cryptography exceed the capacity of a single thread, yet remain insufficient to utilize the NTT algorithm efficiently. Furthermore, as _IMCompiler_ does not rely on cooperative groups or frequent warp data shuffle operations, it maintains compatibility with non-CUDA GPUs. However, when deployed on an AMD 6900XT GPU, the performance of _IMCompiler_ drops to 11.6% of that on an Nvidia H100 GPU, while the throughput of _BASE_ remains at 12.5% of its level. While _BASE_ can function directly on AMD GPUs, _IMCompiler_ requires a modification: the warp data shuffle operation at Line 22 of Algorithm 2 must be replaced with a shared memory data exchange. Although the peak shared memory usage remains unchanged, it is worth noting that data exchange via shared memory is less efficient than warp data shuffle operations.  
此外，基础 GPU 实现性能比 _GMP_ 高出 294.3×倍，支持 GPU 作为加速加密系统中高精度整数乘法操作的有效平台的观点。同时，_IMCompiler_ 和 _CGBN_ 的吞吐量明显高于 _BASE_ 和 _NTT_，表明密码学领域的位长超过单线程容量，但仍不足以高效利用 NTT 算法。此外，由于 _IMCompiler_ 不依赖协作组或频繁的 warp data 洗牌操作，它保持了与非 CUDA GPU 的兼容性。然而，当部署在 AMD 6900XT GPU 上时，_IMCompiler_ 的性能降至 Nvidia H100 GPU 的 11.6%，而 _BASE_ 的吞吐量仍为其水平的 12.5%。虽然 _BASE_ 可以直接运行在 AMD GPU 上，但 _IMCompiler_ 需要修改： Algorithm 2 必须将第 22 行的 warp data shuffle 操作替换为共享内存数据交换。尽管峰值共享内存使用率保持不变，但值得注意的是，通过共享内存交换数据交换的效率低于 warp data shuffle 操作。

### B. End-To-End Performance of Cryptographic Applications  
B. 密码学应用的端到端性能

This section evaluates the end-to-end performance using six widely-used cryptographic applications. **Digital signatures (DS)** verify the authenticity of digital documents [7]. A digital signature can be created with a decryption operation (i.e., big integer exponentiation) of RSA cryptosystem. It proves the sender's knowledge of the private key, thereby proving the message's authenticity. **Secret sharing (SS)** disperses a secret among a group, ensuring no single member can reconstruct the secret alone [20]. In our experimental setup, each participant executes the DH key exchange n=3 times (i.e., big integer exponentiation), with n representing the total number of participants. **Private Set Intersection (PSI)** allows two parties to identify common elements in their datasets without revealing non-intersecting elements, vital for vertical federated learning [42]. In PSI, each party conducts two DH key exchanges for each element in their set. Privacy-**preserving SQL SUM querying (PPSQL)** enables data analysis on sensitive information that must remain encrypted [6]. Utilizing the Paillier cryptosystem for encryption, its homomorphic property supports operations such as sums on encrypted data, where each component undergoes an HAdd (i.e., big integer multiplication) operation. **Privacy-preserving linear regression (PPLR)** analyzes encrypted user data [46]. Utilizing the Paillier cryptosystem, users encrypt their data and send it to the model owner, who computes the linear regression directly on the encrypted data. Each encrypted element is mul-tiplied by corresponding features using HScale (i.e., big integer exponentiation), with results aggregated through HAdd. Post-**quantum cryptography (PQC)** is rapidly advancing due to its security against quantum computer attacks. Our experiments focus on the widely-used NTRU encryption algorithm [17], defined by the equation e=r⋅h+ m mod q, where r,h, and **m** are polynomials with coefficients as big integers.  
本节利用六种广泛使用的密码学应用评估端到端性能。 **数字签名（DS）验证**数字文档 [7] 的真实性。数字签名可以通过 RSA 密码系统的解密操作（即大整数指数）创建。它证明发送方对私钥的了解，从而证明消息的真实性。 **秘密共享（SS）** 将秘密分散到一个小组，确保没有单个成员能单独重建该秘密 [20] 。在我们的实验设置中，每个参与者执行 DH 密钥交换 n=3 时间（即大整数指数），代表 n 参与者总数。 **私有集合交叉（PSI）** 允许双方识别数据集中的共同元素而不暴露非相交元素，这对垂直联邦学习 [42] 至关重要。在 PSI 中，双方对其集合中的每个元素进行两次 DH 密钥交换。保护隐私的 **SQL SUM 查询（PPSQL）** 支持对必须保持加密的敏感信息进行数据分析 [6] 。利用 Paillier 密码系统进行加密，其同态性质支持对加密数据的求和等操作，每个组件都经历 HAdd（即大整数乘法）操作。隐私保护**线性回归（PPLR）** 分析加密用户数据 [46] 。利用 Paillier 密码系统，用户将数据加密后发送给模型所有者，后者直接在加密数据上计算线性回归。每个加密元素通过 HScale（即大整数指数）对相应特征进行多倍波化，结果通过 HAdd 汇总。 后**量子密码学（PQC）** 因其对量子计算机攻击的安全性而迅速发展。我们的实验聚焦于广泛使用的 NTRU 加密算法 [17] ，定义为方程 e=r⋅h+ m 模 q ，其中 r,h ， **和 m** 是系数为大整数的多项式。

[![Fig. 8: - End-to-end performance (relative to base in Table ii) of different algorithms on various cryptographic applications](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-8-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-8-source-large.gif)

**Fig. 8:   图8：**

End-to-end performance (relative to base in Table ii) of different algorithms on various cryptographic applications  
不同算法在各种密码学应用上的端到端性能（相对于基 ） Table ii 的表现

Show All  全部显示

**Table II:** Absolute throughput of base (operations/second)  
**表二：** 基准绝对吞吐量（运算/秒）

[![Table II:- Absolute throughput of base (operations/second)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-table-2-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-table-2-source-large.gif)

The key sizes are uniformly set at 2048 bits for all applications, and the polynomial degree for NTRU is 509. Our end-to-end evaluation encompasses a variety of GPUs, including Nvidia's H100, RTX4090, Jetson AGX Xavier, and AMD's Radeon 6900XT, covering both server and embedded GPUs from Nvidia and AMD. Table II presents the absolute throughput of _BASE_, and Figure 8 illustrates the relative throughput compared to _BASE_ of different methods.  
所有应用的密钥大小统一设置为 2048 位，NTRU 的多项式次数为 509。我们的端到端评估涵盖了多种 GPU，包括 Nvidia 的 H100、RTX4090、Jetson AGX Xavier 和 AMD 的 Radeon 6900XT，涵盖了 Nvidia 和 AMD 的服务器和嵌入式 GPU。 Table II 展示了 _BASE_ 的绝对吞吐量，并 Figure 8 展示了不同方法相对于 _BASE_ 的相对吞吐量。

In summary, _IMCompiler_ achieves 1.39 x throughput on average compared with _CGBN_. The speedup stems from two key aspects: the backend, which ensures an efficient imple-mentation of UNIT_MUL, and the frontend, which leverages the optimization opportunities across various integer multipli-cation variants. Both _IMCompiler_ and _CGBN_ show significantly higher throughput compared to _BASE_. It indicates that merely leveraging inter-operation parallelism is insufficient for cryptographic applications. While _CGBN_ may be fine-tuned for high-end GPUs, _IMCompiler's_ frontend-IR-backend frame-work makes it more adaptable across different GPU platforms. It is evident in _IMCompiler_ achieving a 1.43 x speedup on the embedded GPUs, Jetson AGX Xavier, surpassing the average.  
总之，_IMCompiler_ 的平均吞吐量是 _CGBN_ 的 1.39 倍。加速来自两个关键方面：后端确保高效实现 UNIT_MUL，以及前端，利用各种整数乘复变体的优化机会。_IMCompiler_ 和 _CGBN_ 的吞吐量显著高于 _BASE_。这表明仅仅利用操作间并行性对密码学应用来说是不够的。虽然 _CGBN_ 可以针对高端 GPU 进行微调，但 _IMCompiler 的_前端-红外-后端帧工作使其在不同 GPU 平台上更具适应性。_IMCompiler_ 在嵌入式 GPU Jetson AGX Xavier 上实现了 1.43 倍的加速，超过平均水平。

The performance improvement of _IMCompiler_ over _CGBN_, while appearing modest, is indeed significant. _CGBN_ sets a high standard in implementation, incorporating a comprehensive set of best practices and optimizations, and employing assembly-level code for critical operations. Achieving further speedups presents a significant challenge, requiring innovative strategies for even modest gains.  
_IMCompiler_ 相较_于 CGBN_ 的性能提升虽显得有限，但确实显著。_CGBN_ 在实现上树立了高标准，整合了全面的最佳实践和优化，并在关键操作中使用汇编级代码。实现进一步加速仍是重大挑战，即使是适度的提升，也需要创新策略。

[![Fig. 9: - Throughput with varying bit lengths](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-9-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-9-source-large.gif)

**Fig. 9:   图9：**

Throughput with varying bit lengths  
不同比特长度下的吞吐量

Show All  全部显示

Absolute throughput of cgbn  
cgbn 的绝对通量

[![Table - Absolute throughput of cgbn](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-table-3-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-table-3-source-large.gif)

### C. Impact of Bit Length  
C. 位长的影响

This subsection evaluates the performance of various imple-mentations across different bit lengths on Nvidia RTX4090. For a clear illustration, Figure 9 presents the performance in terms of relative throughput compared to _CGBN_, and the absolute throughput of _CGBN_ is listed below for reference. The bit lengths investigated go up to _32K_, corresponding to the maximum bit length that _CGBN_ supports. In fact, _IMCompiler_ can accommodate even larger bit lengths, thanks to the segmented integer multiplication algorithm.  
本节评估了不同比特长度下的各种实现在 Nvidia RTX4090 上的性能。为了清晰说明， Figure 9 将以相对于 _CGBN_ 的相对吞吐量来展示性能，并列出 _CGBN_ 的绝对吞吐量供参考。研究的比特长度可达 _32K_，对应 _CGBN_ 支持的最大比特长度。事实上，_IMCompiler_ 还能支持更大的比特长度，这得益于分段整数乘法算法。

For bit lengths ≤512, the _IMCompiler's_ backend exhibits a lower throughput than _CGBN_, as _IMCompiler_ does not optimize the kernel code for each individual bit length. For example, our implementation sets the value of Ns to 512, which implies that the same implementation is used for bit lengths ≤512. In fact, _BASE_ is a more appropriate choice for such bit lengths. Thanks to the compiler-like structure of _IMCompiler_, the frontend can adopt a hybrid strategy that intelligently switches to the embarrassingly parallel approach. Thus, the entire frontend-IR-backend structure _of IMCompiler_ achieves a much higher throughput than _CGBN_. When bit length ≥1024, _IMCompiler's_ backend surpasses _CGBN_. For bit lengths range from 1K to 32_K, IMCompiler_ achieves an average speedup of 1.31×, with a peak at 1.57× and a minimum of 1.17×. The relative speedup initially decreases (1K ∼8K) and subsequently shows an upswing (8K∼32K).  
对于位长 ≤512 ，_IMCompiler 后_端的吞吐量低于 _CGBN_，因为 _IMCompiler_ 不会针对每个单个位长优化内核代码。例如，我们的实现将 的值 Ns 设置为 512，这意味着同样的实现也用于 ≤512 的位长。事实上，对于这种位长，_BASE_ 更为合适。得益于 _IMCompiler_ 的编译器式结构，前端可以采用一种混合策略，智能地切换到令人尴尬的并行方法。因此，_IMCompiler_ 的整个前端-红外-后端结构的吞吐量远高于 _CGBN_。当位长 ≥1024 为时，_IMCompiler 的_后端超过 _CGBN_。对于位长范围为 1K _32K，IMCompiler_ 的平均加速为 1.31× ，峰值为，峰值为 1.57× 1.17× 。相对加速最初下降 (1K ∼8K) ，随后呈现上升 (8K∼32K) 。

_NTT_ exhibits notably low throughput for bit lengths ≤32K. Its time complexity is O(N32log(N32)), as most GPUs have native 32-bit integer multiply instructions. Even with a lower time complexity, _NTT_ actually uses more multiplication operations for integers that are not sufficiently large, as it needs to pad the integer to 2N, employ an extremely large prime to prevent overflow, and execute both the NTT and the inverse NTT for each multiplication. Table III shows the throughput of _IMCompiler_ and _NTT_ at large bit lengths. _NTT_ outperforms _IMCompiler_ when the bit length reaches _256K_, owing to its superior time complexity. Thus, _IMCompiler_ should switch to NTT-based methods for extensive bit lengths via its frontend. Notably, at the _256K_ bit length, a performance inflection point occurs due to the inadequate shared memory for full NTT computation, compelling the initial NTT round to rely on slower global memory. The 4-step NTT algorithm [4] effectively addresses this issue, and adopting it can further enhance _NTT's_ performance.  
_NTT_ 在位长 ≤32K 时吞吐量显著较低。其时间复杂度为 O(N32log(N32)) ，因为大多数 GPU 原生支持 32 位整数乘法指令。即使时间复杂度较低，_NTT_ 实际上对整数不够大时使用更多乘法运算，因为它需要填充整数为 2N ，使用极大的素数以防止溢出，并对每次乘法同时执行 NTT 和逆 NTT。 Table III 显示了 _IMCompiler_ 和 _NTT_ 在大比特长度下的吞吐量。当位长达到 _256K_ 时，_NTT_ 表现优于 _IMCompiler_，这得益于其更优越的时间复杂度。因此，_IMCompiler_ 应通过前端转向基于 NTT 的长比特方法。值得注意的是，在 _256K_ 位长处，由于共享内存不足以完成完整的 NTT 计算，导致性能拐点出现，迫使初始 NTT 轮依赖较慢的全局内存。四步 NTT 算法 [4] 有效解决了这一问题，采用该算法还能进一步提升 _NTT_ 的性能。

[![Fig. 10: - Relative throughput of different variants](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-10-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-10-source-large.gif)

**Fig. 10:   图10：**

Relative throughput of different variants  
不同变体的相对通量

Show All  全部显示

**Table III:** Throughput for extensive bit lengths (in mops)  
**表 III：** 宽比特长度的吞吐量（在 MOP 中）

[![Table III:- Throughput for extensive bit lengths (in mops)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-table-4-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-table-4-source-large.gif)

### D. Effects of the Frontend  
D. 前端的影响

As discussed above, _IMCompiler_ shows a lower speedup for standard multiplication than for end-to-end usages. The additional speedup is contributed by the frontend, which leverages optimization opportunities inherent in various special variants used in cryptosystems. Figure 10 shows the relative throughput of _CGBN_ and _IMCompiler_ for various integer multiplications, in comparison with the standard one. The experiments are conducted on Nvidia RTX4090 with bit length s ranging from 1K to 32 K, while the average throughput is reported.  
如上所述，_IMCompiler_ 在标准乘法下的速度提升低于端到端使用。额外的加速来自前端，前端利用了加密系统中各种特殊变体固有的优化机会。 Figure 10 显示了 _CGBN_ 和 _IMCompiler_ 在各种整数乘法下的相对吞吐量，与标准相比较。实验在 Nvidia RTX4090 上进行，位长从 32 1K K 不等，同时报告平均吞吐量。

The squaring operation is pivotal in cryptosystems due to its role in fast modular exponentiation. _IMCompiler_ effectively eliminates redundant UNIT _MUL function calls during squaring, yielding 1.51× throughput. On the other hand, _CGBN_ merely avoids redundant input loading, as eliminating redundant computations in the squaring operation is challenging without a frontend. In fact, _CGBN's_ squaring operation exhibits a comparable speedup to _IMCompiler's_ multiply-constant operation, as both reduce memory access by approximately 25%. However, the resulting speedup is marginal. Given that GPUs employ massively parallelism and inter-warp context switching to achieve thread-level parallelism, the memory access latency has already been well hidden.  
平方运算在密码系统中至关重要，因为它在快速的模指数运算中的作用。_IMCompiler_ 在平方过程中有效消除了冗余的 UNIT _MUL 函数调用，从而实现 1.51× 了吞吐量。另一方面，_CGBN_ 仅仅避免了冗余输入加载，因为没有前端，消除平方运算中的冗余计算是个挑战。事实上，_CGBN 的_平方运算速度与 _IMCompiler_ 的乘法常数运算相当，两者都将内存访问量降低约 25%。然而，最终的加速幅度有限。鉴于 GPU 采用大规模并行性和跨曲速上下文切换以实现线程级并行，内存访问延迟已被很好地隐藏。

[![Fig. 11: - Warp state sampling of different parallelization strategies](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-11-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-11-source-large.gif)

**Fig. 11:   图11：**

Warp state sampling of different parallelization strategies  
不同并行化策略的曲速状态采样

Show All  全部显示

Both _CGBN_ and _IMCompiler_ offer tailored implementations for multiply-low and multiply-high operations. The segmented integer multiplication algorithm employed by _IMCompiler_ simplifies the optimization process for eliminating redundant computations. As a result, _IMCompiler_ achieves a higher speedup compared to _CGBN_. Notably, this performance improvement is especially pronounced in multiply-high operations, which is 50.9% higher than _CGBN_, owing to _IMCom-piler's_ conditional early termination strategy.  
_CGBN_ 和 _IMCompiler_ 都为乘低和乘高运算提供定制化实现。_IMCompiler_ 采用的分段整数乘法算法简化了消除冗余计算的优化过程。因此，_IMCompiler_ 相比 _CGBN_ 实现了更高的加速。值得注意的是，这种性能提升在乘法高运算中尤为明显，高乘法比 _CGBN_ 高 50.9%，这得益于 _IMCom-piler 的_条件提前终止策略。

Detailed experiments indicate that the relative speedup _of_ both the square and multiply-high operations exhibit an upward trend as the bit lengths increase. As shown in Figure 3, these two operations account for 19.2% and 35.6% of the total multiplication operations, respectively. Thus, _IMCompiler_ is expected to yield more significant end-to-end performance gains for cryptosystems with larger key size.  
详细实验表明，随着位长增加，平方运算和乘高运算的相对加速呈上升趋势。如图所示 Figure 3 ，这两种运算分别占总乘法运算的 19.2%和 35.6%。因此，_IMCompiler_ 预计在较大密钥大小的加密系统中，端到端性能提升将更显著。

### E. Effects of the Backend  
E. 后端的影响

This subsection evaluates the effects _of_ the backend by comparing the parallelization strategies discussed in Section IV-A. The experiments are conducted on Nvidia RTX4090 using 2048-bit integers. The breakdowns of their execution times are measured through periodic sampling of warp scheduler states (Figure 11). The profiler and the details of these metrics are described in [35]. A dummy metric, Speedup, is added for intuitive comparison. It indicates how much execution time is actually unnecessary if a method were given the same execution time as _BASE_. The approach of allocating a wide-multiply operation to a thread shows very poor performance (i.e., 5.13 x slower than _BASE_) and is thus excluded.  
本节通过比较文中 Section IV-A 讨论的并行化策略来评估后_端的影响。_ 实验在 Nvidia RTX4090 上使用 2048 位整数进行。其执行时间的分解通过周期性抽样 warp scheduler 状态（ Figure 11 ）。分析器及这些指标的详细信息详见 [35] 。为直观比较，新增了一个虚拟指标 Speedup。它表示如果方法与 _BASE_ 执行时间相同，实际执行时间是多么多余。将宽乘法运算分配给线程的方法性能非常差（比 _BASE_ 慢 5.13 倍），因此不予考虑。

Although _IMCompiler_ relies on slower shared memory for data exchange, it experiences fewer wait stalls compared with _CGBN_ due to fewer communication requirements. Yet, this reduction does not completely account for speedup. _IMCom-piler_ suffers the most portion of math stalls, indicating that the bottleneck is the capacity of integer ALUs. This suggests that adding more integer ALUs could enhance _IMCompiler's_ advantages, which explains why _IMCompiler_ exhibits higher relative speedup on Nvidia RTX4090 than on Nvidia H100. The 1_D-shmem_ necessitates a similar amount of communication as _CGBN_ but uses more expensive shared memory for data exchange, resulting in significantly more wait stalls.  
虽然 IMCompiler 依赖较慢的共享内存进行数据交换，但由于通信需求较少，_IMCompiler_ 相比 _CGBN_ 遭遇的等待停顿更少。然而，这一减少并不能完全解释速度提升。_IMCompiler_ 遭遇数学停滞的比例最大，表明瓶颈在于整数算术逻辑单元的容量。这表明增加更多整数算术逻辑单元可以增强 _IMCompiler_ 的优势，这也解释了为什么 _IMCompiler_ 在 Nvidia RTX4090 上相较于 Nvidia H100 的相对加速。_1D-smem_ 所需的通信量与 _CGBN_ 相似，但使用更昂贵的共享内存进行数据交换，导致等待停滞次数显著增加。

However, this waiting time also allows the math instructions to be finished, thereby resulting in fewer math stalls. Thus, its throughput lags only 23.0% behind _CGBN_.  
然而，这种等待时间也允许数学指令完成，从而减少数学停滞。因此，其吞吐量仅比 _CGBN_ 慢 23.0%。

Other strategies show poor performance and unsatisfactory profiling outcomes. The strategy of dedicating a thread to each word (i.e., Ci) incurs significant warp divergence. It also suffers from a large number of memory stalls due to its non-coalesced memory access pattern. Moreover, _BASE_ requires more resource allocation, leading to much lower occupancy and thus reduced thread-level parallelism. This condition negatively affects memory latency hiding, causing _BASE_ to experience many memory stalls and wait stalls.  
其他策略表现不佳，分析结果不理想。为每个字专用一个线程的策略（ Ci) 即引发显著的曲速发散）。由于其非聚合的内存访问模式，BASE 还存在大量内存停顿。此外，_BASE_ 需要更多的资源分配，导致内存占用率大幅降低，线程级并行性降低。这种状况对内存延迟隐藏产生负面影响，导致 _BASE_ 经历多次内存停顿和等待停顿。

**Table IV:** Breakdown analysis of backend optimizations  
**表 IV：** 后端优化的分解分析

[![Table IV:- Breakdown analysis of backend optimizations](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-table-5-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-table-5-source-large.gif)

Table IV presents a breakdown analysis of backend optimizations' with an unoptimized two-dimensional parallelization kernel as the baseline. Each optimization significantly boosts the performance of the kernel, particularly index trans-posing, which reduces severe warp divergence and achieves a 1.27 x speedup. It also compares the calculated speedup (i.e., the product of the four optimizations' speedups) with the actual observed overall speedup. The actual speedup is slightly lower than the predicted one, mainly due to the non-independence _of_ the optimizations, where the tailored caching strategy reduces more bank conflicts without the workload distribution optimization by index transposing.  
以未优化的二维并行化核为基线，对后端优化进行了拆解分析。每一次优化都显著提升了内核的性能，特别是索引转置，它减少了严重的曲速发散，实现了 1.27 倍的加速。它还比较了计算出的加速（即四个优化加速的乘积）与实际观察到的整体加速。实际加速略低于预测，主要由于优化不独立，定制缓存策略减少了更多银行冲突，而没有通过索引转置优化实现工作负载分配优化。 Table IV

### F. Impact of Number of Input Elements  
F. 输入元素数量的影响

In the previous experiments, the number of input elements is set to 222, which fully utilized the GPU resources for all methods. This subsection explores the minimum number of input elements each method requires to reach peak throughput. Figure 12 shows the throughput for each method at different input sizes on the Nvidia RTX4090 using 2048-bit integers, normalized to the throughput at 223 input elements.  
在之前的实验中，输入元素数设置为 2 22 ，充分利用了所有方法的 GPU 资源。本节探讨每种方法达到峰值吞吐量所需的最小输入元素数。 Figure 12 在 Nvidia RTX4090 上，使用 2048 位整数显示每种方法在不同输入尺寸下的吞吐量，该整数归一化为 2 个 23 输入元素时的吞吐量。

[![Fig. 12: - Throughput across different parallelism degrees](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-12-source-small.gif)](https://ieeexplore.ieee.org/mediastore/IEEE/content/media/10763948/10764425/10764432/10764432-fig-12-source-large.gif)

**Fig. 12:   图12：**

Throughput across different parallelism degrees  
不同并行度的吞吐量

Show All  全部显示

To achieve at least 97% of the maximum attainable through-put, _BASE, CGBN_, and _IMCompiler_ require a minimum _of_ 220, 218, and 216 input elements, respectively. _IMCompiler's_ two-dimensional parallelization strategy allows for more threads per multiplication operation without significantly increasing communication overhead. This enables effective utilization of the GPU's capabilities even with fewer concurrent high-precision integer multiplications. In contrast, _BASE_ assigns one item per thread, necessitating a larger number _of_ items to reach optimal throughput. Notably, the relative performance does not scale linearly with the number of threads per item. For instance, assigning 16 threads per item in _IMCompiler_ does not mean that _BASE_ requires 16 x the number _of_ items for comparable throughput due to its lower occupancy and higher resource demands.  
为了达到至少 97%的最大可直通输入，_BASE、CGBN_ 和 _IMCompiler_ 分别_需要至少 2_ 20 、2 18 和 2 个 16 输入元素。_IMCompiler_ 的二维并行化策略允许每次乘法操作使用更多线程，同时显著增加通信开销。这使得即使同时进行的高精度整数乘法减少，也能有效利用 GPU 的能力。相比之下，_BASE_ 为每个线程分配一个项目，需要_更多项目才能_达到最佳吞吐量。值得注意的是，相对性能不会随每个项目线程数量线性增长。例如，在 _IMCompiler_ 中为每个项目分配 16 个线程并不意味着 _BASE_ 需要 16 倍_的项目数量_才能实现可比吞吐量，因为其占用率较低且资源需求更高。

SECTION VI.  第六部分。

## Related Work  相关工作

High-precision integer multiplication is a well-studied area, offering numerous algorithms beyond the schoolbook one. The Comba method [40], which is also adopted by _IMCompiler_, minimizes intermediate variables by iterating over the words of the output. Khachatrian et al. propose a software-efficient improvement _of_ the schoolbook multiplication algorithm [22], which saves about 33% arithmetic operations. Furthermore, many divide-and-conquer algorithms have been proposed to reduce time complexity, such as the Karatsuba algorithm [21] and Toom-Cook multiplication [24]. A comprehensive evaluation of these algorithms on Field Programmable Gate Arrays (FPGAs) is provided in [40]. The NTT-based method [43] offers an even lower time complexity, but its benefits become evident only with extremely large bit lengths (e.g., _256K_ in our experiments). Furthermore, in cryptography, NTT is often employed to accelerate polynomial multiplication [26]. Our experiments also indicate that for small-scale polynomials, schoolbook polynomial multiplication may outperform NTT, and the optimization strategies within _IMCompiler_ are adapt-able to these scenarios.  
高精度整数乘法是一个研究较为深入的领域，提供了许多超越教科书算法的算法。Comba 方法 [40] 也被 _IMCompiler_ 采用，通过对输出的字进行迭代来最小化中间变量。Khachatrian 等人提出了对教科书乘法算法的软件高效改进 [22] ，该算法节省约 33%的算术运算。此外，许多分治算法已被提出以降低时间复杂度，如 Karatsuba 算法 [21] 和 Toom-Cook 乘法 [24] 。这些算法在现场可编程门数组（FPGA）上的全面评估见于。 [40] 基于 NTT [43] 的方法提供了更低的时间复杂度，但其优势仅在极大比特长度（例如我们实验中的 _256K_）时显现。此外，在密码学中，NTT 常被用来加速多项式乘法 [26] 。我们的实验还表明，对于小尺度多项式，教科书式多项式乘法可能优于 NTT，IMCompiler 中的优化策略能够适应这些场景。

Numerous studies have explored the acceleration of high-precision integer multiplication on GPUs, each with its unique focus. GPUMP [47], an early high-precision integer arithmetic library, employs _BASE_ as its multiplication function, making it more suitable for integers with bit lengths under 512. Conversely, _CGBN_ [34] supports a wide range _of_ bit lengths, though it primarily shows advantages for moderate bit lengths. Honda et al. enhance the one-dimensional parallelization strategy by using the Toom-Cook algorithm to reduce computation, thereby improving performance [18]. On the other hand, some studies focus on the multiplication of extremely large integers. For instance, Dieguez et al. develop a technique to efficiently distribute a single multiplication operation across an entire GPU [11], which enhances the level of parallelism but necessitates global communication. In studies [13] and [14], a hybrid of Karatsuba and the Strassen algorithms is proposed to optimize the multiplication of integers in millions of bits. Their divide-and-conquer approach aims to reduce computation but does not decouple the design details from high-level parameters due to the conquering stage, setting their goals apart from _IMCompiler's_. Other studies [23], [30] focus on multiple-precision integer arithmetic, where the two integers involved can have different bit lengths. Particularly, Kitano et al. introduce a novel data structure, the product digit table, to reduce warp divergence for multiple-precision integer multiplication [23], a concept that aligns with the design philosophy behind _IMCompiler's_ lazy carrying mechanism.  
大量研究探讨了 GPU 上高精度整数乘法的加速，每款 GPU 都有其独特的关注点。 [47] GPUMP 是早期高精度整数算术库，采用 _BASE_ 作为乘法函数，更适合位长小于 512 的整数。相反，_CGBN_ [34] 支持_广泛的位长_范围，但主要在中等位长方面表现出优势。Honda 等人通过使用 Toom-Cook 算法减少计算量，增强了一维并行化策略，从而提升性能 [18] 。另一方面，一些研究关注极大整数的乘法。例如，Dieguez 等人开发了一种技术，能够高效地将单一乘法运算分布到整个 GPU [11] 上，这提高了并行性水平，但需要全局通信。在研究 [13] 和 [14] 中，提出了 Karatsuba 和 Strassen 算法的混合，以优化整数以百万比特为单位的乘法。他们的分而治之方法旨在减少计算，但由于征服阶段的限制，不将设计细节与高层参数分离，这使得他们的目标区别于 _IMCompiler_。其他研究 [23] 则 [30] 聚焦于多精度整数算术，其中涉及的两个整数可以具有不同的位长。特别是，Kitano 等人引入了一种新颖的数据结构——乘积数字表，以减少多精度整数乘法的曲速散度 [23] ，这一概念与 _IMCompiler_ 懒散传递机制的设计理念相契合。

SECTION VII.  第七节。

## Conclusion and Future Work  
结论与未来工作

This paper introduces _IMCompiler_, a compiler-like frame-work aimed at automating the generation _of_ GPU kernels for diverse integer multiplication tasks used in cryptographic ap-plications. With a frontend-IR-backend structure, _IMCompiler_ decouples architecture-specific optimizations from high-level parameters, enabling automatic generation of efficient GPU kernels for integer multiplications with varying bit lengths and special variants. Guided by the proposed computation diagram, _IMCompiler_ implements a set of optimizations to align algorithms with hardware, including two-dimensional parallelization, tailored caching strategy, index transposing, and lazy carrying. Thorough evaluations show that _IMCom-piler_ achieves throughput improvements of 4.47 x and 142 x compared to _BASE_ and _NTT_, respectively. When compared with _CGBN, IMCompiler_ exhibits a 1.42 x throughput for cryptosystems with a 2048-bit key. The performance further increases with larger key sizes and higher-performance GPUs.  
本文介绍了 _IMCompiler_，一种类编译器的框架工作，旨在_自动化生成用于_密码学应用映射中各种整数乘法任务的 GPU 内核生成。通过前端-红外-后端结构，_IMCompiler_ 将架构特定的优化与高层参数解耦，实现高效 GPU 内核，支持不同位长和特殊变体的整数乘法。在所提计算图的指导下，_IMCompiler_ 实现了一套优化，使算法与硬件对齐，包括二维并行化、定制缓存策略、索引转置和懒散进位。全面评估显示，_IMCom-piler_ 相比 _BASE_ 和 _NTT_ 分别实现了 4.47 倍和 142 倍的吞吐量提升。与 _CGBN 相比，IMCompiler_ 在拥有 2048 位密钥的加密系统中展现出 1.42 倍的吞吐量。随着密钥尺寸增大和性能更高的 GPU 性能进一步提升。

### ACKNOWLEDGMENT  致谢

We appreciate the anonymous reviewers for their constructive comments and suggestions. This work is supported by the Natural Science Foundation of China grant No. 62432005, No. 62402282, and No. 62372272, Department _of_ Science & Tech-nology of Shandong Province grant No. SYS202201, Quan Cheng Laboratory grant No. QCLZD202302, Taishan Scholars Program No. tsqn202211281. This work was supported by Ant Group Research Fund. The authors from Ant Group are supported by the Leading Innovative and Entrepreneur Team Introduction Program of HangZhou (Grant No.TD2020001).  
感谢匿名评审者提供的建设性意见和建议。本工作由中国自然科学基金会 62432005 号、62402282 号和 62372272 号资助，山东_省科技系_资助号为 SYS202201，全城实验室资助编号。QCLZD202302，台山学者项目编号。TSQN202211281。本工作由蚁群研究基金支持。蚁群的作者获得杭州领先创新与创业团队介绍计划（资助号 TD2020001）的支持。
