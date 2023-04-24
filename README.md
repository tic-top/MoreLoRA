## MoreLoRA

Author: kirp

Data: 21/4/2023

Update:  24/4/2023

### Idea:

$W = W_0 + UV^T$ and $rank(UV^T)\leq r$. 

We can do more than that.

$W = W_0 - U_0{V_0}^T + UV^T$

$W = W_0 + UI_{r(1\times \frac{n}{r})}+I_{r(\frac{m}{r}\times 1)}V^T$ where $U\in \R^{m\times r}, V\in{\R^{n \times r}}$ and $rank(UV^T)\leq 2r$. 

$W = W_0 + (U_1V_1^T)\odot(U_2V_2^T)$ where $U=[U_1, U_2]\in \R^{m\times r}, V=[V_1,V_2]\in{\R^{n \times r}}$ and $rank(UV^T)\leq r^2/4$. 

### Todo:

- Learn from PEFT, LoRA and AdaLoRA
- Derive the methods by hand
- Initialization $W = W_0 - U_0{V_0}^T + UV^T$
- Substitute multiplication by addition
- Estimate the parameter
- Learn to analysis the code
- Support Deepspeed

### Reference:

```bibtex
@online{kexuefm-9590,
    title={梯度视角下的LoRA：简介、分析、猜测及推广},
    author={苏剑林},
    year={2023},
    month={Apr},
    url={\url{https://spaces.ac.cn/archives/9590}},
}
```

```bibtex
@misc{hyeonwoo2023fedpara,
      title={FedPara: Low-Rank Hadamard Product for Communication-Efficient Federated Learning}, 
      author={Nam Hyeon-Woo and Moon Ye-Bin and Tae-Hyun Oh},
      year={2023},
      eprint={2108.06098},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

