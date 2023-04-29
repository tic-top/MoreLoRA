# MoreLoRA

##### Original LoRA:

$W = W_0 + UV$ and $rank(UV)\leq r$

##### Better Initialization:

$W = W_0 - U_0{V_0} + UV$

##### Additive LoRA:

$W = W_0 + UI_{r(1\times \frac{n}{r})}+I_{r(\frac{m}{r}\times 1)}V$ where $U\in \mathbb{R}^{m\times r}, V\in{\mathbb{R}^{r \times n}}$ and $rank(UV)\leq 2r$

##### Hadamard Mul LoRA:

$W = W_0 + \odot_{i=1}^{i=k}(\Delta_i)$ where $\Delta_i = U_iV_i$

$r'= \frac{r}{k},U_i\in\mathbb{R}^{m\times r'}$, $V_i\in\mathbb{R}^{r'\times n}$ and $rank(\odot_{i=1}^{i=k}(\Delta_i^T))\leq (\frac{r}{k})^k$

##### Hadamard Add LoRA:

$W = W_0 + \odot_{i=1}^{i=k}(\Delta_i)$ where $\Delta_i = U_iI_{r'(1\times \frac{n}{r'})}+I_{r'(\frac{m}{r'}\times 1)}V_i$

$r'= \frac{r}{k}, U_i\in\mathbb{R}^{r'\times n}, V_i\in\mathbb{R}^{m\times r'}$and $rank(\odot_{i=1}^{i=k}(\Delta_i))\leq (\frac{2r}{k})^k$

##### Hadamard LoRA: Activation

$\Delta = \odot_{i=1}^{i=k}(\tanh(U_iV_i^T))$ 


$\Delta = \odot_{i=1}^{i=k}(\sigma(U_iV_i^T)) $

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

