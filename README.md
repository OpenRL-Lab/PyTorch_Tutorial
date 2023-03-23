# PyTorch_Tutorial

## 0. 环境设置

- Ubuntu or Red Hat
- Python3.8
- `pip install -r requirements.txt`

## 1. Pytorch 2.0 初探

这部分主要讲`torch.comple()`函数的使用。详细介绍见: [知乎:PyTorch 2.0初探](https://zhuanlan.zhihu.com/p/608527355)。
以下是一些测试代码，进入`./compile/`文件后通过`python xxx.py`进行运行：

- 运行测试简单函数: python [test_simple.py](./compile/test_simple.py)
- 运行测试resnet50: python [test_resnet50.py](./compile/test_resnet50.py)
- 运行Huggingface上的BERT模型: python  [test_bert.py](./compile/test_bert.py)

## Citing PyTorch_Tutorial

If you use PyTorch_Tutorial in your work, please cite us:

```bibtex
@article{tartrl2023ptt,
    title={PyTorch Tutorial},
    author={TARTRL Contributors},
    year={2023},
    howpublished={\url{https://github.com/TARTRL/PyTorch_Tutorial}},
}
```