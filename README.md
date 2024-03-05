# ⚖️ SELF-JUDGE
> [**Aligning Large Language Models by On-Policy Self-Judgment**](https://arxiv.org/abs/2402.11253),            
Sangkyu Lee<sup>1,*</sup>,
Sungdong Kim<sup>2,3,&dagger;</sup>,
Ashkan Yousefpour<sup>1</sup>, 
Minjoon Seo<sup>3</sup>, 
Kang Min Yoo<sup>2,4</sup>,
Youngjae Yu<sup>1,&dagger;</sup><br>
<sup>1</sup>Yonsei University,
<sup>2</sup>NAVER Cloud, 
<sup>3</sup>KAIST AI,
<sup>4</sup>SNU AI Center<br>
<sup>\*</sup>Work done during internship at NAVER Cloud, 
<sup>&dagger;</sup>Corresponding Author

> **Abstract:** *Existing approaches for aligning large language models with human preferences face a trade-off that requires a separate reward model (RM) for on-policy learning. In this paper, we present a novel alignment framework, SELF-JUDGE that (1) does on-policy learning and 2) is parameter efficient, as it does not require an additional RM for evaluating the samples for on-policy learning. To this end, we propose Judge-augmented Supervised Fine-Tuning (JSFT) to train a single model to act as both a policy and a judge. Specifically, we view the pairwise judgment task, choosing the better response from a response pair, as a special case of the instruction-following task. The resulting model can judge preferences of on-the-fly responses from current policy initialized from itself. Experimental results show the efficacy of SELF-JUDGE, outperforming baselines in preference benchmarks. We also show that the rejecting sampling by itself can improve performance further without an additional evaluator.*

## Usage
**Available Soon!**

## Citation
If you find this repo useful for your research, please consider citing the paper
```
@misc{lee2024aligning,
      title={Aligning Large Language Models by On-Policy Self-Judgment}, 
      author={Sangkyu Lee and Sungdong Kim and Ashkan Yousefpour and Minjoon Seo and Kang Min Yoo and Youngjae Yu},
      year={2024},
      eprint={2402.11253},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```