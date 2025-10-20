首先看一下KL的基础公式

### KL

KL1:

大模型的KL一般是反向的：

KL(πθ||πref)=Ex∼πθ(⋅|o<t)logπθ(x|o<t)πref(x|o<t)KL(πθ||πref)=Ex∼πθ(⋅|o<t)logπθ(x|o<t)πref(x|o<t)

x∼πθ(⋅|o<t)x∼πθ(⋅|o<t) 代表 当前模型根据前t-1个token采样得到第t个token x

KL3(GRPO使用的无偏，低方差KL1估计) [http://joschu.net/blog/kl-approx.html：](https://github.com)

KL(πθ||πref)=Ex∼πθ(⋅|o<t)πrefπθ−log(πrefπθ)−1KL(πθ||πref)=Ex∼πθ(⋅|o<t)πrefπθ−log(πrefπθ)−1

* **正向KL**：倾向于使模型分布 *Q* 覆盖目标分布 *P* 的所有支持点，适合于需要模型分布更广泛覆盖的情况。
* **反向KL**：倾向于使模型分布 *Q* 集中在目标分布 *P* 的高概率区域，适合于生成任务，能够提高生成样本的质量和稳定性。

因此，在大语言模型和生成任务中，反向KL通常更受青睐。

## 不同RL算法 loss的计算

对于q的第ii个sample的第tt个token的loss: lossi,t=pg\_lossi,t+entropy\_lossi,t+kl\_lossi,tlossi,t=pg\_lossi,t+entropy\_lossi,t+kl\_lossi,t

再对一个batch中所有的token loss lossi,tlossi,t做聚合agg，得到这个batch的整体loss，可用于后续的反向传播和模型更新。

| 每个token的loss | pg\_lossi,tpg\_lossi,t | kl\_lossi,tkl\_lossi,t | loss agg mode |
| --- | --- | --- | --- |
| PPO | max(ISi,t∗−Ai,t,clip(ISi,t)∗−Ai,t)max(ISi,t∗−Ai,t,clip(ISi,t)∗−Ai,t) | rt=−D1KL(πold||πref)+rtrt=−D1KL(πold||πref)+rt | 1G∑Gi=11|oi|∑|oi|t=1lossi,t1G∑i=1G1|oi|∑t=1|oi|lossi,tseq-mean-token-mean |
| Dual-clip PPO | for A<0,min(max(ISi,t∗−Ai,t,clip(ISi,t)∗−A),clip\_c∗−A)min(max(ISi,t∗−Ai,t,clip(ISi,t)∗−A),clip\_c∗−A) | rt=−D1KL(πold||πref)+rtrt=−D1KL(πold||πref)+rt | 1G∑Gi=11|oi|∑|oi|t=1lossi,t1G∑i=1G1|oi|∑t=1|oi|lossi,tseq-mean-token-mean |
| GRPO | max(ISi,t∗−Ai,t,clip(ISi,t)∗−Ai,t)max(ISi,t∗−Ai,t,clip(ISi,t)∗−Ai,t) | β∗D3KL(πθ||πref)β∗D3KL(πθ||πref) | 1G∑Gi=11|oi|∑|oi|t=1lossi,t1G∑i=1G1|oi|∑t=1|oi|lossi,tseq-mean-token-mean |
| GSPO | ISi,t=sg[πθ(oi|q)πold(oi|q)]∗πθ(oi,t|q,oi,<t)sg[πθ(oi,t|q,oi,<t)]ISi,t=sg[πθ(oi|q)πold(oi|q)]∗πθ(oi,t|q,oi,<t)sg[πθ(oi,t|q,oi,<t)]max(ISi,t∗−Ai,t,clip(ISi,t)∗−Ai,t)max(ISi,t∗−Ai,t,clip(ISi,t)∗−Ai,t) | β∗D3KL(πθ||πref)β∗D3KL(πθ||πref) | 1G∑Gi=11|oi|∑|oi|t=1lossi,t1G∑i=1G1|oi|∑t=1|oi|lossi,tseq-mean-token-mean |
| DAPO | max(ISi,t∗−Ai,t,clip(ISi,t)∗−Ai,t)max(ISi,t∗−Ai,t,clip(ISi,t)∗−Ai,t) | β∗D3KL(πθ||πref)β∗D3KL(πθ||πref) | 1∑Gi=1|oi|∑Gi=1∑|oi|t=1lossi,t1∑i=1G|oi|∑i=1G∑t=1|oi|lossi,ttoken-mean |

### PPO

优化目标：

J=Eo∼πold1|o||o|∑i=1[min(πθ(oi|o<i,q)πold(oi|o<i,q)Ai,clip(πθ(oi|o<i,q)πold(oi|o<i,q),1−ϵ,1+ϵ)Ai]J=Eo∼πold1|o|∑i=1|o|[min(πθ(oi|o<i,q)πold(oi|o<i,q)Ai,clip(πθ(oi|o<i,q)πold(oi|o<i,q),1−ϵ,1+ϵ)Ai]

优势： GAE
递推公式，t步的累积优势=t步的优势+ t+1步的累积优势=t步及之后 每一步的优势=t步及之后所有的奖励-第t步的预计奖励

At=(rt+γVt+1−Vt)+γAt+1At=T∑i=tγi−t(rt+γVt+1−Vt)At=rt+γrt+1+γ2rt+2+...+γT−trT−VtAt=(rt+γVt+1−Vt)+γAt+1At=∑i=tTγi−t(rt+γVt+1−Vt)At=rt+γrt+1+γ2rt+2+...+γT−trT−Vt

奖励：

rt={−KL(πold||πref),t≠T−KL(πold||πref)+RM(q,oi),t=Trt={−KL(πold||πref),t≠T−KL(πold||πref)+RM(q,oi),t=T

`verl/trainer/ppo/ray_trainer.py` [verl](https://github.com):[westworld加速](https://westworldjs.com) ｜ 如何在奖励中添加KL惩罚项？

```
###################################################
# 将KL惩罚loss应用到reward中。原始的reward是[0, 0, 0, ..., RM(q,o_i)]
# return KL(\pi_old||\pi_{ref}) + reward
###################################################
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld
```

KL

KL(πold||πref)=log(πold(ot|q,o<t)πref(ot|q,o<t))KL(πold||πref)=log(πold(ot|q,o<t)πref(ot|q,o<t))

PPO的KL散度是old到ref的

PPO的代码实现详见下面的Dual-clip PPO（PPO的改进版）

### Dual-clip PPO

[https://arxiv.org/pdf/1912.09729：对A](https://github.com)<0的token的重要性采样IS做clip

![image-20251020144504938](https://p.ipic.vip/awpfmq.png)

论文发现当A<0时，重要性采样的比值\*A可以**是负无穷**，这会导致训练不稳定（梯度爆炸）的现象，因此在ppo的clip上，对于A<0又进一步添加了新的clip (clip\_ratio\_c)。

per token objection={min(IS∗A,clip(IS,1−ϵ,1+ϵ)∗A),A≥0max(min(IS∗A,clip(IS,1−ϵ,1+ϵ)∗A),clip\_ratio\_c∗A),A<0per token objection={min(IS∗A,clip(IS,1−ϵ,1+ϵ)∗A),A≥0max(min(IS∗A,clip(IS,1−ϵ,1+ϵ)∗A),clip\_ratio\_c∗A),A<0

代码：

整体的ppo\_loss是由pg\_loss + kl\_loss + entropy\_loss构成，不同的RL方法pg\_loss, kl\_loss的计算方法是不同的。

* pg\_loss：具体于`verl/trainer/ppo/core_algos.py`（我将在dual-clip ppo和gspo部分介绍对应的pg\_loss代码）。
* kl\_loss：同样位于`verl/trainer/ppo/core_algos.py`（我将会在grpo部分介绍具体的low\_var\_kl代码）。

`verl/verl/workers/roles/utils/losses.py`: ppo\_loss的计算

```
######################################################
# 此函数用于计算整体的actor loss
######################################################
def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    log_prob = model_output["log_probs"]
    entropy = model_output.get("entropy", None)

    log_prob = no_padding_2_padding(log_prob, data)  # (bsz, response_length)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)  # (bsz, response_length)

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    # 调用下面的计算pg_loss的代码框
    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
    )

    metrics.update(
        {
            "pg_loss": pg_loss.detach().item(),
            "pg_clipfrac": pg_clipfrac.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        }
    )
    policy_loss = pg_loss

    # 是否使用entropy loss
    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
        entropy_coeff = config.entropy_coeff
        # token的entropy越大越好，而loss是越小越好，因此是 减去 entropy
        policy_loss -= entropy_coeff * entropy_loss

    # 是否使用KL loss（grpo/gspo使用，ppo/dapo不使用）
    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode)

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = kl_loss.detach().item()
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics
```

`verl/trainer/ppo/core_algos.py`不同的RL方法计算pg\_loss是不同的，这里的是ppo的pg\_loss，后面还会介绍gspo的pg\_loss的实现。

```
######################################################
# 此函数用于计算pg_loss，并不计算KL惩罚项
######################################################
@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    # 计算每一个token的重要性采样的比值的log
    # log(\pi_{\theta}(o_{i,t}|q,o_{i,
    negative_approx_kl = log_prob - old_log_prob
    # 对IS的log做clip，避免过大或过小
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    # 这里ratio是真正的IS 重要性采样
    ratio = torch.exp(negative_approx_kl)
    # 计算出-IS在token-level上的均值
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    ######################################################
    # 下面开始计算pg_loss=
    #A>0, max(ratio*-A, clip(ratio, 1-\epsilon_low, 1+\epsilon_high)*-A)
    #A<0, min(max(ratio*-A, clip(ratio, 1-\epsilon_low, 1+\epsilon_high)*-A), clip_ratio_c*-A)
    ######################################################
    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    # clip后的loss
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    # ppo per token loss
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    # 计算被才剪掉的token在 这个batch的所有未mask的token的比例（axis=None）【常数】
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    # 这里是dual-clip PPO提出，使用clip_ratio_c限制A<0的token的loss
    pg_losses3 = -advantages * clip_ratio_c
    # min(max(ratio*-A, clip(ratio, 1-\epsilon_low, 1+\epsilon_high)*-A), clip_ratio_c*-A)
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    # 记录在传统ppo下，进一步裁减的A<0的IS大于clip_ratio_c的token在 这个batch的所有未mask的token的比例【常数】
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    # pg_losses是分段函数(记录每个token的loss)，A<0时用clip_pg_losses2, A>=0时用clip_pg_losses1
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    # pg_losses: (bsz, response_length)
    # 如何计算一整个batch的所有token的整体loss。这有多种方式，主要看配置的loss_agg_mode
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
```

咱们继续看几种token loss的agg mode。不同RL方法，loss agg mode也是不同的

`verl/trainer/ppo/core_algos.py`

```
def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss
```

### GRPO

优化目标：

J=E{oi}Gi=1∼πold(⋅|q)1|G||G|∑i=11|o||oi|∑t=1{min[πθ(oi,t|q,oi,<t)πold(oi,t|q,oi,<t)Ai,t,clip(πθ(oi,t|q,oi,<t)πold(oi,t|q,oi,<t),1−ϵ,1+ϵ)Ai,t]−βDKL(πθ||πref)}J=E{oi}i=1G∼πold(⋅|q)1|G|∑i=1|G|1|o|∑t=1|oi|{min[πθ(oi,t|q,oi,<t)πold(oi,t|q,oi,<t)Ai,t,clip(πθ(oi,t|q,oi,<t)πold(oi,t|q,oi,<t),1−ϵ,1+ϵ)Ai,t]−βDKL(πθ||πref)}

优势：

Ai,t=ri−mean(r)std(r)Ai,t=ri−mean(r)std(r)

KL3

DKL(πθ||πref)=πref(oi,t|q,oi,<t)πθ(oi,t|q,oi,<t)−log(πref(oi,t|q,oi,<t)πθ(oi,t|q,oi,<t))−1DKL(πθ||πref)=πref(oi,t|q,oi,<t)πθ(oi,t|q,oi,<t)−log(πref(oi,t|q,oi,<t)πθ(oi,t|q,oi,<t))−1

KL3的方差比KL1小，且是KL1的无偏估计

证明

D3KL(P||Q)=∑x∼PP(x)[Q(x)P(x)−log(P(x)Q(x))−1]=∑x∼PQ(x)+P(x)log(P(x)Q(x))−P(x)=∑x∼PQ(x)−∑x∼PP(x)+D1KL(P||Q)=D1KL(P||Q)+∑x∼PQ(x)−1         当P所有采样在Q中的概率和为1时（vocab一样的话）=D1KL(P||Q)D3KL(P||Q)=∑x∼PP(x)[Q(x)P(x)−log(P(x)Q(x))−1]=∑x∼PQ(x)+P(x)log(P(x)Q(x))−P(x)=∑x∼PQ(x)−∑x∼PP(x)+D1KL(P||Q)=D1KL(P||Q)+∑x∼PQ(x)−1         当P所有采样在Q中的概率和为1时（vocab一样的话）=D1KL(P||Q)

`verl/trainer/ppo/core_algos.py` 下面是verl对kl\_loss的实现：

```
def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    ##############################################################
    # 这里的low_var_kl与上述的grpo的KL计算公式相同
    ##############################################################
    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
```

### GSPO

seq-level 优化目标：

J=E{oi}Gi=1∼πold(⋅|q)1|G||G|∑i=1min[(πθ(oi|q)πold(oi|q))1|oi|Ai,clip((πθ(oi|q)πold(oi|q))1|oi|,1−ϵ,1+ϵ)Ai]J=E{oi}i=1G∼πold(⋅|q)1|G|∑i=1|G|min[(πθ(oi|q)πold(oi|q))1|oi|Ai,clip((πθ(oi|q)πold(oi|q))1|oi|,1−ϵ,1+ϵ)Ai]

πθ(oi|q)πold(oi|q)=Π|oi|t=1πθ(oi,t|q,oi,<t)Π|oi|t=1πold(oi,t|q,oi,<t)πθ(oi|q)πold(oi|q)=Πt=1|oi|πθ(oi,t|q,oi,<t)Πt=1|oi|πold(oi,t|q,oi,<t)

token-level 优化目标：

J=E{oi}Gi=1∼πold(⋅|q)1GG∑i=11|oi||oi|∑t=1min(si,tAi,t,clip(si,t,1−ϵ,1+ϵ)Ai,t)^si,t=sg[(πθ(oi|q)πold(oi|q))1|oi|]∗πθ(oi,t|q,oi,<t)sg[πθ(oi,t|q,oi,<t)]J=E{oi}i=1G∼πold(⋅|q)1G∑i=1G1|oi|∑t=1|oi|min(si,tAi,t,clip(si,t,1−ϵ,1+ϵ)Ai,t)s^i,t=sg[(πθ(oi|q)πold(oi|q))1|oi|]∗πθ(oi,t|q,oi,<t)sg[πθ(oi,t|q,oi,<t)]

可以发现的是 sg[si,t]=sg[si],si=(πθ(oi|q)πold(oi|q))1|oi|sg[si,t]=sg[si],si=(πθ(oi|q)πold(oi|q))1|oi|，但是在方向上不同

通过证明，可以发现，当Ai,t=AiAi,t=Ai时，seq-level和token-level在前向传播和反向传播上是一样的
token-level 可以更好地扩展 同sample不同token的A的灵活度（每个token的A可以不相同）

`verl/trainer/ppo/core_algos.py`

```
##########################################################
# 计算gspo的pg_loss,重点关注IS的计算
##########################################################
@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
```

### DAPO

优化目标：

J=E(q,a)∼D,{oi}Gi=1∼πold(⋅|q)[1∑Gi=1|oi|G∑i=1|oi|∑t=1min(ri,t(θ)Ai,t,clip(ri,t(θ),1−ϵlow,1+ϵhigh)Ai,t)]s.t. 0<|{oi|is\_equivalent(oi,a)}|<GJ=E(q,a)∼D,{oi}i=1G∼πold(⋅|q)[1∑i=1G|oi|∑i=1G∑t=1|oi|min(ri,t(θ)Ai,t,clip(ri,t(θ),1−ϵlow,1+ϵhigh)Ai,t)]s.t. 0<|{oi|is\_equivalent(oi,a)}|<G

其中

ri,t(θ)=πθ(oi,t|q,oi,<t)πold(oi,t|q,oi,<t),Ai,t=Ri−mean({Ri}Gi=1)std({Ri}Gi=1)ri,t(θ)=πθ(oi,t|q,oi,<t)πold(oi,t|q,oi,<t),Ai,t=Ri−mean({Ri}i=1G)std({Ri}i=1G)

其loss agg mode是token-mean。
