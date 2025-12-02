import torch


class AWP:
    """Adversarial Weight Perturbation

    AWP adds small perturbations to model weights during training to improve
    model robustness and generalization. It works by:
    1. Computing adversarial perturbations that maximize loss
    2. Temporarily applying these perturbations
    3. Computing gradients on perturbed weights
    4. Restoring original weights before optimizer step

    Reference: https://arxiv.org/abs/2004.05884
    """

    def __init__(self, model, optimizer, adv_lr=1.0, adv_eps=0.001, start_epoch=0, adv_param='weight'):
        """
        Args:
            model: The model to apply AWP to
            optimizer: The optimizer used for training
            adv_lr: Learning rate for computing adversarial perturbations
            adv_eps: Maximum perturbation magnitude (L2 norm)
            start_epoch: Epoch to start applying AWP (allows warmup without AWP)
            adv_param: Which parameters to perturb ('weight' or 'all')
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_param = adv_param
        self.backup = {}
        self.backup_eps = {}

    # 1. ATTACK: Save weights, add noise
    def attack(self):
        self._save()
        self._attack_step()

    # 2. RESTORE: Remove noise, put original weights back
    def restore(self):
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # calculate perturbation
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
        self.backup_eps = {}

    def _should_perturb(self, name):
        if self.adv_param == 'weight':
            return 'weight' in name and 'LayerNorm' not in name
        return True
