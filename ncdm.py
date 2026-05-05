from typing import Optional

import numpy as np
from tqdm import tqdm

from .logging_utils import get_logger

logger = get_logger()


def _get_torch_net(obj):
    if hasattr(obj, "to") and callable(getattr(obj, "to")) and hasattr(obj, "parameters"):
        return obj
    for name in ["net", "model", "_net", "ncdm_net", "nn", "network"]:
        if hasattr(obj, name):
            cand = getattr(obj, name)
            if hasattr(cand, "to") and callable(getattr(cand, "to")) and hasattr(cand, "parameters"):
                return cand
    return None


def _find_student_embedding(net, n_user: int):
    try:
        import torch
    except Exception:
        return None
    if not hasattr(net, "named_modules"):
        return None
    for _, sub in net.named_modules():
        if isinstance(sub, torch.nn.Embedding):
            w = getattr(sub, "weight", None)
            if w is None or not hasattr(w, "shape") or len(w.shape) != 2:
                continue
            if int(w.shape[0]) == int(n_user):
                return sub
    return None


def _attach_pre_rep_debias_mlp(
    net,
    n_user: int,
    user_propensity: np.ndarray,
    hidden_dim: int,
    device: str,
    mode: str = "subtract",
) -> bool:
    import torch
    import torch.nn as nn

    z = np.asarray(user_propensity, dtype=np.float32)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if z.ndim != 2:
        raise ValueError("pre_rep_propensity must be 1D or 2D array")
    if int(z.shape[0]) != int(n_user):
        raise ValueError("pre_rep_propensity first dimension must match n_user")

    target_name = None
    target_emb = None
    for name, sub in net.named_modules():
        if isinstance(sub, torch.nn.Embedding):
            w = getattr(sub, "weight", None)
            if w is None or not hasattr(w, "shape") or len(w.shape) != 2:
                continue
            if int(w.shape[0]) == int(n_user):
                target_name = name
                target_emb = sub
                break
    if target_emb is None or target_name is None:
        return False
    if target_name == "":
        return False

    parent = net
    parts = target_name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    attr_name = parts[-1]

    emb_dim = int(target_emb.weight.shape[1])
    z_dim = int(z.shape[1])
    h = max(int(hidden_dim), 1)
    rep_mlp = nn.Sequential(
        nn.Linear(z_dim, h),
        nn.ReLU(),
        nn.Linear(h, emb_dim),
    ).to(device)
    z_table = torch.as_tensor(z, dtype=torch.float32, device=device)

    mode = str(mode).lower().strip()
    if mode not in {"subtract", "gated"}:
        raise ValueError(f"Unknown pre_rep mode: {mode}")

    class _RepDebiasEmbedding(nn.Module):
        def __init__(self, base_emb: nn.Embedding, mlp: nn.Module, table: torch.Tensor, mode_name: str):
            super().__init__()
            self.base = base_emb
            self.rep_mlp = mlp
            self.register_buffer("z_table", table, persistent=False)
            self._mode = mode_name
            self._debias_scale = 1.0

        def set_debias_scale(self, value: float):
            self._debias_scale = float(max(0.0, min(1.0, value)))

        def forward(self, idx):
            base_v = self.base(idx)
            z_v = self.z_table[idx]
            delta = self.rep_mlp(z_v)
            if self._mode == "gated":
                gate = torch.sigmoid(delta)
                return base_v * (1.0 - self._debias_scale * gate)
            return base_v - self._debias_scale * delta

        def debiased_weight(self):
            delta_all = self.rep_mlp(self.z_table)
            if self._mode == "gated":
                gate_all = torch.sigmoid(delta_all)
                return self.base.weight * (1.0 - self._debias_scale * gate_all)
            return self.base.weight - self._debias_scale * delta_all

        def magnitude_regularizer(self):
            delta_all = self.rep_mlp(self.z_table)
            return (delta_all**2).sum()

    setattr(parent, attr_name, _RepDebiasEmbedding(target_emb, rep_mlp, z_table, mode))
    return True


def _set_rep_debias_scale(net, scale: float):
    if not hasattr(net, "named_modules"):
        return
    for _, sub in net.named_modules():
        if hasattr(sub, "set_debias_scale") and callable(getattr(sub, "set_debias_scale")):
            try:
                sub.set_debias_scale(scale)
            except Exception:
                continue


def train_and_eval_ncdm(
    train_u,
    train_i,
    train_k,
    train_y,
    test_u,
    test_i,
    test_k,
    test_y,
    n_user,
    n_item,
    n_skill,
    epochs=5,
    lr=1e-3,
    device="cpu",
    batch_size=1024,
    progress_desc: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
    dr_residual_coef: float = 0.0,
    dr_residual_target: Optional[np.ndarray] = None,
    dr_residual_weight: Optional[np.ndarray] = None,
    dr_residual_mask: Optional[np.ndarray] = None,
    pre_rep_debias: bool = False,
    pre_rep_propensity: Optional[np.ndarray] = None,
    pre_rep_hidden: int = 32,
    pre_rep_mag_coef: float = 0.0,
    pre_rep_mode: str = "subtract",
    pre_rep_warmup_epochs: int = 0,
    dr_residual_warmup_epochs: int = 0,
):
    import inspect

    import torch
    import torch.nn.functional as F

    # 1) import NCDM
    try:
        from EduCDM.NCDM.NCDM import NCDM
    except Exception:
        from EduCDM.NCDM import NCDM

    def build_ncdm_safely(NCDM_cls, n_user, n_item, n_skill):
        sig = inspect.signature(NCDM_cls.__init__)
        # remove self
        names = [p.name for p in sig.parameters.values() if p.name != "self"]

        # try by name first
        kw = {}
        for name in names:
            lname = name.lower()
            if any(k in lname for k in ["stu", "student", "user"]):
                kw[name] = n_user
            elif any(k in lname for k in ["exer", "item", "prob", "question", "exercise"]):
                kw[name] = n_item
            elif any(k in lname for k in ["know", "skill", "concept", "kc"]):
                kw[name] = n_skill

        # if names not enough, fallback to positional guesses
        try:
            obj = NCDM_cls(**kw)
            return obj, sig
        except TypeError:
            for args in [
                (n_user, n_item, n_skill),
                (n_item, n_user, n_skill),
                (n_user, n_skill, n_item),
                (n_item, n_skill, n_user),
                (n_skill, n_user, n_item),
                (n_skill, n_item, n_user),
            ]:
                try:
                    obj = NCDM_cls(*args)
                    return obj, sig
                except TypeError:
                    pass
            raise

    logger.info("[DEBUG] NCDM.__init__ signature: %s", inspect.signature(NCDM.__init__))
    model, _sig = build_ncdm_safely(NCDM, n_user, n_item, n_skill)

    def dump_embeddings(m):
        net = None
        for name in ["net", "model", "_net", "ncdm_net"]:
            if hasattr(m, name):
                net = getattr(m, name)
                break
        if net is None:
            net = m  # some versions expose the net directly

        logger.info("[DEBUG] Dump nn.Embedding num_embeddings:")
        for n, mod in net.named_modules():
            if isinstance(mod, torch.nn.Embedding):
                logger.info("  %s => %s x %s", n, mod.num_embeddings, mod.embedding_dim)

    dump_embeddings(model)

    # 2) find actual torch module inside model
    net = _get_torch_net(model)
    if net is None:
        raise RuntimeError(
            "找不到 NCDM 内部的 torch 网络。请打印 dir(model) 看看内部网络字段名是什么（常见 net/model/_net）。"
        )

    dev = torch.device(device)
    net.to(dev)
    if pre_rep_debias:
        if pre_rep_propensity is None:
            raise ValueError("pre_rep_debias=True requires pre_rep_propensity")
        ok = _attach_pre_rep_debias_mlp(
            net=net,
            n_user=n_user,
            user_propensity=pre_rep_propensity,
            hidden_dim=int(pre_rep_hidden),
            device=dev,
            mode=str(pre_rep_mode),
        )
        if not ok:
            logger.warning("[REP-MLP] student embedding not found; skip pre-forward debias module.")
    pre_rep_modules = [
        m for _, m in net.named_modules() if hasattr(m, "magnitude_regularizer") and callable(getattr(m, "magnitude_regularizer"))
    ]
    use_pre_rep_mag = bool(float(pre_rep_mag_coef) > 0.0 and len(pre_rep_modules) > 0)
    if float(pre_rep_mag_coef) > 0.0 and len(pre_rep_modules) == 0:
        logger.warning("[REP-MLP] pre_rep_mag_coef > 0 but no rep-debias module found; skip R_mag.")

    # 3) build tensors
    u_tr = torch.as_tensor(train_u, dtype=torch.long, device=dev)
    i_tr = torch.as_tensor(train_i, dtype=torch.long, device=dev)
    k_tr = torch.as_tensor(train_k, dtype=torch.float32, device=dev)
    y_tr = torch.as_tensor(train_y, dtype=torch.float32, device=dev).view(-1)
    if sample_weight is not None:
        w_tr = torch.as_tensor(sample_weight, dtype=torch.float32, device=dev).view(-1)
        if w_tr.numel() != y_tr.numel():
            raise ValueError("sample_weight size mismatch with train_y")
    else:
        w_tr = None
    use_dr_residual = (
        (dr_residual_coef is not None)
        and (float(dr_residual_coef) > 0.0)
        and (dr_residual_target is not None)
        and (dr_residual_weight is not None)
        and (dr_residual_mask is not None)
    )
    if use_dr_residual:
        r_tgt = torch.as_tensor(dr_residual_target, dtype=torch.float32, device=dev).view(-1)
        r_w = torch.as_tensor(dr_residual_weight, dtype=torch.float32, device=dev).view(-1)
        r_m = torch.as_tensor(dr_residual_mask, dtype=torch.float32, device=dev).view(-1)
        if (r_tgt.numel() != y_tr.numel()) or (r_w.numel() != y_tr.numel()) or (r_m.numel() != y_tr.numel()):
            raise ValueError("dr_residual arrays size mismatch with train_y")
    else:
        r_tgt, r_w, r_m = None, None, None

    u_te = torch.as_tensor(test_u, dtype=torch.long, device=dev)
    i_te = torch.as_tensor(test_i, dtype=torch.long, device=dev)
    k_te = torch.as_tensor(test_k, dtype=torch.float32, device=dev)

    # 4) minimal training loop (avoid dependency on EduCDM train signature)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    n = y_tr.shape[0]
    idx = torch.arange(n, device=dev)
    logger.info("[CHECK:k] k in [%s,%s] (n_skill=%s)", train_k.min(), train_k.max(), n_skill)
    assert train_k.min() >= 0 and train_k.max() < n_skill

    for ep in tqdm(
        range(1, epochs + 1),
        desc=f"{progress_desc} epochs" if progress_desc else "NCDM epochs",
        leave=False,
        dynamic_ncols=True,
    ):
        dr_scale = 1.0
        if int(dr_residual_warmup_epochs) > 0:
            dr_scale = min(1.0, float(ep) / float(max(int(dr_residual_warmup_epochs), 1)))
        rep_scale = 1.0
        if int(pre_rep_warmup_epochs) > 0:
            rep_scale = min(1.0, float(ep) / float(max(int(pre_rep_warmup_epochs), 1)))
        _set_rep_debias_scale(net, rep_scale)

        net.train(True)
        perm = idx[torch.randperm(n, device=dev)]
        total = 0.0

        for s in range(0, n, batch_size):
            b = perm[s : s + batch_size]
            uu, ii, kk, yy = u_tr[b], i_tr[b], k_tr[b], y_tr[b]

            opt.zero_grad(set_to_none=True)

            # compatible with different forward signatures
            try:
                yhat = net(uu, ii, kk).view(-1)
            except TypeError:
                yhat = net((uu, ii, kk)).view(-1)

            # some implementations already output sigmoid
            yhat = torch.clamp(yhat, 1e-6, 1 - 1e-6)
            if w_tr is None:
                loss_or = F.binary_cross_entropy(yhat, yy)
            else:
                ww = w_tr[b]
                loss_or = F.binary_cross_entropy(yhat, yy, reduction="none")
                loss_or = (loss_or * ww).mean()

            loss = loss_or
            if use_dr_residual:
                rr = r_tgt[b]
                rw = r_w[b]
                rm = r_m[b]
                resid = ((yhat - rr) ** 2) * rw * rm
                denom = torch.clamp(rm.sum(), min=1.0)
                loss_dr = resid.sum() / denom
                loss = loss_or + (float(dr_residual_coef) * dr_scale) * loss_dr
            if use_pre_rep_mag:
                reg_mag = None
                for mod in pre_rep_modules:
                    term = mod.magnitude_regularizer()
                    reg_mag = term if reg_mag is None else (reg_mag + term)
                if reg_mag is not None:
                    loss = loss + float(pre_rep_mag_coef) * reg_mag

            loss.backward()
            opt.step()
            total += float(loss.detach()) * int(yy.numel())

        # print(f"[NCDM][ep {ep}] train_bce={total/n:.4f}")

    # 5) predict probabilities
    _set_rep_debias_scale(net, 1.0)
    net.eval()
    outs = []
    with torch.no_grad():
        for s in tqdm(
            range(0, len(test_u), batch_size),
            desc=f"{progress_desc} infer" if progress_desc else "NCDM infer",
            leave=False,
            dynamic_ncols=True,
        ):
            uu = u_te[s : s + batch_size]
            ii = i_te[s : s + batch_size]
            kk = k_te[s : s + batch_size]
            try:
                yhat = net(uu, ii, kk).view(-1)
            except TypeError:
                yhat = net((uu, ii, kk)).view(-1)
            outs.append(yhat.detach().float().cpu().numpy())

    proba = np.concatenate(outs, axis=0).reshape(-1)
    return model, proba


def student_state_from_model(model: object, n_user: int, n_skill: int) -> Optional[np.ndarray]:
    """
    Best-effort extraction of student mastery/ability vectors.
    Different EduCDM versions store embeddings differently.
    """
    def _as_numpy(v):
        try:
            return v.detach().cpu().numpy()
        except Exception:
            try:
                return np.asarray(v)
            except Exception:
                return None

    # If a pre-forward rep-debias wrapper is used, prefer debiased student states.
    def _scan_debiased_weight(mod):
        if not hasattr(mod, "named_modules"):
            return None
        for _, sub in mod.named_modules():
            if hasattr(sub, "debiased_weight") and callable(getattr(sub, "debiased_weight")):
                try:
                    w = sub.debiased_weight()
                    if hasattr(w, "shape") and len(w.shape) == 2 and int(w.shape[0]) == int(n_user):
                        return w.detach().cpu().numpy()
                except Exception:
                    continue
        return None

    for attr in ["net", "model", "_net", "ncdm_net", "nn", "network"]:
        if hasattr(model, attr):
            cand = getattr(model, attr)
            arr = _scan_debiased_weight(cand)
            if arr is not None:
                return arr
    arr = _scan_debiased_weight(model)
    if arr is not None:
        return arr

    # Try common attributes
    for attr in ["stu_emb", "student_emb", "theta", "student", "emb_stu"]:
        if hasattr(model, attr):
            v = getattr(model, attr)
            # handle Embedding modules
            if hasattr(v, "weight"):
                try:
                    return v.weight.detach().cpu().numpy()
                except Exception:
                    pass
            arr = _as_numpy(v)
            if arr is not None and hasattr(arr, "ndim") and arr.ndim == 2:
                return arr
    # Try model.net
    if hasattr(model, "net"):
        net = getattr(model, "net")
        for attr in ["stu_emb", "student_emb", "emb_stu"]:
            if hasattr(net, attr):
                v = getattr(net, attr)
                if hasattr(v, "weight"):
                    try:
                        return v.weight.detach().cpu().numpy()
                    except Exception:
                        pass
                arr = _as_numpy(v)
                if arr is not None and hasattr(arr, "ndim") and arr.ndim == 2:
                    return arr

    # Fallback: scan embedding modules and match by n_user
    try:
        import torch
    except Exception:
        torch = None

    def _scan_for_student_embedding(mod):
        if torch is None or not hasattr(mod, "named_modules"):
            return None
        for _, sub in mod.named_modules():
            if isinstance(sub, torch.nn.Embedding):
                w = getattr(sub, "weight", None)
                if w is None or not hasattr(w, "shape") or len(w.shape) != 2:
                    continue
                if int(w.shape[0]) == int(n_user):
                    return w.detach().cpu().numpy()
        return None

    for attr in ["net", "model", "_net", "ncdm_net", "nn", "network"]:
        if hasattr(model, attr):
            cand = getattr(model, attr)
            arr = _scan_for_student_embedding(cand)
            if arr is not None:
                return arr

    arr = _scan_for_student_embedding(model)
    if arr is not None:
        return arr
    return None


def set_student_state_in_model(model: object, state: np.ndarray, n_user: int) -> bool:
    """
    Best-effort: overwrite student embedding with provided state.
    Returns True if applied successfully.
    """
    if state is None or state.ndim != 2:
        return False
    if int(state.shape[0]) != int(n_user):
        return False
    try:
        import torch
    except Exception:
        return False

    net = _get_torch_net(model)
    if net is None:
        return False
    emb = _find_student_embedding(net, n_user)
    if emb is None:
        return False
    w = getattr(emb, "weight", None)
    if w is None or w.ndim != 2:
        return False
    if int(w.shape[1]) != int(state.shape[1]):
        return False
    with torch.no_grad():
        emb.weight.copy_(torch.as_tensor(state, device=w.device, dtype=w.dtype))
    return True


def finetune_ncdm(
    model: object,
    train_u,
    train_i,
    train_k,
    train_y,
    n_user: int,
    epochs: int = 1,
    lr: float = 1e-3,
    device: str = "cpu",
    batch_size: int = 1024,
    freeze_student: bool = True,
    sample_weight: Optional[np.ndarray] = None,
):
    """
    Fine-tune an existing NCDM model in-place.
    If freeze_student is True, student embedding is frozen (requires n_user).
    """
    import torch
    import torch.nn.functional as F

    net = _get_torch_net(model)
    if net is None:
        raise RuntimeError("Could not find NCDM torch network for finetune.")
    dev = torch.device(device)
    net.to(dev)

    if freeze_student:
        emb = _find_student_embedding(net, n_user)
        if emb is not None:
            emb.weight.requires_grad_(False)

    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        return
    opt = torch.optim.Adam(params, lr=lr)

    u_tr = torch.as_tensor(train_u, dtype=torch.long, device=dev)
    i_tr = torch.as_tensor(train_i, dtype=torch.long, device=dev)
    k_tr = torch.as_tensor(train_k, dtype=torch.float32, device=dev)
    y_tr = torch.as_tensor(train_y, dtype=torch.float32, device=dev).view(-1)
    if sample_weight is not None:
        w_tr = torch.as_tensor(sample_weight, dtype=torch.float32, device=dev).view(-1)
        if w_tr.numel() != y_tr.numel():
            raise ValueError("sample_weight size mismatch with train_y")
    else:
        w_tr = None

    n = y_tr.shape[0]
    idx = torch.arange(n, device=dev)

    for _ in tqdm(
        range(1, epochs + 1),
        desc="NCDM finetune",
        leave=False,
        dynamic_ncols=True,
    ):
        net.train(True)
        perm = idx[torch.randperm(n, device=dev)]
        for s in range(0, n, batch_size):
            b = perm[s : s + batch_size]
            uu, ii, kk, yy = u_tr[b], i_tr[b], k_tr[b], y_tr[b]
            opt.zero_grad(set_to_none=True)
            try:
                yhat = net(uu, ii, kk).view(-1)
            except TypeError:
                yhat = net((uu, ii, kk)).view(-1)
            yhat = torch.clamp(yhat, 1e-6, 1 - 1e-6)
            if w_tr is None:
                loss = F.binary_cross_entropy(yhat, yy)
            else:
                ww = w_tr[b]
                loss = F.binary_cross_entropy(yhat, yy, reduction="none")
                loss = (loss * ww).mean()
            loss.backward()
            opt.step()


def predict_ncdm(
    model: object,
    test_u,
    test_i,
    test_k,
    device: str = "cpu",
    batch_size: int = 1024,
    progress_desc: Optional[str] = None,
) -> np.ndarray:
    import torch

    net = _get_torch_net(model)
    if net is None:
        raise RuntimeError("Could not find NCDM torch network for inference.")
    dev = torch.device(device)
    net.to(dev)
    net.eval()

    u_te = torch.as_tensor(test_u, dtype=torch.long, device=dev)
    i_te = torch.as_tensor(test_i, dtype=torch.long, device=dev)
    k_te = torch.as_tensor(test_k, dtype=torch.float32, device=dev)

    outs = []
    with torch.no_grad():
        for s in tqdm(
            range(0, len(test_u), batch_size),
            desc=progress_desc if progress_desc else "NCDM infer",
            leave=False,
            dynamic_ncols=True,
        ):
            uu = u_te[s : s + batch_size]
            ii = i_te[s : s + batch_size]
            kk = k_te[s : s + batch_size]
            try:
                yhat = net(uu, ii, kk).view(-1)
            except TypeError:
                yhat = net((uu, ii, kk)).view(-1)
            outs.append(yhat.detach().float().cpu().numpy())

    return np.concatenate(outs, axis=0).reshape(-1)
