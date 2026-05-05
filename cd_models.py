from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .logging_utils import get_logger
from .ncdm import train_and_eval_ncdm
from .ncdm import finetune_ncdm, predict_ncdm, set_student_state_in_model

logger = get_logger()


def _build_sequence_map(
    user_ids,
    item_ids,
    labels,
    sample_weight: Optional[np.ndarray] = None,
    dr_residual_target: Optional[np.ndarray] = None,
    dr_residual_weight: Optional[np.ndarray] = None,
    dr_residual_mask: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    u = np.asarray(user_ids).astype(np.int64).reshape(-1)
    i = np.asarray(item_ids).astype(np.int64).reshape(-1)
    y = np.asarray(labels).astype(np.float32).reshape(-1)
    n = int(y.shape[0])
    if int(u.shape[0]) != n or int(i.shape[0]) != n:
        raise ValueError("user/item/label size mismatch")

    if sample_weight is not None:
        sw = np.asarray(sample_weight).astype(np.float32).reshape(-1)
        if int(sw.shape[0]) != n:
            raise ValueError("sample_weight size mismatch")
    else:
        sw = None

    if dr_residual_target is not None:
        dr_tgt = np.asarray(dr_residual_target).astype(np.float32).reshape(-1)
        if int(dr_tgt.shape[0]) != n:
            raise ValueError("dr_residual_target size mismatch")
    else:
        dr_tgt = None

    if dr_residual_weight is not None:
        dr_w = np.asarray(dr_residual_weight).astype(np.float32).reshape(-1)
        if int(dr_w.shape[0]) != n:
            raise ValueError("dr_residual_weight size mismatch")
    else:
        dr_w = None

    if dr_residual_mask is not None:
        dr_m = np.asarray(dr_residual_mask).astype(np.float32).reshape(-1)
        if int(dr_m.shape[0]) != n:
            raise ValueError("dr_residual_mask size mismatch")
    else:
        dr_m = None

    seq_map: Dict[int, Dict[str, List[float]]] = {}
    for idx in range(n):
        uid = int(u[idx])
        rec = seq_map.get(uid)
        if rec is None:
            rec = {
                "items": [],
                "y": [],
                "sample_weight": [],
                "dr_tgt": [],
                "dr_w": [],
                "dr_m": [],
            }
            seq_map[uid] = rec
        rec["items"].append(int(i[idx]))
        rec["y"].append(float(y[idx]))
        rec["sample_weight"].append(float(sw[idx]) if sw is not None else 1.0)
        rec["dr_tgt"].append(float(dr_tgt[idx]) if dr_tgt is not None else 0.0)
        rec["dr_w"].append(float(dr_w[idx]) if dr_w is not None else 0.0)
        rec["dr_m"].append(float(dr_m[idx]) if dr_m is not None else 0.0)

    out: Dict[int, Dict[str, np.ndarray]] = {}
    for uid, rec in seq_map.items():
        out[uid] = {
            "items": np.asarray(rec["items"], dtype=np.int64),
            "y": np.asarray(rec["y"], dtype=np.float32),
            "sample_weight": np.asarray(rec["sample_weight"], dtype=np.float32),
            "dr_tgt": np.asarray(rec["dr_tgt"], dtype=np.float32),
            "dr_w": np.asarray(rec["dr_w"], dtype=np.float32),
            "dr_m": np.asarray(rec["dr_m"], dtype=np.float32),
        }
    return out


def _train_dkt_core(
    model,
    train_seq_map: Dict[int, Dict[str, np.ndarray]],
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
    batch_size: int = 1024,
    progress_desc: Optional[str] = None,
    dr_residual_coef: float = 0.0,
    use_dr_residual: bool = False,
    pre_rep_mag_coef: float = 0.0,
    pre_rep_warmup_epochs: int = 0,
    dr_residual_warmup_epochs: int = 0,
):
    import torch
    import torch.nn.functional as F

    dev = torch.device(device)
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return
    opt = torch.optim.Adam(params, lr=lr)

    user_ids = np.asarray(list(train_seq_map.keys()), dtype=np.int64)
    if user_ids.size == 0:
        return

    user_batch_size = max(8, min(128, max(1, int(batch_size) // 8)))

    pre_rep_modules = [
        m for _, m in model.named_modules() if hasattr(m, "magnitude_regularizer") and callable(getattr(m, "magnitude_regularizer"))
    ]
    use_pre_rep_mag = bool(float(pre_rep_mag_coef) > 0.0 and len(pre_rep_modules) > 0)
    if float(pre_rep_mag_coef) > 0.0 and len(pre_rep_modules) == 0:
        logger.warning("[REP-MLP] pre_rep_mag_coef > 0 but no rep-debias module found; skip R_mag.")

    for ep in tqdm(
        range(1, epochs + 1),
        desc=f"{progress_desc} epochs" if progress_desc else "DKT epochs",
        leave=False,
        dynamic_ncols=True,
    ):
        dr_scale = 1.0
        if int(dr_residual_warmup_epochs) > 0:
            dr_scale = min(1.0, float(ep) / float(max(int(dr_residual_warmup_epochs), 1)))
        rep_scale = 1.0
        if int(pre_rep_warmup_epochs) > 0:
            rep_scale = min(1.0, float(ep) / float(max(int(pre_rep_warmup_epochs), 1)))
        _set_rep_debias_scale(model, rep_scale)

        model.train(True)
        perm = np.random.permutation(user_ids.shape[0])
        for s in range(0, user_ids.shape[0], user_batch_size):
            idxs = perm[s : s + user_batch_size]
            batch_users = user_ids[idxs]
            seqs = [train_seq_map[int(uid)] for uid in batch_users]
            lens = [int(seq["y"].shape[0]) for seq in seqs]
            if not lens:
                continue
            max_len = int(max(lens))
            bsz = int(len(seqs))
            if max_len <= 0 or bsz <= 0:
                continue

            items_pad = torch.zeros((bsz, max_len), dtype=torch.long, device=dev)
            y_pad = torch.zeros((bsz, max_len), dtype=torch.float32, device=dev)
            sw_pad = torch.ones((bsz, max_len), dtype=torch.float32, device=dev)
            dr_tgt_pad = torch.zeros((bsz, max_len), dtype=torch.float32, device=dev)
            dr_w_pad = torch.zeros((bsz, max_len), dtype=torch.float32, device=dev)
            dr_m_pad = torch.zeros((bsz, max_len), dtype=torch.float32, device=dev)
            mask_pad = torch.zeros((bsz, max_len), dtype=torch.bool, device=dev)

            for b, seq in enumerate(seqs):
                l = int(lens[b])
                if l <= 0:
                    continue
                items_pad[b, :l] = torch.as_tensor(seq["items"], dtype=torch.long, device=dev)
                y_pad[b, :l] = torch.as_tensor(seq["y"], dtype=torch.float32, device=dev)
                sw_pad[b, :l] = torch.as_tensor(seq["sample_weight"], dtype=torch.float32, device=dev)
                dr_tgt_pad[b, :l] = torch.as_tensor(seq["dr_tgt"], dtype=torch.float32, device=dev)
                dr_w_pad[b, :l] = torch.as_tensor(seq["dr_w"], dtype=torch.float32, device=dev)
                dr_m_pad[b, :l] = torch.as_tensor(seq["dr_m"], dtype=torch.float32, device=dev)
                mask_pad[b, :l] = True

            uu = torch.as_tensor(batch_users, dtype=torch.long, device=dev)
            h = model.student_emb(uu)

            opt.zero_grad(set_to_none=True)
            loss_or_sum = torch.zeros((), dtype=torch.float32, device=dev)
            loss_dr_sum = torch.zeros((), dtype=torch.float32, device=dev)
            n_or = 0.0
            n_dr = torch.zeros((), dtype=torch.float32, device=dev)

            for t in range(max_len):
                mt = mask_pad[:, t]
                if not bool(mt.any()):
                    continue
                idx_t = mt.nonzero(as_tuple=False).view(-1)
                items_t = items_pad[idx_t, t]
                y_t = y_pad[idx_t, t]
                sw_t = sw_pad[idx_t, t]
                h_t = h.index_select(0, idx_t)
                p_t = model.predict_from_hidden(h_t, items_t)
                bce = F.binary_cross_entropy(torch.clamp(p_t, 1e-6, 1 - 1e-6), y_t, reduction="none")
                loss_or_sum = loss_or_sum + torch.sum(bce * sw_t)
                n_or += float(idx_t.numel())

                if use_dr_residual:
                    rr = dr_tgt_pad[idx_t, t]
                    rw = dr_w_pad[idx_t, t]
                    rm = dr_m_pad[idx_t, t]
                    resid = ((p_t - rr) ** 2) * rw * rm
                    loss_dr_sum = loss_dr_sum + torch.sum(resid)
                    n_dr = n_dr + torch.sum(rm)

                h_new = model.update_hidden(h_t, items_t, y_t)
                h = h.index_copy(0, idx_t, h_new)

            if n_or <= 0:
                continue
            loss_or = loss_or_sum / float(max(n_or, 1.0))
            loss = loss_or
            if use_dr_residual:
                denom = torch.clamp(n_dr, min=1.0)
                loss_dr = loss_dr_sum / denom
                loss = loss + (float(dr_residual_coef) * dr_scale) * loss_dr
            if use_pre_rep_mag:
                reg_mag = None
                for mod in pre_rep_modules:
                    term = mod.magnitude_regularizer()
                    reg_mag = term if reg_mag is None else (reg_mag + term)
                if reg_mag is not None:
                    loss = loss + float(pre_rep_mag_coef) * reg_mag

            loss.backward()
            opt.step()

    _set_rep_debias_scale(model, 1.0)


def _warm_start_hidden_from_train(model, train_seq_map: Dict[int, Dict[str, np.ndarray]], users: np.ndarray, device: str):
    import torch

    dev = torch.device(device)
    user_tensor = torch.as_tensor(users, dtype=torch.long, device=dev)
    h_all = model.student_emb(user_tensor)
    h_map = {int(users[idx]): h_all[idx : idx + 1] for idx in range(int(users.shape[0]))}
    with torch.no_grad():
        for uid in users:
            seq = train_seq_map.get(int(uid))
            if seq is None or int(seq["items"].shape[0]) <= 0:
                continue
            h = h_map[int(uid)]
            items = torch.as_tensor(seq["items"], dtype=torch.long, device=dev).view(-1)
            y = torch.as_tensor(seq["y"], dtype=torch.float32, device=dev).view(-1)
            for t in range(int(items.shape[0])):
                it = items[t : t + 1]
                yt = y[t : t + 1]
                h = model.update_hidden(h, it, yt)
            h_map[int(uid)] = h
    return h_map


def _build_model_safely(model_cls, n_user: int, n_item: int, n_skill: Optional[int]):
    import inspect

    sig = inspect.signature(model_cls.__init__)
    names = [p.name for p in sig.parameters.values() if p.name != "self"]
    kw = {}
    for name in names:
        lname = name.lower()
        if any(k in lname for k in ["stu", "student", "user"]):
            kw[name] = n_user
        elif any(k in lname for k in ["exer", "item", "prob", "question", "exercise"]):
            kw[name] = n_item
        elif n_skill is not None and any(k in lname for k in ["know", "skill", "concept", "kc"]):
            kw[name] = n_skill

    try:
        obj = model_cls(**kw)
        return obj
    except TypeError:
        args_list = []
        if n_skill is not None:
            args_list = [
                (n_user, n_item, n_skill),
                (n_item, n_user, n_skill),
                (n_user, n_skill, n_item),
                (n_item, n_skill, n_user),
                (n_skill, n_user, n_item),
                (n_skill, n_item, n_user),
            ]
        else:
            args_list = [(n_user, n_item), (n_item, n_user)]
        for args in args_list:
            try:
                obj = model_cls(*args)
                return obj
            except TypeError:
                pass
        raise


def _get_torch_net(obj):
    if hasattr(obj, "to") and callable(getattr(obj, "to")) and hasattr(obj, "parameters"):
        return obj
    for name in ["net", "model", "_net", "ncdm_net", "nn", "network"]:
        if hasattr(obj, name):
            cand = getattr(obj, name)
            if hasattr(cand, "to") and callable(getattr(cand, "to")) and hasattr(cand, "parameters"):
                return cand
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


def _forward(net, uu, ii, kk):
    try:
        return net(uu, ii, kk)
    except Exception:
        try:
            return net((uu, ii, kk))
        except Exception:
            if kk is not None and kk.ndim == 2:
                import torch

                kk_idx = torch.argmax(kk, dim=1)
                try:
                    return net(uu, ii, kk_idx)
                except Exception:
                    return net((uu, ii, kk_idx))
            raise


def train_and_eval_kancd(
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
    try:
        from EduCDM.KaNCD.KaNCD import KaNCD
    except Exception:
        try:
            from EduCDM.KaNCD import KaNCD
        except Exception as exc:
            raise RuntimeError("KaNCD not available in EduCDM.") from exc

    import torch
    import torch.nn.functional as F

    # KaNCD expects kwargs: exer_n, student_n, knowledge_n, dim, (mf_type)
    try:
        model = KaNCD(exer_n=n_item, student_n=n_user, knowledge_n=n_skill, dim=16, mf_type="gmf")
    except Exception:
        model = _build_model_safely(KaNCD, n_user, n_item, n_skill)
    net = _get_torch_net(model)
    if net is None:
        raise RuntimeError("Could not find KaNCD torch network inside model.")

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
            logger.warning("[REP-MLP] KaNCD student embedding not found; skip pre-forward debias module.")
    pre_rep_modules = [
        m for _, m in net.named_modules() if hasattr(m, "magnitude_regularizer") and callable(getattr(m, "magnitude_regularizer"))
    ]
    use_pre_rep_mag = bool(float(pre_rep_mag_coef) > 0.0 and len(pre_rep_modules) > 0)
    if float(pre_rep_mag_coef) > 0.0 and len(pre_rep_modules) == 0:
        logger.warning("[REP-MLP] pre_rep_mag_coef > 0 but no rep-debias module found; skip R_mag.")

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

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = y_tr.shape[0]
    idx = torch.arange(n, device=dev)

    for _ in tqdm(
        range(1, epochs + 1),
        desc=f"{progress_desc} epochs" if progress_desc else "KaNCD epochs",
        leave=False,
        dynamic_ncols=True,
    ):
        ep = _
        dr_scale = 1.0
        if int(dr_residual_warmup_epochs) > 0:
            dr_scale = min(1.0, float(ep) / float(max(int(dr_residual_warmup_epochs), 1)))
        rep_scale = 1.0
        if int(pre_rep_warmup_epochs) > 0:
            rep_scale = min(1.0, float(ep) / float(max(int(pre_rep_warmup_epochs), 1)))
        _set_rep_debias_scale(net, rep_scale)

        net.train(True)
        perm = idx[torch.randperm(n, device=dev)]
        for s in range(0, n, batch_size):
            b = perm[s : s + batch_size]
            uu, ii, kk, yy = u_tr[b], i_tr[b], k_tr[b], y_tr[b]
            opt.zero_grad(set_to_none=True)
            yhat = _forward(net, uu, ii, kk).view(-1)
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

    _set_rep_debias_scale(net, 1.0)
    net.eval()
    outs = []
    with torch.no_grad():
        for s in tqdm(
            range(0, len(test_u), batch_size),
            desc=f"{progress_desc} infer" if progress_desc else "KaNCD infer",
            leave=False,
            dynamic_ncols=True,
        ):
            uu = u_te[s : s + batch_size]
            ii = i_te[s : s + batch_size]
            kk = k_te[s : s + batch_size]
            yhat = _forward(net, uu, ii, kk).view(-1)
            outs.append(yhat.detach().float().cpu().numpy())

    proba = np.concatenate(outs, axis=0).reshape(-1)
    return model, proba


def train_and_eval_irt(
    train_u,
    train_i,
    train_y,
    test_u,
    test_i,
    n_user,
    n_item,
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
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SimpleIRT(nn.Module):
        def __init__(self, n_user, n_item):
            super().__init__()
            self.theta = nn.Embedding(n_user, 1)
            self.beta = nn.Embedding(n_item, 1)

        def forward(self, u, i):
            t = self.theta(u).view(-1)
            b = self.beta(i).view(-1)
            return torch.sigmoid(t - b)

    dev = torch.device(device)
    model = SimpleIRT(n_user, n_item).to(dev)
    if pre_rep_debias:
        if pre_rep_propensity is None:
            raise ValueError("pre_rep_debias=True requires pre_rep_propensity")
        ok = _attach_pre_rep_debias_mlp(
            net=model,
            n_user=n_user,
            user_propensity=pre_rep_propensity,
            hidden_dim=int(pre_rep_hidden),
            device=dev,
            mode=str(pre_rep_mode),
        )
        if not ok:
            logger.warning("[REP-MLP] IRT student embedding not found; skip pre-forward debias module.")
    pre_rep_modules = [
        m for _, m in model.named_modules() if hasattr(m, "magnitude_regularizer") and callable(getattr(m, "magnitude_regularizer"))
    ]
    use_pre_rep_mag = bool(float(pre_rep_mag_coef) > 0.0 and len(pre_rep_modules) > 0)
    if float(pre_rep_mag_coef) > 0.0 and len(pre_rep_modules) == 0:
        logger.warning("[REP-MLP] pre_rep_mag_coef > 0 but no rep-debias module found; skip R_mag.")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    u_tr = torch.as_tensor(train_u, dtype=torch.long, device=dev)
    i_tr = torch.as_tensor(train_i, dtype=torch.long, device=dev)
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

    n = y_tr.shape[0]
    idx = torch.arange(n, device=dev)

    for _ in tqdm(
        range(1, epochs + 1),
        desc=f"{progress_desc} epochs" if progress_desc else "IRT epochs",
        leave=False,
        dynamic_ncols=True,
    ):
        ep = _
        dr_scale = 1.0
        if int(dr_residual_warmup_epochs) > 0:
            dr_scale = min(1.0, float(ep) / float(max(int(dr_residual_warmup_epochs), 1)))
        rep_scale = 1.0
        if int(pre_rep_warmup_epochs) > 0:
            rep_scale = min(1.0, float(ep) / float(max(int(pre_rep_warmup_epochs), 1)))
        _set_rep_debias_scale(model, rep_scale)

        model.train(True)
        perm = idx[torch.randperm(n, device=dev)]
        for s in range(0, n, batch_size):
            b = perm[s : s + batch_size]
            uu, ii, yy = u_tr[b], i_tr[b], y_tr[b]
            opt.zero_grad(set_to_none=True)
            yhat = model(uu, ii)
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

    _set_rep_debias_scale(model, 1.0)
    model.eval()
    outs = []
    with torch.no_grad():
        for s in tqdm(
            range(0, len(test_u), batch_size),
            desc=f"{progress_desc} infer" if progress_desc else "IRT infer",
            leave=False,
            dynamic_ncols=True,
        ):
            uu = u_te[s : s + batch_size]
            ii = i_te[s : s + batch_size]
            yhat = model(uu, ii).view(-1)
            outs.append(yhat.detach().float().cpu().numpy())

    proba = np.concatenate(outs, axis=0).reshape(-1)
    return model, proba


def train_and_eval_dkt(
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
    import torch
    import torch.nn as nn

    class SimpleDKT(nn.Module):
        def __init__(self, n_user: int, n_item: int, hidden_dim: int = 64):
            super().__init__()
            self.n_item = int(n_item)
            self.student_emb = nn.Embedding(int(n_user), int(hidden_dim))
            self.item_emb = nn.Embedding(int(n_item), int(hidden_dim))
            self.inter_emb = nn.Embedding(int(n_item) * 2, int(hidden_dim))
            self.gru_cell = nn.GRUCell(int(hidden_dim), int(hidden_dim))
            self.pred_head = nn.Sequential(
                nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), 1),
            )

        def predict_from_hidden(self, h, item_ids):
            item_v = self.item_emb(item_ids)
            return torch.sigmoid(self.pred_head(torch.cat([h, item_v], dim=1)).view(-1))

        def update_hidden(self, h, item_ids, y_soft):
            y_soft = y_soft.view(-1, 1).float()
            emb_w = self.inter_emb(item_ids)
            emb_c = self.inter_emb(item_ids + self.n_item)
            x = (1.0 - y_soft) * emb_w + y_soft * emb_c
            return self.gru_cell(x, h)

    hidden_dim = max(32, min(128, int(n_skill) * 2))
    model = SimpleDKT(n_user=int(n_user), n_item=int(n_item), hidden_dim=int(hidden_dim))
    dev = torch.device(device)
    model.to(dev)

    if pre_rep_debias:
        if pre_rep_propensity is None:
            raise ValueError("pre_rep_debias=True requires pre_rep_propensity")
        ok = _attach_pre_rep_debias_mlp(
            net=model,
            n_user=n_user,
            user_propensity=pre_rep_propensity,
            hidden_dim=int(pre_rep_hidden),
            device=dev,
            mode=str(pre_rep_mode),
        )
        if not ok:
            logger.warning("[REP-MLP] DKT student embedding not found; skip pre-forward debias module.")

    use_dr_residual = (
        (dr_residual_coef is not None)
        and (float(dr_residual_coef) > 0.0)
        and (dr_residual_target is not None)
        and (dr_residual_weight is not None)
        and (dr_residual_mask is not None)
    )
    train_seq_map = _build_sequence_map(
        user_ids=train_u,
        item_ids=train_i,
        labels=train_y,
        sample_weight=sample_weight,
        dr_residual_target=dr_residual_target if use_dr_residual else None,
        dr_residual_weight=dr_residual_weight if use_dr_residual else None,
        dr_residual_mask=dr_residual_mask if use_dr_residual else None,
    )
    _train_dkt_core(
        model=model,
        train_seq_map=train_seq_map,
        epochs=int(epochs),
        lr=float(lr),
        device=device,
        batch_size=int(batch_size),
        progress_desc=progress_desc,
        dr_residual_coef=float(dr_residual_coef),
        use_dr_residual=bool(use_dr_residual),
        pre_rep_mag_coef=float(pre_rep_mag_coef),
        pre_rep_warmup_epochs=int(pre_rep_warmup_epochs),
        dr_residual_warmup_epochs=int(dr_residual_warmup_epochs),
    )

    # cache train histories for sequential warm-start in inference
    hist: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for uid, seq in train_seq_map.items():
        hist[int(uid)] = (seq["items"].copy(), seq["y"].copy())
    setattr(model, "_dkt_train_history", hist)

    proba = predict_dkt(
        model=model,
        test_u=test_u,
        test_i=test_i,
        test_k=test_k,
        device=device,
        batch_size=batch_size,
        progress_desc=f"{progress_desc} infer" if progress_desc else "DKT infer",
    )
    return model, proba


def train_and_eval_cd(
    model_name: str,
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
    name = model_name.lower()
    if name == "ncdm":
        return train_and_eval_ncdm(
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
            epochs=epochs,
            lr=lr,
            device=device,
            batch_size=batch_size,
            progress_desc=progress_desc,
            sample_weight=sample_weight,
            dr_residual_coef=dr_residual_coef,
            dr_residual_target=dr_residual_target,
            dr_residual_weight=dr_residual_weight,
            dr_residual_mask=dr_residual_mask,
            pre_rep_debias=pre_rep_debias,
            pre_rep_propensity=pre_rep_propensity,
            pre_rep_hidden=pre_rep_hidden,
            pre_rep_mag_coef=pre_rep_mag_coef,
            pre_rep_mode=pre_rep_mode,
            pre_rep_warmup_epochs=pre_rep_warmup_epochs,
            dr_residual_warmup_epochs=dr_residual_warmup_epochs,
        )
    if name == "kancd":
        return train_and_eval_kancd(
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
            epochs=epochs,
            lr=lr,
            device=device,
            batch_size=batch_size,
            progress_desc=progress_desc,
            sample_weight=sample_weight,
            dr_residual_coef=dr_residual_coef,
            dr_residual_target=dr_residual_target,
            dr_residual_weight=dr_residual_weight,
            dr_residual_mask=dr_residual_mask,
            pre_rep_debias=pre_rep_debias,
            pre_rep_propensity=pre_rep_propensity,
            pre_rep_hidden=pre_rep_hidden,
            pre_rep_mag_coef=pre_rep_mag_coef,
            pre_rep_mode=pre_rep_mode,
            pre_rep_warmup_epochs=pre_rep_warmup_epochs,
            dr_residual_warmup_epochs=dr_residual_warmup_epochs,
        )
    if name == "irt":
        return train_and_eval_irt(
            train_u,
            train_i,
            train_y,
            test_u,
            test_i,
            n_user,
            n_item,
            epochs=epochs,
            lr=lr,
            device=device,
            batch_size=batch_size,
            progress_desc=progress_desc,
            sample_weight=sample_weight,
            dr_residual_coef=dr_residual_coef,
            dr_residual_target=dr_residual_target,
            dr_residual_weight=dr_residual_weight,
            dr_residual_mask=dr_residual_mask,
            pre_rep_debias=pre_rep_debias,
            pre_rep_propensity=pre_rep_propensity,
            pre_rep_hidden=pre_rep_hidden,
            pre_rep_mag_coef=pre_rep_mag_coef,
            pre_rep_mode=pre_rep_mode,
            pre_rep_warmup_epochs=pre_rep_warmup_epochs,
            dr_residual_warmup_epochs=dr_residual_warmup_epochs,
        )
    if name == "dkt":
        return train_and_eval_dkt(
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
            epochs=epochs,
            lr=lr,
            device=device,
            batch_size=batch_size,
            progress_desc=progress_desc,
            sample_weight=sample_weight,
            dr_residual_coef=dr_residual_coef,
            dr_residual_target=dr_residual_target,
            dr_residual_weight=dr_residual_weight,
            dr_residual_mask=dr_residual_mask,
            pre_rep_debias=pre_rep_debias,
            pre_rep_propensity=pre_rep_propensity,
            pre_rep_hidden=pre_rep_hidden,
            pre_rep_mag_coef=pre_rep_mag_coef,
            pre_rep_mode=pre_rep_mode,
            pre_rep_warmup_epochs=pre_rep_warmup_epochs,
            dr_residual_warmup_epochs=dr_residual_warmup_epochs,
        )
    raise ValueError(f"Unknown CD model: {model_name}")


def finetune_kancd(
    model: object,
    train_u,
    train_i,
    train_k,
    train_y,
    n_user: int,
    epochs=1,
    lr=1e-3,
    device="cpu",
    batch_size=1024,
    freeze_student: bool = True,
    sample_weight: Optional[np.ndarray] = None,
):
    import torch
    import torch.nn.functional as F

    net = _get_torch_net(model)
    if net is None:
        raise RuntimeError("Could not find KaNCD torch network for finetune.")
    dev = torch.device(device)
    net.to(dev)

    if freeze_student:
        emb = None
        for _, sub in net.named_modules():
            if isinstance(sub, torch.nn.Embedding) and int(sub.weight.shape[0]) == int(n_user):
                emb = sub
                break
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
        desc="KaNCD finetune",
        leave=False,
        dynamic_ncols=True,
    ):
        net.train(True)
        perm = idx[torch.randperm(n, device=dev)]
        for s in range(0, n, batch_size):
            b = perm[s : s + batch_size]
            uu, ii, kk, yy = u_tr[b], i_tr[b], k_tr[b], y_tr[b]
            opt.zero_grad(set_to_none=True)
            yhat = _forward(net, uu, ii, kk).view(-1)
            yhat = torch.clamp(yhat, 1e-6, 1 - 1e-6)
            if w_tr is None:
                loss = F.binary_cross_entropy(yhat, yy)
            else:
                ww = w_tr[b]
                loss = F.binary_cross_entropy(yhat, yy, reduction="none")
                loss = (loss * ww).mean()
            loss.backward()
            opt.step()


def finetune_irt(
    model: object,
    train_u,
    train_i,
    train_y,
    n_user: int,
    epochs=1,
    lr=1e-3,
    device="cpu",
    batch_size=1024,
    freeze_student: bool = True,
    sample_weight: Optional[np.ndarray] = None,
):
    import torch
    import torch.nn.functional as F

    net = _get_torch_net(model)
    if net is None:
        raise RuntimeError("Could not find IRT torch network for finetune.")
    dev = torch.device(device)
    net.to(dev)

    if freeze_student:
        emb = None
        for _, sub in net.named_modules():
            if isinstance(sub, torch.nn.Embedding) and int(sub.weight.shape[0]) == int(n_user):
                emb = sub
                break
        if emb is not None:
            emb.weight.requires_grad_(False)

    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        return
    opt = torch.optim.Adam(params, lr=lr)

    u_tr = torch.as_tensor(train_u, dtype=torch.long, device=dev)
    i_tr = torch.as_tensor(train_i, dtype=torch.long, device=dev)
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
        desc="IRT finetune",
        leave=False,
        dynamic_ncols=True,
    ):
        net.train(True)
        perm = idx[torch.randperm(n, device=dev)]
        for s in range(0, n, batch_size):
            b = perm[s : s + batch_size]
            uu, ii, yy = u_tr[b], i_tr[b], y_tr[b]
            opt.zero_grad(set_to_none=True)
            yhat = net(uu, ii).view(-1)
            yhat = torch.clamp(yhat, 1e-6, 1 - 1e-6)
            if w_tr is None:
                loss = F.binary_cross_entropy(yhat, yy)
            else:
                ww = w_tr[b]
                loss = F.binary_cross_entropy(yhat, yy, reduction="none")
                loss = (loss * ww).mean()
            loss.backward()
            opt.step()


def finetune_dkt(
    model: object,
    train_u,
    train_i,
    train_k,
    train_y,
    n_user: int,
    epochs=1,
    lr=1e-3,
    device="cpu",
    batch_size=1024,
    freeze_student: bool = True,
    sample_weight: Optional[np.ndarray] = None,
):
    net = _get_torch_net(model)
    if net is None:
        raise RuntimeError("Could not find DKT torch network for finetune.")

    if freeze_student and hasattr(net, "student_emb") and hasattr(net.student_emb, "weight"):
        try:
            net.student_emb.weight.requires_grad_(False)
        except Exception:
            pass

    train_seq_map = _build_sequence_map(
        user_ids=train_u,
        item_ids=train_i,
        labels=train_y,
        sample_weight=sample_weight,
    )
    _train_dkt_core(
        model=net,
        train_seq_map=train_seq_map,
        epochs=int(epochs),
        lr=float(lr),
        device=device,
        batch_size=int(batch_size),
        progress_desc="DKT finetune",
    )

    hist: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for uid, seq in train_seq_map.items():
        hist[int(uid)] = (seq["items"].copy(), seq["y"].copy())
    setattr(net, "_dkt_train_history", hist)


def finetune_cd(
    model_name: str,
    model: object,
    train_u,
    train_i,
    train_k,
    train_y,
    n_user: int,
    epochs=1,
    lr=1e-3,
    device="cpu",
    batch_size=1024,
    freeze_student: bool = True,
    sample_weight: Optional[np.ndarray] = None,
):
    name = model_name.lower()
    if name == "ncdm":
        return finetune_ncdm(
            model,
            train_u,
            train_i,
            train_k,
            train_y,
            n_user=n_user,
            epochs=epochs,
            lr=lr,
            device=device,
            batch_size=batch_size,
            freeze_student=freeze_student,
            sample_weight=sample_weight,
        )
    if name == "kancd":
        return finetune_kancd(
            model,
            train_u,
            train_i,
            train_k,
            train_y,
            n_user=n_user,
            epochs=epochs,
            lr=lr,
            device=device,
            batch_size=batch_size,
            freeze_student=freeze_student,
            sample_weight=sample_weight,
        )
    if name == "irt":
        return finetune_irt(
            model,
            train_u,
            train_i,
            train_y,
            n_user=n_user,
            epochs=epochs,
            lr=lr,
            device=device,
            batch_size=batch_size,
            freeze_student=freeze_student,
            sample_weight=sample_weight,
        )
    if name == "dkt":
        return finetune_dkt(
            model,
            train_u,
            train_i,
            train_k,
            train_y,
            n_user=n_user,
            epochs=epochs,
            lr=lr,
            device=device,
            batch_size=batch_size,
            freeze_student=freeze_student,
            sample_weight=sample_weight,
        )
    raise ValueError(f"Unknown CD model: {model_name}")


def predict_kancd(
    model: object,
    test_u,
    test_i,
    test_k,
    device="cpu",
    batch_size=1024,
    progress_desc: Optional[str] = None,
):
    import torch

    net = _get_torch_net(model)
    if net is None:
        raise RuntimeError("Could not find KaNCD torch network for inference.")
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
            desc=progress_desc if progress_desc else "KaNCD infer",
            leave=False,
            dynamic_ncols=True,
        ):
            uu = u_te[s : s + batch_size]
            ii = i_te[s : s + batch_size]
            kk = k_te[s : s + batch_size]
            yhat = _forward(net, uu, ii, kk).view(-1)
            outs.append(yhat.detach().float().cpu().numpy())
    return np.concatenate(outs, axis=0).reshape(-1)


def predict_irt(
    model: object,
    test_u,
    test_i,
    device="cpu",
    batch_size=1024,
    progress_desc: Optional[str] = None,
):
    import torch

    net = _get_torch_net(model)
    if net is None:
        raise RuntimeError("Could not find IRT torch network for inference.")
    dev = torch.device(device)
    net.to(dev)
    net.eval()

    u_te = torch.as_tensor(test_u, dtype=torch.long, device=dev)
    i_te = torch.as_tensor(test_i, dtype=torch.long, device=dev)

    outs = []
    with torch.no_grad():
        for s in tqdm(
            range(0, len(test_u), batch_size),
            desc=progress_desc if progress_desc else "IRT infer",
            leave=False,
            dynamic_ncols=True,
        ):
            uu = u_te[s : s + batch_size]
            ii = i_te[s : s + batch_size]
            yhat = net(uu, ii).view(-1)
            outs.append(yhat.detach().float().cpu().numpy())
    return np.concatenate(outs, axis=0).reshape(-1)


def predict_dkt(
    model: object,
    test_u,
    test_i,
    test_k,
    device="cpu",
    batch_size=1024,
    progress_desc: Optional[str] = None,
):
    import torch

    net = _get_torch_net(model)
    if net is None:
        raise RuntimeError("Could not find DKT torch network for inference.")
    dev = torch.device(device)
    net.to(dev)
    net.eval()

    u = np.asarray(test_u).astype(np.int64).reshape(-1)
    i = np.asarray(test_i).astype(np.int64).reshape(-1)
    if int(u.shape[0]) != int(i.shape[0]):
        raise ValueError("test_u/test_i size mismatch")

    users_order = []
    seen = set()
    for uid in u:
        v = int(uid)
        if v not in seen:
            seen.add(v)
            users_order.append(v)
    users_arr = np.asarray(users_order, dtype=np.int64)

    hist_raw = getattr(net, "_dkt_train_history", {})
    hist_map: Dict[int, Dict[str, np.ndarray]] = {}
    if isinstance(hist_raw, dict):
        for uid, pair in hist_raw.items():
            try:
                uu = int(uid)
            except Exception:
                continue
            if isinstance(pair, tuple) and len(pair) == 2:
                items, labels = pair
                hist_map[uu] = {
                    "items": np.asarray(items).astype(np.int64).reshape(-1),
                    "y": np.asarray(labels).astype(np.float32).reshape(-1),
                }

    h_map = _warm_start_hidden_from_train(net, hist_map, users_arr, device=device)
    out = np.zeros_like(u, dtype=np.float32)
    it = range(int(u.shape[0]))
    it = tqdm(it, desc=progress_desc if progress_desc else "DKT infer", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for idx in it:
            uid = int(u[idx])
            item = int(i[idx])
            h = h_map.get(uid)
            if h is None:
                uu = torch.as_tensor([uid], dtype=torch.long, device=dev)
                h = net.student_emb(uu)
            item_t = torch.as_tensor([item], dtype=torch.long, device=dev)
            p = net.predict_from_hidden(h, item_t)
            p = torch.clamp(p, 1e-6, 1 - 1e-6)
            out[idx] = float(p[0].detach().cpu().item())
            h_next = net.update_hidden(h, item_t, p.detach())
            h_map[uid] = h_next

    return out.astype(float).reshape(-1)


def predict_cd(
    model_name: str,
    model: object,
    test_u,
    test_i,
    test_k,
    device="cpu",
    batch_size=1024,
    progress_desc: Optional[str] = None,
):
    name = model_name.lower()
    if name == "ncdm":
        return predict_ncdm(model, test_u, test_i, test_k, device=device, batch_size=batch_size, progress_desc=progress_desc)
    if name == "kancd":
        return predict_kancd(model, test_u, test_i, test_k, device=device, batch_size=batch_size, progress_desc=progress_desc)
    if name == "irt":
        return predict_irt(model, test_u, test_i, device=device, batch_size=batch_size, progress_desc=progress_desc)
    if name == "dkt":
        return predict_dkt(model, test_u, test_i, test_k, device=device, batch_size=batch_size, progress_desc=progress_desc)
    raise ValueError(f"Unknown CD model: {model_name}")
