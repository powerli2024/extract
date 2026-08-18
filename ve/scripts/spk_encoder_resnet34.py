#!/usr/bin/env python3
"""
wespeaker ResNet34-LM 说话人编码器（对齐 PS4 / wespeakerruntime）。

优先顺序:
  1) ONNX（目录内 *_resnet34_LM.onnx）— 避开 wespeaker 循环导入
  2) PyTorch：直接加载 ResNet34 类（绕过 wespeaker/__init__.py 重依赖）

兼容：新版 torchaudio 已移除 set_audio_backend / sox_effects，
wespeaker→s3prl 导入链会炸；本文件在 import wespeaker 前打补丁，
并优先按文件加载 models/resnet.py + pooling_layers.py。
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def _patch_torchaudio() -> None:
    """新版 torchaudio 兼容补丁（须在 import wespeaker / s3prl 之前调用）。"""
    try:
        import torchaudio as ta
    except ImportError:
        return

    if not hasattr(ta, "set_audio_backend"):
        ta.set_audio_backend = lambda *_a, **_k: None  # type: ignore[attr-defined]
    if not hasattr(ta, "get_audio_backend"):
        ta.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]

    mod = sys.modules.get("torchaudio.sox_effects")
    if mod is not None and callable(getattr(mod, "apply_effects_tensor", None)):
        return
    try:
        import torchaudio.sox_effects as _se  # noqa: F401

        if callable(getattr(_se, "apply_effects_tensor", None)):
            return
    except Exception:
        pass

    stub = types.ModuleType("torchaudio.sox_effects")

    def _unavailable(*_a, **_k):
        raise RuntimeError(
            "torchaudio.sox_effects 不可用（新版已移除）；声纹推理不依赖它。"
        )

    stub.apply_effects_tensor = _unavailable  # type: ignore[attr-defined]
    stub.apply_effects_file = _unavailable  # type: ignore[attr-defined]
    sys.modules["torchaudio.sox_effects"] = stub
    try:
        ta.sox_effects = stub  # type: ignore[attr-defined]
    except Exception:
        pass


_patch_torchaudio()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a / na, b / nb))


def _wav_to_fbank_np(wav: np.ndarray, sr: int = 16000, n_mels: int = 80) -> np.ndarray:
    """与 wespeakerruntime 对齐的 fbank+CMN，返回 [T, D]。"""
    import torchaudio.compliance.kaldi as kaldi

    w = torch.from_numpy(np.asarray(wav, dtype=np.float32).reshape(1, -1))
    w = w * (1 << 15)
    feat = kaldi.fbank(
        w,
        num_mel_bins=n_mels,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=sr,
        window_type="hamming",
        use_energy=False,
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    return feat.numpy().astype(np.float32)


def _resolve_paths(model_dir: Path) -> tuple[Path, Path | None, Path | None]:
    """返回 (config.yaml, pt_weight|None, onnx|None)。"""
    config = model_dir / "config.yaml"
    if not config.is_file():
        alts = list(model_dir.glob("**/config.yaml"))
        if alts:
            config = alts[0]
            model_dir = config.parent
    onnx = None
    for p in sorted(model_dir.glob("*resnet34*.onnx")) + sorted(model_dir.glob("*.onnx")):
        onnx = p
        break
    pts = [
        model_dir / "avg_model.pt",
        model_dir / "avg_model",
        model_dir / "model_5.pt",
    ] + sorted(model_dir.glob("model_*.pt"), reverse=True) + sorted(
        model_dir.glob("*.pt")
    )
    pt = next((p for p in pts if p.is_file()), None)
    return config, pt, onnx


def _ensure_ns_pkg(fullname: str, path: Path) -> None:
    """注册空包命名空间，避免执行有副作用的 __init__.py。"""
    if fullname in sys.modules:
        mod = sys.modules[fullname]
        if not hasattr(mod, "__path__"):
            mod.__path__ = [str(path)]  # type: ignore[attr-defined]
        return
    m = types.ModuleType(fullname)
    m.__path__ = [str(path)]  # type: ignore[attr-defined]
    m.__package__ = fullname
    m.__file__ = str(path / "__init__.py")
    sys.modules[fullname] = m


def _load_submodule(fullname: str, file_path: Path):
    if fullname in sys.modules:
        return sys.modules[fullname]
    spec = importlib.util.spec_from_file_location(fullname, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {fullname} ← {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_resnet_cls_via_file():
    """不执行 wespeaker/__init__.py，直接加载 models/resnet.py + pooling_layers。"""
    spec = importlib.util.find_spec("wespeaker")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("未安装 wespeaker（pip/git install wespeaker）")
    root = Path(list(spec.submodule_search_locations)[0])
    models = root / "models"
    if not (models / "resnet.py").is_file():
        raise ImportError(f"找不到 {models / 'resnet.py'}")

    _ensure_ns_pkg("wespeaker", root)
    _ensure_ns_pkg("wespeaker.models", models)

    # resnet.py: import wespeaker.models.pooling_layers as pooling_layers
    pool_py = models / "pooling_layers.py"
    if not pool_py.is_file():
        raise ImportError(f"找不到 {pool_py}")
    _load_submodule("wespeaker.models.pooling_layers", pool_py)

    return _load_submodule("wespeaker.models.resnet", models / "resnet.py")


def _load_resnet_cls(model_name: str):
    """按名取 ResNet 类；优先绕过 wespeaker 包初始化。"""
    _patch_torchaudio()
    name = (model_name or "ResNet34").replace("-", "_")
    errors: list[str] = []

    try:
        mod = _load_resnet_cls_via_file()
        if hasattr(mod, name):
            return getattr(mod, name)
        if hasattr(mod, "ResNet34"):
            return getattr(mod, "ResNet34")
        raise ImportError(f"wespeaker.models.resnet 中无 {name}")
    except Exception as e:
        errors.append(f"file_load={e}")

    try:
        mod = importlib.import_module("wespeaker.models.resnet")
        if hasattr(mod, name):
            return getattr(mod, name)
        if hasattr(mod, "ResNet34"):
            return getattr(mod, "ResNet34")
    except Exception as e:
        errors.append(f"import_module={e}")

    raise ImportError("无法加载 ResNet 类: " + "; ".join(errors))


class FrozenSpeakerEncoder(nn.Module):
    """
    冻结 ResNet34 声纹编码器。
    输入波形 [B, T] float32 @16kHz → embedding [B, D]
    """

    def __init__(self, model_dir: str | Path, device: str = "cuda:0"):
        super().__init__()
        import yaml

        _patch_torchaudio()

        model_dir = Path(model_dir)
        config_path, pt_path, onnx_path = _resolve_paths(model_dir)
        if not config_path.is_file():
            raise FileNotFoundError(f"config.yaml 不存在: {model_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            spk_cfg = yaml.safe_load(f) or {}
        spk_model_name = spk_cfg.get("model", "ResNet34")
        spk_model_kwargs = dict(spk_cfg.get("model_args") or {})
        self.num_mel_bins = int(spk_model_kwargs.get("feat_dim", 80))
        self.sample_rate = 16000
        self.model_dir = str(model_dir)
        self._device = device
        self._backend = ""
        self.encoder = None
        self._ort = None
        self._ort_in = None
        self._ort_out = None

        # ── 1) ONNX 优先 ──
        if onnx_path is not None:
            try:
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.intra_op_num_threads = 4
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                try:
                    self._ort = ort.InferenceSession(str(onnx_path), so, providers=providers)
                except Exception:
                    self._ort = ort.InferenceSession(
                        str(onnx_path), so, providers=["CPUExecutionProvider"]
                    )
                self._ort_in = self._ort.get_inputs()[0].name
                self._ort_out = self._ort.get_outputs()[0].name
                self._backend = f"onnx:{onnx_path.name}"
                self.register_buffer("_dummy", torch.zeros(1), persistent=False)
                return
            except Exception as e:
                self._ort = None
                self._onnx_err = str(e)

        # ── 2) PyTorch ResNet ──
        if pt_path is None:
            raise FileNotFoundError(
                f"无可用权重: {model_dir}（需 .onnx 或 avg_model/model_5.pt）"
            )
        try:
            spk_cls = _load_resnet_cls(spk_model_name)
        except Exception as e:
            try:
                _patch_torchaudio()
                from wespeaker.models.speaker_model import get_speaker_model

                spk_cls = get_speaker_model(spk_model_name)
            except Exception as e2:
                raise RuntimeError(
                    f"无法加载 ResNet 类: direct={e}; get_speaker_model={e2}\n"
                    "可先: pip install modelscope 并用 PRESENCE_BACKEND=eres2netv2"
                ) from e2

        self.encoder = spk_cls(**spk_model_kwargs)
        try:
            ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(pt_path), map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        if isinstance(state_dict, dict) and any(
            k.startswith("module.") for k in state_dict
        ):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        self.encoder.load_state_dict(state_dict, strict=False)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self._backend = f"pt:{pt_path.name}"
        self.to(device)

    def _wav_to_fbank(self, wav: torch.Tensor) -> torch.Tensor:
        import torchaudio.compliance.kaldi as kaldi

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        fbank_list = []
        for i in range(wav.shape[0]):
            w = wav[i].unsqueeze(0) * (1 << 15)
            feat = kaldi.fbank(
                w,
                num_mel_bins=self.num_mel_bins,
                frame_length=25,
                frame_shift=10,
                dither=0.0,
                sample_frequency=self.sample_rate,
                window_type="hamming",
                use_energy=False,
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            fbank_list.append(feat)
        max_len = max(f.shape[0] for f in fbank_list)
        padded = []
        for f in fbank_list:
            if f.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - f.shape[0], f.shape[1], dtype=f.dtype, device=f.device
                )
                f = torch.cat([f, pad], dim=0)
            padded.append(f)
        return torch.stack(padded, dim=0)

    @torch.inference_mode()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        if self._ort is not None:
            embs = []
            for i in range(wav.shape[0]):
                feat = _wav_to_fbank_np(
                    wav[i].detach().cpu().numpy(),
                    sr=self.sample_rate,
                    n_mels=self.num_mel_bins,
                )
                x = feat[np.newaxis, ...]
                try:
                    out = self._ort.run([self._ort_out], {self._ort_in: x})[0]
                except Exception:
                    out = self._ort.run([self._ort_out], {self._ort_in: feat})[0]
                e = np.asarray(out).reshape(-1)
                embs.append(torch.from_numpy(e.astype(np.float32)))
            return torch.stack(embs, dim=0)

        assert self.encoder is not None
        device = next(self.encoder.parameters()).device
        wav = wav.to(device)
        fbank = self._wav_to_fbank(wav)
        emb = self.encoder(fbank)
        if isinstance(emb, (tuple, list)):
            emb = emb[-1]
        return emb

    def embed_numpy(self, wav: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.asarray(wav, dtype=np.float32).reshape(-1))
        e = self.forward(t)
        return e.squeeze(0).detach().cpu().numpy().astype(np.float32)


class DualLangSpeakerEncoder:
    """按语言选择 en / chs ResNet34；中文数据默认 chs。"""

    def __init__(
        self,
        chs_dir: str = "",
        en_dir: str = "",
        device: str = "cuda:0",
        default_lang: str = "zh",
    ):
        self.default_lang = (default_lang or "zh").lower()
        self.chs: Optional[FrozenSpeakerEncoder] = None
        self.en: Optional[FrozenSpeakerEncoder] = None
        self.device = device
        errs = []
        if chs_dir and Path(chs_dir).is_dir():
            try:
                self.chs = FrozenSpeakerEncoder(chs_dir, device=device)
            except Exception as e:
                errs.append(f"chs({chs_dir}): {e}")
        if en_dir and Path(en_dir).is_dir():
            try:
                self.en = FrozenSpeakerEncoder(en_dir, device=device)
            except Exception as e:
                errs.append(f"en({en_dir}): {e}")
        self.load_errors = errs

    @property
    def available(self) -> bool:
        return self.chs is not None or self.en is not None

    def status(self) -> str:
        parts = []
        if self.chs:
            parts.append(f"chs={self.chs.model_dir} [{self.chs._backend}]")
        if self.en:
            parts.append(f"en={self.en.model_dir} [{self.en._backend}]")
        if not parts:
            return "unavailable: " + "; ".join(self.load_errors)
        return " | ".join(parts)

    def _pick(self, lang: str) -> FrozenSpeakerEncoder:
        lang = (lang or self.default_lang).lower()
        if lang in ("zh", "chs", "cn", "chinese"):
            if self.chs is not None:
                return self.chs
            if self.en is not None:
                return self.en
        else:
            if self.en is not None:
                return self.en
            if self.chs is not None:
                return self.chs
        raise RuntimeError("无可用 ResNet34 编码器")

    def embed(self, wav: np.ndarray, lang: str = "") -> np.ndarray:
        return self._pick(lang or self.default_lang).embed_numpy(wav)

    def pair_sims(
        self,
        enroll: np.ndarray,
        tse: np.ndarray,
        mix: np.ndarray | None = None,
        lang: str = "",
    ) -> dict:
        e_enr = self.embed(enroll, lang)
        e_tse = self.embed(tse, lang)
        sim_tse = cosine_sim(e_enr, e_tse)
        out = {
            "sim_enroll_tse": round(sim_tse, 4),
            "spk_encoder": "cnceleb_resnet34_LM"
            if (lang or self.default_lang).lower() in ("zh", "chs", "cn", "chinese")
            and self.chs
            else "voxceleb_resnet34_LM",
        }
        if mix is not None:
            e_mix = self.embed(mix, lang)
            sim_mix = cosine_sim(e_enr, e_mix)
            out["sim_enroll_mix"] = round(sim_mix, 4)
            out["sim_delta"] = round(sim_tse - sim_mix, 4)
        return out
