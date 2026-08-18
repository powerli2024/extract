#!/usr/bin/env python3
"""Presence 声纹编码器：优先 ERes2NetV2（中文短时），回退 Wespeaker ResNet34-LM。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from audio_io import cosine_sim
from paths import setup_sys_path

setup_sys_path()


class PresenceEncoder:
    """统一接口: embed(wav) -> np.ndarray[D]。"""

    name: str = "base"

    def embed(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        raise NotImplementedError

    def embed_batch(self, wavs: list[np.ndarray], sr: int = 16000) -> list[np.ndarray]:
        return [self.embed(w, sr) for w in wavs]

    def sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return cosine_sim(self.embed(a), self.embed(b))


def _diagnose_python() -> str:
    import sys

    return f"python={sys.executable}"


def _import_modelscope_sv():
    """导入 ModelScope speaker-verification pipeline；失败时给出装到当前解释器的明确提示。"""
    import sys

    try:
        import modelscope as ms
    except ImportError as e:
        raise ImportError(
            f"当前解释器未安装 modelscope（{_diagnose_python()}）。\n"
            f"请用同一解释器安装: {sys.executable} -m pip install -U modelscope\n"
            f"（不要只 pip install；conda 环境内须 conda activate 后再装）\n"
            f"原始错误: {e}"
        ) from e

    try:
        from modelscope.pipelines import pipeline
    except Exception as e:
        raise ImportError(
            f"modelscope 在 {_diagnose_python()} 可 import，但 pipelines 失败: "
            f"{type(e).__name__}: {e}\n"
            f"modelscope={getattr(ms, '__file__', None)} version={getattr(ms, '__version__', '?')}"
        ) from e

    # Tasks 枚举在部分版本路径不同
    tasks_sv = None
    try:
        from modelscope.utils.constant import Tasks

        tasks_sv = getattr(Tasks, "speaker_verification", None) or getattr(
            Tasks, "speaker-verification", None
        )
    except Exception:
        Tasks = None  # type: ignore
    if tasks_sv is None:
        tasks_sv = "speaker-verification"
    return pipeline, tasks_sv, ms


class ERes2NetV2Encoder(PresenceEncoder):
    """ModelScope iic/speech_eres2netv2_sv_zh-cn_16k-common。"""

    name = "eres2netv2_zh"

    def __init__(self, model_dir: str | Path | None = None, device: str = "cuda:0"):
        self.device = device
        self.model_dir = Path(model_dir) if model_dir else None
        self._sv = None
        self._load()

    def _load(self) -> None:
        import sys

        print(f"[INFO] ERes2NetV2: {_diagnose_python()}", flush=True)
        pipeline, task, ms = _import_modelscope_sv()
        print(
            f"[INFO] modelscope={getattr(ms, '__version__', '?')} "
            f"← {getattr(ms, '__file__', None)}",
            flush=True,
        )

        model_id = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
        model_ref: str = model_id
        if self.model_dir and self.model_dir.is_dir():
            tip = self.model_dir / "MODELSCOPE_PATH.txt"
            if tip.is_file():
                model_ref = tip.read_text(encoding="utf-8").strip() or model_id
            elif (self.model_dir / "config.yaml").is_file() or (
                self.model_dir / "configuration.json"
            ).is_file():
                model_ref = str(self.model_dir)
        print(f"[INFO] load PresenceEncoder ERes2NetV2 ← {model_ref}", flush=True)

        last_err: Exception | None = None
        # 兼容不同 modelscope 的 device / revision 参数
        attempts = [
            dict(task=task, model=model_ref, model_revision="master", device=self.device),
            dict(task=task, model=model_ref, device=self.device),
            dict(task=task, model=model_ref, model_revision="master"),
            dict(task=task, model=model_ref),
            dict(task="speaker-verification", model=model_ref, device=self.device),
            dict(task="speaker-verification", model=model_id),
        ]
        for kwargs in attempts:
            try:
                self._sv = pipeline(**kwargs)
                print(f"[INFO] ERes2NetV2 pipeline OK kwargs={list(kwargs)}", flush=True)
                return
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"ERes2NetV2 pipeline 构建失败（{_diagnose_python()}）: {last_err}"
        ) from last_err

    def embed(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        import soundfile as sf
        import tempfile

        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, wav, sr)
            out = None
            # 兼容不同 modelscope 版本的 embedding API
            for kwargs in (
                {"output_emb": True},
                {"extract_emb": True},
                {},
            ):
                try:
                    out = self._sv([tmp], **kwargs) if kwargs else self._sv([tmp, tmp])
                    break
                except TypeError:
                    continue
                except Exception:
                    if not kwargs:
                        raise
                    continue
            if out is None:
                raise RuntimeError("ERes2NetV2 pipeline 调用失败")
        finally:
            Path(tmp).unlink(missing_ok=True)

        emb = self._parse_emb(out)
        if emb is None:
            # 最后手段：用模型内部 encoder（若暴露）
            raise RuntimeError(
                f"ERes2NetV2 输出无 embedding: type={type(out)} "
                f"keys={list(out) if isinstance(out, dict) else None}"
            )
        return emb

    @staticmethod
    def _parse_emb(out: Any) -> np.ndarray | None:
        if isinstance(out, dict):
            for k in ("embs", "embedding", "spk_embedding", "emb"):
                if k in out and out[k] is not None:
                    arr = np.asarray(out[k], dtype=np.float32)
                    if arr.ndim == 2:
                        arr = arr[0]
                    return arr.reshape(-1)
        if isinstance(out, (list, tuple)) and out:
            return ERes2NetV2Encoder._parse_emb(out[0])
        if isinstance(out, np.ndarray):
            arr = np.asarray(out, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[0]
            return arr.reshape(-1)
        return None


class ResNet34Encoder(PresenceEncoder):
    """回退：cnceleb_resnet34_LM（VE/scripts/spk_encoder_resnet34.py，不依赖 VD 仓在场）。"""

    name = "resnet34_lm"

    def __init__(self, model_dir: str | Path, device: str = "cuda:0"):
        setup_sys_path()
        try:
            from spk_encoder_resnet34 import FrozenSpeakerEncoder
        except ImportError as e:
            raise ImportError(
                "无法导入 spk_encoder_resnet34。"
                "请确认 VE/scripts/spk_encoder_resnet34.py 存在，"
                "并已 pip install wespeaker（或 onnxruntime）。"
                f" 原始错误: {e}"
            ) from e

        print(f"[INFO] load PresenceEncoder ResNet34-LM ← {model_dir}", flush=True)
        self._enc = FrozenSpeakerEncoder(model_dir, device=device)

    def embed(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        return self._enc.embed_numpy(wav)


class CAMPlusEncoder(ERes2NetV2Encoder):
    """ModelScope iic/speech_campplus_sv_zh-cn_16k-common。"""

    name = "campplus_zh"

    def _load(self) -> None:
        import sys

        print(f"[INFO] CAM++: {_diagnose_python()}", flush=True)
        pipeline, task, ms = _import_modelscope_sv()
        model_id = "iic/speech_campplus_sv_zh-cn_16k-common"
        model_ref: str = model_id
        if self.model_dir and self.model_dir.is_dir():
            tip = self.model_dir / "MODELSCOPE_PATH.txt"
            if tip.is_file():
                model_ref = tip.read_text(encoding="utf-8").strip() or model_id
            elif (self.model_dir / "config.yaml").is_file() or (
                self.model_dir / "configuration.json"
            ).is_file():
                model_ref = str(self.model_dir)
        print(f"[INFO] load PresenceEncoder CAM++ ← {model_ref}", flush=True)
        last_err: Exception | None = None
        attempts = [
            dict(task=task, model=model_ref, model_revision="master", device=self.device),
            dict(task=task, model=model_ref, device=self.device),
            dict(task=task, model=model_ref),
            dict(task="speaker-verification", model=model_id),
        ]
        for kwargs in attempts:
            try:
                self._sv = pipeline(**kwargs)
                print(f"[INFO] CAM++ pipeline OK kwargs={list(kwargs)}", flush=True)
                return
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"CAM++ pipeline 失败（{sys.executable}）: {last_err}") from last_err


class WespeakerLocalEncoder(PresenceEncoder):
    """WeSpeaker load_model_local：VoxBlink2 SimAM-ResNet 等。"""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        name: str = "wespeaker_local",
        device: str = "cuda:0",
    ):
        setup_sys_path()
        try:
            from spk_encoder_resnet34 import _patch_torchaudio

            _patch_torchaudio()
        except Exception:
            pass
        import wespeaker

        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"WeSpeaker 本地目录不存在: {model_dir}")
        print(f"[INFO] load PresenceEncoder {name} ← {model_dir}", flush=True)
        self.name = name
        self._m = wespeaker.load_model_local(str(model_dir))
        # device: 'cuda' / 'cpu' / 'cuda:0'
        dev = device
        if device.startswith("cuda"):
            dev = "cuda"
        try:
            self._m.set_device(dev)
        except Exception as e:
            print(f"[WARN] set_device({dev}) failed: {e}", flush=True)

    def embed(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        import soundfile as sf
        import tempfile

        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, wav, sr)
            emb = self._m.extract_embedding(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)
        arr = np.asarray(emb, dtype=np.float32).reshape(-1)
        return arr


def create_presence_encoder(
    backend: str = "eres2netv2",
    *,
    eres_dir: str | Path | None = None,
    resnet_dir: str | Path | None = None,
    campplus_dir: str | Path | None = None,
    vblink_dir: str | Path | None = None,
    device: str = "cuda:0",
) -> PresenceEncoder:
    backend = (backend or "eres2netv2").lower().strip()
    if backend in ("eres2netv2", "eres", "eres2net"):
        try:
            return ERes2NetV2Encoder(model_dir=eres_dir, device=device)
        except Exception as e:
            import sys

            print(f"[WARN] ERes2NetV2 加载失败，回退 ResNet34-LM", flush=True)
            print(f"[WARN]   python={sys.executable}", flush=True)
            print(f"[WARN]   {type(e).__name__}: {e}", flush=True)
            print(
                f"[HINT] 装到当前解释器: {sys.executable} -m pip install -U modelscope",
                flush=True,
            )
            cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
            if cause is not None:
                print(f"[WARN]   cause: {type(cause).__name__}: {cause}", flush=True)
            backend = "resnet34"
    if backend in ("campplus", "cam++", "campplus_zh"):
        return CAMPlusEncoder(model_dir=campplus_dir or eres_dir, device=device)
    if backend in ("vblink2", "vblink", "vblinkp", "samresnet34"):
        if not vblink_dir:
            raise FileNotFoundError(
                "vblink2 需要 --vblink-dir（WeSpeaker VoxBlink2 SimAM-ResNet34 目录，"
                "含 avg_model.pt + config.yaml）"
            )
        return WespeakerLocalEncoder(
            vblink_dir, name="vblink2_samresnet34", device=device
        )
    if backend in ("resnet34", "resnet34_lm", "wespeaker"):
        if not resnet_dir:
            raise FileNotFoundError("ResNet34 需要 --spk-chs-dir / SPK_CHS_DIR")
        return ResNet34Encoder(resnet_dir, device=device)
    raise ValueError(
        f"未知 presence backend: {backend}；"
        "可选: eres2netv2 | campplus | resnet34_lm | vblink2"
    )
