"""Shared serialization helpers for omni engine request payloads."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.engine import (
    AdditionalInformationEntry,
    AdditionalInformationPayload,
)

logger = init_logger(__name__)


def dtype_to_name(dtype: torch.dtype) -> str:
    """Convert torch dtype to a stable string name for serialization."""
    mapping = {
        torch.float32: "float32",
        torch.float: "float32",
        torch.float16: "float16",
        torch.half: "float16",
        torch.bfloat16: "bfloat16",
        torch.float64: "float64",
        torch.double: "float64",
        torch.int64: "int64",
        torch.long: "int64",
        torch.int32: "int32",
        torch.int: "int32",
        torch.int16: "int16",
        torch.short: "int16",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    return mapping.get(dtype, str(dtype).replace("torch.", ""))


def serialize_additional_information(
    raw_info: dict[str, Any] | AdditionalInformationPayload | None,
    *,
    log_prefix: str | None = None,
) -> AdditionalInformationPayload | None:
    """Serialize omni request metadata for EngineCore transport."""
    if raw_info is None:
        return None
    if isinstance(raw_info, AdditionalInformationPayload):
        return raw_info

    entries: dict[str, AdditionalInformationEntry] = {}
    for key, value in raw_info.items():
        if isinstance(value, torch.Tensor):
            value_cpu = value.detach().to("cpu").contiguous()
            if value_cpu.dtype == torch.bfloat16:
                value_cpu = value_cpu.to(torch.float32)
            entries[key] = AdditionalInformationEntry(
                tensor_data=value_cpu.numpy().tobytes(),
                tensor_shape=list(value_cpu.shape),
                tensor_dtype=dtype_to_name(value.dtype),
            )
            continue

        if isinstance(value, np.ndarray):
            arr = value.astype(np.float32) if value.dtype == np.float16 else value
            entries[key] = AdditionalInformationEntry(
                tensor_data=arr.tobytes(),
                tensor_shape=list(arr.shape),
                tensor_dtype=str(arr.dtype),
            )
            continue

        if isinstance(value, list):
            entries[key] = AdditionalInformationEntry(list_data=value)
            continue

        entries[key] = AdditionalInformationEntry(scalar_data=value)

    return AdditionalInformationPayload(entries=entries) if entries else None


def deserialize_additional_information(
    payload: dict | AdditionalInformationPayload | object | None,
) -> dict:
    """Deserialize an *additional_information* payload into a plain dict.

    Accepts:
    - ``dict`` – returned as-is.
    - ``AdditionalInformationPayload`` (or duck-typed with
      ``.entries``) – decoded entry-by-entry.
    - ``None`` – returns ``{}``.
    """

    if payload is None:
        return {}

    if isinstance(payload, dict):
        return payload

    try:
        entries = getattr(payload, "entries", None)
        if not isinstance(entries, dict):
            logger.exception("Failed to decode additional_information payload, entries field not a dict")
            return {}
        info: dict[str, object] = {}
        for k, entry in entries.items():
            if getattr(entry, "tensor_data", None) is not None:
                tensor_dtype = str(entry.tensor_dtype)
                if tensor_dtype == "bfloat16":
                    dt = np.dtype("float32")
                else:
                    dt = np.dtype(tensor_dtype)
                arr = np.frombuffer(entry.tensor_data, dtype=dt)
                arr = arr.reshape(getattr(entry, "tensor_shape", ()))
                # Return numpy array for non-torch dtypes (e.g. float32 from
                # audio_array), torch tensor for torch-originated data.
                if tensor_dtype in ("float32", "float64", "float16", "int32", "int64", "int16", "int8", "uint8", "bool"):
                    info[k] = arr
                else:
                    tensor = torch.from_numpy(arr.copy())
                    if tensor_dtype == "bfloat16":
                        tensor = tensor.to(torch.bfloat16)
                    info[k] = tensor
            elif getattr(entry, "list_data", None) is not None:
                info[k] = entry.list_data
            elif getattr(entry, "scalar_data", None) is not None:
                info[k] = entry.scalar_data
            else:
                info[k] = None
        return info
    except Exception:
        logger.exception("Failed to decode additional_information payload")

    return {}
