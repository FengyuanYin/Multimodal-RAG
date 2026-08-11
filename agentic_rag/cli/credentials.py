"""Windows Credential Manager access with environment overrides."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import sys
from threading import RLock
from typing import Mapping

from .errors import ConfigurationError


ENV_NAMES = {
    "llm_api_key": ("AUTOMEMORY_LLM_API_KEY", "AGR_LLM_API_KEY", "OPENAI_API_KEY"),
    "embedding_api_key": ("AUTOMEMORY_EMBEDDING_API_KEY", "AGR_EMBEDDING_API_KEY", "OPENAI_API_KEY"),
    "vlm_api_key": ("AUTOMEMORY_VLM_API_KEY", "AGR_VLM_API_KEY", "OPENAI_API_KEY"),
    "reranker_api_key": ("AUTOMEMORY_RERANKER_API_KEY", "COHERE_API_KEY"),
    "mineru_api_key": ("AUTOMEMORY_MINERU_API_KEY", "MINERU_API_KEY"),
    "tavily_api_key": ("AUTOMEMORY_TAVILY_API_KEY", "TAVILY_API_KEY"),
}


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


class CREDENTIALW(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD), ("Type", wintypes.DWORD), ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR), ("LastWritten", FILETIME), ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)), ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD), ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR), ("UserName", wintypes.LPWSTR),
    ]


class CredentialStore:
    PREFIX = "AutoMemory"
    CRED_TYPE_GENERIC = 1
    CRED_PERSIST_LOCAL_MACHINE = 2
    ERROR_NOT_FOUND = 1168

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self.environ = environ if environ is not None else os.environ
        self._runtime: dict[str, str] = {}
        self._lock = RLock()
        self._advapi = None
        if sys.platform == "win32":
            self._advapi = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
            self._advapi.CredWriteW.argtypes = [ctypes.POINTER(CREDENTIALW), wintypes.DWORD]
            self._advapi.CredWriteW.restype = wintypes.BOOL
            self._advapi.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.POINTER(CREDENTIALW))]
            self._advapi.CredReadW.restype = wintypes.BOOL
            self._advapi.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
            self._advapi.CredDeleteW.restype = wintypes.BOOL
            self._advapi.CredFree.argtypes = [ctypes.c_void_p]

    @classmethod
    def _target(cls, name: str) -> str:
        if name not in ENV_NAMES:
            raise KeyError(name)
        return f"{cls.PREFIX}/{name}"

    def source(self, name: str) -> str:
        for env_name in ENV_NAMES.get(name, ()):
            if str(self.environ.get(env_name, "")).strip():
                return f"environment:{env_name}"
        with self._lock:
            if name in self._runtime:
                return "session"
        if self._read_windows(name):
            return "windows-credential-manager"
        return "not-configured"

    def get(self, name: str) -> str:
        for env_name in ENV_NAMES.get(name, ()):
            value = str(self.environ.get(env_name, "")).strip()
            if value:
                return value
        with self._lock:
            if name in self._runtime:
                return self._runtime[name]
        return self._read_windows(name)

    def get_persisted(self, name: str) -> str | None:
        """Read the managed slot without applying environment overrides."""
        self._target(name)
        with self._lock:
            if name in self._runtime:
                return self._runtime[name]
        value = self._read_windows(name)
        return value if value else None

    def restore_persisted(self, name: str, value: str | None) -> None:
        """Restore an exact snapshot without exposing it in diagnostics."""
        self._target(name)
        if value is None:
            self.delete(name)
        else:
            self.set(name, value, persist=True)

    def set(self, name: str, value: str, *, persist: bool = True) -> None:
        value = value.strip()
        if not value:
            raise ConfigurationError("Credential value cannot be empty")
        self._target(name)
        if persist and self._advapi:
            self._write_windows(name, value)
        else:
            with self._lock:
                self._runtime[name] = value

    def delete(self, name: str) -> bool:
        self._target(name)
        with self._lock:
            self._runtime.pop(name, None)
        if not self._advapi:
            return True
        if self._advapi.CredDeleteW(self._target(name), self.CRED_TYPE_GENERIC, 0):
            return True
        error = ctypes.get_last_error()
        if error == self.ERROR_NOT_FOUND:
            return False
        raise ConfigurationError(f"Windows Credential Manager delete failed ({error})")

    def configured(self, name: str) -> bool:
        return bool(self.get(name))

    def redaction_values(self) -> tuple[str, ...]:
        return tuple(value for name in ENV_NAMES if (value := self.get(name)))

    def _read_windows(self, name: str) -> str:
        if not self._advapi:
            return ""
        pointer = ctypes.POINTER(CREDENTIALW)()
        if not self._advapi.CredReadW(self._target(name), self.CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)):
            error = ctypes.get_last_error()
            if error == self.ERROR_NOT_FOUND:
                return ""
            raise ConfigurationError(f"Windows Credential Manager read failed ({error})")
        try:
            credential = pointer.contents
            raw = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
            return raw.decode("utf-16-le").rstrip("\x00")
        finally:
            self._advapi.CredFree(pointer)

    def _write_windows(self, name: str, value: str) -> None:
        raw = value.encode("utf-16-le")
        buffer = ctypes.create_string_buffer(raw)
        credential = CREDENTIALW()
        credential.Type = self.CRED_TYPE_GENERIC
        credential.TargetName = self._target(name)
        credential.CredentialBlobSize = len(raw)
        credential.CredentialBlob = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
        credential.Persist = self.CRED_PERSIST_LOCAL_MACHINE
        credential.UserName = "AutoMemory"
        try:
            if not self._advapi.CredWriteW(ctypes.byref(credential), 0):
                raise ConfigurationError(f"Windows Credential Manager write failed ({ctypes.get_last_error()})")
        finally:
            ctypes.memset(buffer, 0, len(buffer))
