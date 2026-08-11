"""Guided, transactional cloud API configuration."""

from __future__ import annotations

from .config import AutoMemoryConfig, validate_config
from .errors import CancelledError, ConfigurationError, UsageError
from .models import CommandResult, EventKind, OutputEvent, SetupDraft
from .provider_presets import normalize_base_url, presets_for
from .security import validate_http_url


SERVICE_LABELS = {
    "llm": "LLM chat",
    "embedding": "Embedding",
    "vlm": "Vision model (VLM)",
    "reranker": "Reranker",
    "mineru": "MinerU parser",
    "web": "Web search",
}
SERVICE_ORDER = tuple(SERVICE_LABELS)


class _UserCancelled(Exception):
    pass


class SetupApplier:
    @staticmethod
    def commit(context, draft: SetupDraft) -> None:
        old_config = AutoMemoryConfig.from_dict(context.config.to_dict())
        snapshots = {name: context.credentials.get_persisted(name) for name in draft.secrets}
        try:
            for name, value in draft.secrets.items():
                context.credentials.set(name, value, persist=True)
            context.config_store.save(draft.config)
            context.reload_services(draft.config)
        except Exception as exc:
            rollback_failed = False
            try:
                for name, value in snapshots.items():
                    context.credentials.restore_persisted(name, value)
                context.config_store.save(old_config)
                context.reload_services(old_config)
            except Exception:
                rollback_failed = True
            hint = f"Restore the previous config from {context.paths.backups_dir}" if rollback_failed else "The previous configuration was restored"
            raise ConfigurationError("API setup could not be saved safely", hint=hint) from exc


class SetupWizard:
    def __init__(self, context, terminal, cancel) -> None:
        self.context, self.terminal, self.cancel = context, terminal, cancel

    def run(self) -> CommandResult:
        if not self.terminal.interactive:
            raise UsageError("/setup requires an interactive terminal", hint="Use AUTOMEMORY_* environment variables for automation")
        draft = SetupDraft(AutoMemoryConfig.from_dict(self.context.config.to_dict()))
        self.terminal.emit(OutputEvent(EventKind.RESULT, text="AutoMemory cloud API setup\nValues are staged in memory until final confirmation."))
        index = 0
        try:
            while index < len(SERVICE_ORDER):
                self.cancel.checkpoint()
                service = SERVICE_ORDER[index]
                choice = self.terminal.choose(
                    f"Configure {SERVICE_LABELS[service]}?",
                    [("configure", "Configure or replace"), ("keep", "Keep current settings")],
                    allow_back=index > 0,
                    allow_skip=True,
                )
                if choice == "cancel":
                    return self._cancelled()
                if choice == "back":
                    index -= 1
                    continue
                if choice in {"keep", "skip"}:
                    index += 1
                    continue
                try:
                    if self._configure_service(draft, service):
                        index += 1
                except ConfigurationError as exc:
                    self.terminal.emit(OutputEvent(EventKind.WARNING, text=exc.message, code=exc.code))
            validate_config(draft.config)
            self._show_summary(draft)
            draft.test_after_save = self.terminal.confirm("Test changed services after saving?", default=True)
            if not self.terminal.confirm("Save this API configuration?", default=False):
                return self._cancelled()
            SetupApplier.commit(self.context, draft)
            self.terminal.emit(OutputEvent(EventKind.RESULT, text="API configuration saved securely."))
            results = None
            if draft.test_after_save and draft.changed_services:
                results = self.context.connectivity.test_services(draft.changed_services, self.terminal, self.cancel)
            return CommandResult(text="API configuration saved securely.", data=results)
        except _UserCancelled:
            return self._cancelled()
        except CancelledError:
            self.terminal.emit(OutputEvent(EventKind.WARNING, text="Setup cancelled; existing configuration was not changed.", code="SETUP_CANCELLED"))
            raise

    def _configure_service(self, draft: SetupDraft, service: str) -> bool:
        options = [(item.id, item.label) for item in presets_for(service)]
        selected = self.terminal.choose(f"Select {SERVICE_LABELS[service]} provider", options, allow_back=True, allow_skip=True)
        if selected == "cancel":
            raise _UserCancelled()
        if selected == "back":
            return False
        if selected == "skip":
            return True
        preset = next(item for item in presets_for(service) if item.id == selected)
        if service in {"llm", "embedding", "vlm", "reranker"}:
            profile = getattr(draft.config, service)
            base_default = preset.base_url or profile.base_url
            model_default = profile.model if preset.id == "custom" else preset.default_model
            base_url = self.terminal.read_form_value("Base URL", base_default)
            if base_url.lower() == "back":
                return False
            validate_http_url(base_url, allow_private=True, resolve=False)
            model = self.terminal.read_form_value("Model", model_default)
            if model.lower() == "back":
                return False
            if not model.strip():
                raise ConfigurationError(f"{SERVICE_LABELS[service]} model is required")
            profile.base_url, profile.model = base_url.rstrip("/"), model.strip()
            self._collect_secret(draft, preset.credential_name, preset.requires_secret)
        elif service == "mineru":
            draft.config.mineru_mode = preset.id
            if preset.id == "official":
                draft.config.mineru_url = preset.base_url
            else:
                value = self.terminal.read_form_value("Self-hosted MinerU endpoint", draft.config.mineru_url)
                if value.lower() == "back":
                    return False
                validate_http_url(value, allow_private=True, resolve=False)
                draft.config.mineru_url = value.rstrip("/")
            self._collect_secret(draft, preset.credential_name, preset.requires_secret)
        elif service == "web":
            draft.config.web_provider = preset.id
            self._collect_secret(draft, preset.credential_name, preset.requires_secret)
        draft.changed_services.add(service)
        return True

    def _collect_secret(self, draft: SetupDraft, credential_name: str, required: bool) -> None:
        if not credential_name:
            return
        configured = self.context.credentials.configured(credential_name)
        if configured and not self.terminal.confirm(f"Replace the existing {credential_name}?", default=False):
            return
        if not required and not self.terminal.confirm(f"Configure optional {credential_name}?", default=False):
            return
        value = self.terminal.read_secret(f"{credential_name}: ").strip()
        if required and not value:
            raise ConfigurationError(f"{credential_name} is required")
        if value:
            draft.secrets[credential_name] = value

    def _show_summary(self, draft: SetupDraft) -> None:
        lines = ["Setup summary (credentials are never displayed):"]
        for service in SERVICE_ORDER:
            if service not in draft.changed_services:
                continue
            if service in {"llm", "embedding", "vlm", "reranker"}:
                profile = getattr(draft.config, service)
                lines.append(f"  {service}: {normalize_base_url(profile.base_url)} | model={profile.model} | credential={'updated' if profile.credential_name in draft.secrets else self.context.credentials.source(profile.credential_name)}")
            elif service == "mineru":
                lines.append(f"  mineru: {draft.config.mineru_mode} | {normalize_base_url(draft.config.mineru_url)} | credential={'updated' if 'mineru_api_key' in draft.secrets else self.context.credentials.source('mineru_api_key')}")
            else:
                credential = "updated" if "tavily_api_key" in draft.secrets else self.context.credentials.source("tavily_api_key")
                lines.append(f"  web: {draft.config.web_provider} | credential={credential if draft.config.web_provider == 'tavily' else 'not-required'}")
        self.terminal.emit(OutputEvent(EventKind.RESULT, text="\n".join(lines)))

    def _cancelled(self) -> CommandResult:
        text = "Setup cancelled; existing configuration was not changed."
        self.terminal.emit(OutputEvent(EventKind.WARNING, text=text, code="SETUP_CANCELLED"))
        return CommandResult(ok=False, text=text)
