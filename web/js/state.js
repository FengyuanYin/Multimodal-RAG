import { TASK_STATES, TERMINAL_STATES } from "./constants.js";

const transitions = Object.freeze({
  idle: new Set(["running", "cancelled"]),
  running: new Set(["degraded", "succeeded", "failed", "cancelled"]),
  degraded: new Set(["running", "succeeded", "failed", "cancelled"]),
  succeeded: new Set(["running"]),
  failed: new Set(["running"]),
  cancelled: new Set(["running"]),
});

export class TaskState {
  constructor(name) {
    this.name = name;
    this.state = "idle";
    this.progress = 0;
    this.warning = "";
    this.error = null;
    this.controller = null;
    this.listeners = new Set();
  }
  subscribe(listener) { this.listeners.add(listener); listener(this.snapshot()); return () => this.listeners.delete(listener); }
  snapshot() { return { name: this.name, state: this.state, progress: this.progress, warning: this.warning, error: this.error }; }
  transition(next, detail = {}) {
    if (!TASK_STATES.includes(next) || !transitions[this.state].has(next)) throw new Error(`非法任务状态迁移: ${this.state} -> ${next}`);
    this.state = next;
    if (typeof detail.progress === "number") this.progress = Math.max(0, Math.min(100, detail.progress));
    if (detail.warning !== undefined) this.warning = detail.warning;
    if (detail.error !== undefined) this.error = detail.error;
    if (next === "running") this.controller = new AbortController();
    if (TERMINAL_STATES.has(next)) this.controller = null;
    this.emit();
  }
  setProgress(progress) { if (!["running", "degraded"].includes(this.state)) throw new Error("任务未运行"); this.progress = Math.max(0, Math.min(100, progress)); this.emit(); }
  cancel() { if (["running", "degraded"].includes(this.state)) { this.controller?.abort(); this.transition("cancelled"); } }
  emit() { const snapshot = this.snapshot(); this.listeners.forEach((listener) => listener(snapshot)); }
}
