export function icon(name, label = "") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("icon"); svg.setAttribute("aria-hidden", label ? "false" : "true");
  if (label) svg.setAttribute("aria-label", label);
  const use = document.createElementNS(svg.namespaceURI, "use"); use.setAttribute("href", `./assets/icons.svg#${name}`); svg.append(use); return svg;
}
export function renderTrace(container, trace) {
  if (!container || !trace) return;
  container.replaceChildren();
  const title = document.createElement("h4"); title.textContent = "检索轨迹"; container.append(title);
  const summary = document.createElement("p"); summary.textContent = `${trace.activeChannels.join(" + ") || "无通道"} · ${Number(trace.stageLatencyMs.total || 0).toFixed(1)} ms`; container.append(summary);
  if (trace.degradedChannels?.length) {
    const list = document.createElement("ul");
    trace.degradedChannels.forEach((item) => { const row = document.createElement("li"); row.textContent = `${item.channel}：${item.reason}`; list.append(row); });
    container.append(list);
  }
}
export function announce(message) { const region = document.getElementById("liveRegion"); if (region) region.textContent = message; }
