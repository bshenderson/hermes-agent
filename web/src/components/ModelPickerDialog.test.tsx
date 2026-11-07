// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ModelPickerDialog } from "./ModelPickerDialog";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const provider = {
  slug: "custom:llamacpp-main", name: "Local", configured: true,
  models: ["chat-default", "media-default", "unknown-model"],
  model_labels: { "chat-default": "Qwen3.8 Chat (160K)", "media-default": "Qwen3.8 Media (160K)" },
};
const data = { model: "chat-default", provider: provider.slug,
  current_provider_name: "Local", current_provider_label: "Local",
  custom_providers: [], providers: [provider] };
let host: HTMLDivElement;
let root: Root;
afterEach(async () => { if (root) await act(async () => root.unmount()); host?.remove(); });

async function mount() {
  host = document.createElement("div"); document.body.append(host);
  root = createRoot(host);
  const apply = vi.fn();
  await act(async () => { root.render(<MemoryRouter><ModelPickerDialog
    onClose={vi.fn()} onApply={apply} loader={async () => data} /></MemoryRouter>); });
  return apply;
}

describe("graphical model labels", () => {
  it("renders provider labels while submitting the exact machine selector", async () => {
    const apply = await mount();
    expect(document.body.textContent).toContain("Qwen3.8 Chat (160K)");
    expect(document.body.textContent).toContain("Qwen3.8 Media (160K)");
    expect(document.body.textContent).toContain("unknown-model");
    expect(document.querySelector("header p")?.textContent).toContain("Qwen3.8 Chat (160K)");
    const row = [...document.querySelectorAll<HTMLElement>('[role="option"],button')].find(e => e.textContent?.includes("Qwen3.8 Media (160K)"));
    expect(row).toBeDefined();
    await act(async () => row!.click());
    const button = [...document.querySelectorAll("button")].find(e => e.textContent === "Switch");
    await act(async () => button!.click());
    expect(apply).toHaveBeenCalledWith({ model: "media-default", provider: provider.slug, persistGlobal: false, confirmExpensiveModel: false });
  });

  it.each(["Qwen3.8 Media", "media-default"])("finds a label by human name or selector: %s", async (query) => {
    await mount();
    const input = document.querySelector("input[placeholder]") as HTMLInputElement;
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, query);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const rows = [...document.querySelectorAll('[role="option"],button')].map(e => e.textContent);
    expect(rows.some(t => t?.includes("Qwen3.8 Media (160K)"))).toBe(true);
    expect(rows.some(t => t?.includes("Qwen3.8 Chat (160K)"))).toBe(false);
  });
});
