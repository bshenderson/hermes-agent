import { describe, expect, it } from "vitest";
import { auxiliaryAssignmentLabel, auxiliaryModelOptions } from "./auxiliary-model-options";
import type { ModelOptionsResult } from "@hermes/shared";

const catalog: ModelOptionsResult = { provider: "custom:homelab", model: "chat-default", providers: [
  { slug: "custom:homelab", name: "Homelab", models: ["chat-default"] },
] };

describe("task-scoped auxiliary presentation", () => {
  it("names legacy services without changing their selectable wire values or chat catalog", () => {
    const before = JSON.stringify(catalog);
    const assignment = { provider: "llamacpp-vision", model: "vision-default" };
    const result = auxiliaryModelOptions(catalog, assignment);
    expect(auxiliaryAssignmentLabel(assignment)).toBe("Homelab · Vision");
    expect(result.providers.at(-1)).toMatchObject({ slug: "llamacpp-vision", name: "Homelab · Vision",
      models: ["vision-default"], model_labels: { "vision-default": "Vision service" } });
    expect(JSON.stringify(catalog)).toBe(before);
    expect(result.providers.flatMap((p) => p.models || [])).not.toContain("routine-aux");
  });

  it("preserves auto, unknown providers and original model IDs", () => {
    expect(auxiliaryModelOptions(catalog, { provider: "auto", model: "" })).toBe(catalog);
    expect(auxiliaryAssignmentLabel({ provider: "auto", model: "" })).toBe("auto (use main model)");
    expect(auxiliaryAssignmentLabel({ provider: "other", model: "custom-model" })).toBe("other · custom-model");
    expect(auxiliaryModelOptions(catalog, { provider: "custom:llama-secondary", model: "local-compression-256k" })
      .providers.at(-1)?.slug).toBe("custom:llama-secondary");
  });

  it("merges a configured current route without mutating or duplicating providers", () => {
    const assignment = { provider: "custom:homelab", model: "chat-default" };
    const result = auxiliaryModelOptions(catalog, assignment);
    expect(result.providers).toHaveLength(1);
    expect(result.providers[0].models).toEqual(["chat-default"]);
    expect(result.providers[0].name).toBe("Homelab");
  });

  it("includes an inherited service without exposing other capability routes", () => {
    const result = auxiliaryModelOptions(catalog, { provider: "auto", model: "", effective: {
      provider: "llamacpp-aux", model: "routine-aux", base_url: "http://local/v1",
    } });
    expect(result.providers.at(-1)?.models).toEqual(["routine-aux"]);
    expect(result.providers.flatMap((p) => p.models || [])).not.toContain("vision-default");
  });
});
