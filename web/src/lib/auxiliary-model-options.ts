import type { ModelOptionsResult } from "@hermes/shared";
import type { AuxiliaryTaskAssignment } from "./api-aux";

// Compatibility identities remain wire values; these names describe services, not engines.
const serviceLabels: Record<string, string> = {
  "llamacpp-vision": "Homelab · Vision",
  "llamacpp-aux": "Homelab · Utility text",
  "llama-aux": "Homelab · Utility text",
  "llama-secondary": "Homelab · Compression",
  "llamacpp-embed": "Homelab · Embeddings",
};

type Assignment = Pick<AuxiliaryTaskAssignment, "provider" | "model" | "effective">;

const routeLabels: Record<string, string> = {
  "vision-default": "Vision service",
  "routine-aux": "Utility text service",
  "local-compression-256k": "Compression service",
  "embed-default": "Embedding service",
  "rerank-default": "Reranking service",
};

export function auxiliaryAssignmentLabel(assignment?: Assignment | null): string {
  if (!assignment?.provider || assignment.provider === "auto") return "auto (use main model)";
  const service = serviceLabels[assignment.provider.replace(/^custom:/, "")];
  if (service && routeLabels[assignment.model]) return service;
  return `${service || assignment.provider} · ${
    assignment.model || "(provider default)"
  }`;
}

/** Retain only this task's configured service alongside ordinary chat alternatives.
 * Never discover an endpoint or add support routes to the shared chat catalog.
 */
export function auxiliaryModelOptions(
  options: ModelOptionsResult,
  assignment?: Assignment | null,
): ModelOptionsResult {
  if ((!assignment?.provider || assignment.provider === "auto") && assignment?.effective) {
    assignment = assignment.effective;
  }
  if (!assignment?.provider || assignment.provider === "auto" || !assignment.model) return options;
  const provider = assignment.provider;
  const service = serviceLabels[provider.replace(/^custom:/, "")];
  const label = (service && routeLabels[assignment.model]) || assignment.model;
  const existing = options.providers.find((row) => row.slug === provider);
  const row = existing
    ? { ...existing, models: [...new Set([...(existing.models || []), assignment.model])],
        model_labels: { ...existing.model_labels, [assignment.model]: existing.model_labels?.[assignment.model] || label } }
    : { slug: provider, name: service || provider, models: [assignment.model],
        model_labels: { [assignment.model]: label } };
  return { ...options, provider, model: assignment.model,
    providers: [...options.providers.filter((entry) => entry.slug !== provider), row] };
}
