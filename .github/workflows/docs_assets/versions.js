/* Shared by archived pages: keep the HTML hook contract backwards compatible. */
(() => {
    "use strict";
    const script = document.currentScript;
    const panel = document.getElementById("econml-docs-versions");
    const base = new URL("../", script.src);
    const select = panel.querySelector("select");
    const status = panel.querySelector('[role="status"]');

    async function loadVersions() {
        try {
            const response = await fetch(new URL("versions.json", base), {cache: "no-cache"});
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const manifest = await response.json();
            if (manifest.schema !== 1 || !Array.isArray(manifest.versions) || !manifest.versions.length) {
                throw new Error("Invalid documentation manifest");
            }
            const seen = new Set();
            const options = manifest.versions.map((entry) => {
                if (!/^(prerelease|v\d+\.\d+\.\d+)$/.test(entry.id)
                    || entry.path !== `${entry.id}/` || typeof entry.label !== "string"
                    || seen.has(entry.id)) {
                    throw new Error("Invalid documentation version");
                }
                seen.add(entry.id);
                const option = document.createElement("option");
                option.textContent = entry.label + (entry.id === manifest.stable ? " (stable)" : "");
                option.value = new URL(`${entry.path}index.html`, base).href;
                option.selected = entry.id === panel.dataset.version;
                return option;
            });
            if (!seen.has(panel.dataset.version) || !seen.has(manifest.stable) || manifest.stable === "prerelease") {
                throw new Error("Missing current or stable documentation");
            }
            select.replaceChildren(...options);
            select.addEventListener("change", () => window.location.assign(select.value));
            select.hidden = false;
        } catch (error) {
            status.textContent = "Version list unavailable. Use the all-versions link.";
            console.error("Could not load documentation versions:", error);
        }
    }

    loadVersions();
})();
