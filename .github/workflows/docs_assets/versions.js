/* Shared by archived pages: keep the HTML hook contract backwards compatible. */
(() => {
    "use strict";
    const script = document.currentScript;
    const panel = document.getElementById("econml-docs-versions");
    const base = new URL("../", script.src);
    const select = panel.querySelector("select");
    const status = panel.querySelector('[role="status"]');
    const currentLabel = panel.querySelector("strong");
    const sidebar = document.querySelector(".wy-nav-side");

    if (sidebar) {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.className = "rst-current-version";
        summary.textContent = `Version: ${panel.dataset.version}`;
        const content = document.createElement("div");
        content.className = "rst-other-versions";
        content.append(currentLabel, panel.querySelector("a"), panel.querySelector("label"), select, status);
        details.append(summary, content);
        panel.replaceChildren(details);
        panel.classList.add("rst-versions");
        sidebar.classList.add("econml-has-versions");
        sidebar.append(panel);
    }

    const main = document.querySelector(".rst-content");
    if (panel.dataset.version === "prerelease" && main) {
        const banner = document.createElement("div");
        banner.className = "econml-docs-prerelease";
        banner.setAttribute("role", "note");
        banner.textContent = `${currentLabel.textContent}. `;
        const stable = document.createElement("a");
        stable.href = new URL("index.html", base).href;
        stable.textContent = "Read the stable documentation.";
        banner.append(stable);
        main.prepend(banner);
    }

    async function switchVersion() {
        select.disabled = true;
        status.textContent = "Checking the selected version...";
        const currentOption = Array.from(select.options).find((option) => option.dataset.current === "true");
        try {
            const current = new URL(window.location.href);
            if (!current.pathname.startsWith(base.pathname)) {
                throw new Error("Current page is outside the documentation site");
            }
            let page = current.pathname.slice(base.pathname.length);
            const prefix = `${panel.dataset.version}/`;
            if (page.startsWith(prefix)) {
                page = page.slice(prefix.length);
            }
            if (!page || page.endsWith("/")) {
                page += "index.html";
            }
            const target = new URL(`./${page}`, select.value);
            target.search = current.search;
            const response = await fetch(target, {method: "HEAD", cache: "no-cache"});
            if (response.status === 404) {
                window.location.assign(select.value);
            } else if (response.ok) {
                target.hash = current.hash;
                window.location.assign(target.href);
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
        } catch (error) {
            select.value = currentOption.value;
            select.disabled = false;
            status.textContent = "Could not check that version. Try again or use the all-versions link.";
            console.error("Could not switch documentation versions:", error);
        }
    }

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
                option.dataset.current = String(option.selected);
                return option;
            });
            if (!seen.has(panel.dataset.version) || !seen.has(manifest.stable) || manifest.stable === "prerelease") {
                throw new Error("Missing current or stable documentation");
            }
            select.replaceChildren(...options);
            select.addEventListener("change", switchVersion);
            select.hidden = false;
        } catch (error) {
            status.textContent = "Version list unavailable. Use the all-versions link.";
            console.error("Could not load documentation versions:", error);
        }
    }

    loadVersions();
})();
