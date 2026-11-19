const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const {test} = require("node:test");
const {runInNewContext} = require("node:vm");

const code = readFileSync(join(__dirname, "..", "docs_assets", "versions.js"), "utf8");

async function load({base = "https://example.test/EconML/", manifest, responseOK = true, reject = false}) {
    const select = {
        hidden: true,
        options: [],
        replaceChildren(...options) { this.options = options; },
        addEventListener(event, callback) { this.callback = callback; },
    };
    const status = {};
    const panel = {
        dataset: {version: "v0.16.0"},
        querySelector(selector) { return selector === "select" ? select : status; },
    };
    const errors = [];
    const context = {
        URL,
        document: {
            currentScript: {src: `${base}_versions/versions.js`},
            getElementById() { return panel; },
            createElement() { return {}; },
        },
        fetch: async (url, options) => {
            assert.equal(url.href, `${base}versions.json`);
            assert.equal(options.cache, "no-cache");
            if (reject) { throw new Error("Offline"); }
            return {ok: responseOK, status: 404, json: async () => manifest};
        },
        window: {location: {assign(url) { context.navigated = url; }}},
        console: {error(...args) { errors.push(args); }},
    };
    runInNewContext(code, context);
    await new Promise((resolve) => setImmediate(resolve));
    return {select, status, errors, context};
}

function versions() {
    return {
        schema: 1,
        stable: "v0.17.1",
        versions: [
            {id: "prerelease", path: "prerelease/", label: "Prerelease (main)"},
            {id: "v0.17.1", path: "v0.17.1/", label: "0.17.1"},
            {id: "v0.16.0", path: "v0.16.0/", label: "0.16.0"},
        ],
    };
}

for (const base of ["https://www.pywhy.org/EconML/", "https://kbattocchi.github.io/EconML/", "https://example.test/"]) {
    test(`archived page sees new versions and switches within ${base}`, async () => {
        const {select, errors, context} = await load({base, manifest: versions()});
        assert.equal(select.hidden, false);
        assert.equal(select.options.length, 3);
        assert.equal(select.options[1].textContent, "0.17.1 (stable)");
        assert.equal(select.options[2].selected, true);
        select.value = select.options[0].value;
        select.callback();
        assert.equal(context.navigated, `${base}prerelease/index.html`);
        assert.equal(errors.length, 0);
    });
}

test("labels are text, not injected markup", async () => {
    const manifest = versions();
    manifest.versions[0].label = "<img src=x onerror=alert(1)>";
    const {select} = await load({manifest});
    assert.equal(select.options[0].textContent, manifest.versions[0].label);
    assert.equal(select.options[0].innerHTML, undefined);
});

for (const failure of ["http", "network", "schema", "path", "id", "duplicate", "missing", "stable"]) {
    test(`${failure} fails visibly and keeps the static fallback`, async () => {
        const manifest = versions();
        if (failure === "schema") { manifest.schema = 99; }
        if (failure === "path") { manifest.versions[0].path = "https://evil.test/"; }
        if (failure === "id") { manifest.versions[0].id = "../../outside"; }
        if (failure === "duplicate") { manifest.versions.push(manifest.versions[0]); }
        if (failure === "missing") { manifest.versions.pop(); }
        if (failure === "stable") { manifest.stable = "prerelease"; }
        const {select, status, errors} = await load({
            manifest, responseOK: failure !== "http", reject: failure === "network",
        });
        assert.equal(select.hidden, true);
        assert.equal(select.options.length, 0);
        assert.match(status.textContent, /all-versions link/);
        assert.equal(errors.length, 1);
    });
}
