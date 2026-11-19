# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Check Pages publishing and the build-only reusable workflow contract."""

from pathlib import Path

import pytest
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
WORKFLOW = yaml.load(
    (WORKFLOW_DIR / "publish-documentation.yml").read_text(encoding="utf-8"),
    Loader=yaml.BaseLoader,
)
CI = yaml.load((WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


@pytest.mark.parametrize("trigger", ["workflow_dispatch", "workflow_call"])
def test_inputs_keep_build_only_without_azure_test_environment(trigger):
    """Both entry points support build-only runs without an obsolete test destination."""
    inputs = WORKFLOW["on"][trigger]["inputs"]
    assert inputs["publish"]["type"] == "boolean"
    assert "environment" not in inputs
    caller = CI["jobs"]["docs"]
    assert caller["with"]["publish"] == "false"
    assert set(caller["with"]) <= set(inputs)
    assert caller["uses"] == "./.github/workflows/publish-documentation.yml"


def test_pages_deployment_uses_successful_build_artifact():
    """Package the HTML even in build-only runs, but deploy only after a successful build."""
    build = WORKFLOW["jobs"]["create_docs"]
    publish = WORKFLOW["jobs"]["publish-docs"]
    assert publish["needs"] == "create_docs"
    assert publish["if"] == "${{ inputs.publish }}"
    assert not any(
        step.get("uses", "").startswith(("actions/configure-pages@", "actions/deploy-pages@"))
        for step in build["steps"]
    )
    build_steps = build["steps"]
    docs_artifact = next(step for step in build_steps if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert docs_artifact["with"]["if-no-files-found"] == "error"
    assert any("sphinx-build" in step.get("run", "") and "-W" in step["run"] for step in build_steps)
    assert any(
        "-b doctest" in step.get("run", "") and step["if"] == "${{ inputs.run_doctests }}"
        for step in build_steps
    )
    pages_artifact = next(
        step for step in build_steps if step.get("uses", "").startswith("actions/upload-pages-artifact@")
    )
    assert pages_artifact["with"]["path"] == docs_artifact["with"]["path"]
    assert "if" not in pages_artifact
    configure, deploy = publish["steps"]
    assert configure["uses"].startswith("actions/configure-pages@")
    assert deploy["uses"].startswith("actions/deploy-pages@")
    assert pages_artifact["with"].get("name", "github-pages") == deploy.get("with", {}).get(
        "artifact_name", "github-pages"
    )
    assert publish["environment"]["url"] == "${{ steps." + deploy["id"] + ".outputs.page_url }}"


def test_publishing_preserves_approval_and_serializes_deployments():
    """Retain production approval gates and avoid cancelling an in-progress site deployment."""
    publish = WORKFLOW["jobs"]["publish-docs"]
    assert publish["environment"]["name"] == "prod"
    assert publish["concurrency"] == {"group": "github-pages", "cancel-in-progress": "false"}


def test_pages_permissions_are_scoped_and_supported_by_ci_caller():
    """Build read-only; permit only the deployment job to publish with the built-in token."""
    assert WORKFLOW["permissions"] == {"contents": "read"}
    publish = WORKFLOW["jobs"]["publish-docs"]
    assert publish["permissions"] == {"pages": "write", "id-token": "write"}
    assert CI["jobs"]["docs"]["permissions"] == {"contents": "read", **publish["permissions"]}
    assert "permissions" not in WORKFLOW["jobs"]["create_docs"]
    assert all("secrets." not in str(step) for step in publish["steps"])
