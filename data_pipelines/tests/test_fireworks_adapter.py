import json

import httpx
import pytest

from data_pipelines.providers import fireworks


def test_existing_dataset_is_never_uploaded(fireworks_transport, tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(409, json={"error": "exists"})

    fireworks_transport(handler)
    path = tmp_path / "input.jsonl"
    path.write_text('{"input":"example"}\n')
    with pytest.raises(FileExistsError):
        fireworks.upload_batch_dataset(path, "existing", 1)
    assert len(requests) == 1
    assert not requests[0].url.path.endswith(":upload")


def test_download_failure_does_not_publish_partial_files(fireworks_transport, tmp_path):
    def handler(request):
        if request.url.path.endswith(":getDownloadEndpoint"):
            return httpx.Response(
                200,
                json={
                    "filenameToSignedUrls": {
                        "a/result.jsonl": "https://download.test/good",
                        "b/result.jsonl": "https://download.test/bad",
                    }
                },
            )
        if request.url.path == "/good":
            return httpx.Response(200, content=b'{"ok":true}\n')
        return httpx.Response(503)

    fireworks_transport(handler)
    with pytest.raises(fireworks.FireworksError):
        fireworks.download_batch_dataset("output", tmp_path, 1)
    assert list(tmp_path.iterdir()) == []


def test_colliding_download_names_keep_every_object(fireworks_transport, tmp_path):
    def handler(request):
        if request.url.path.endswith(":getDownloadEndpoint"):
            return httpx.Response(
                200,
                json={
                    "filenameToSignedUrls": {
                        "a/x.jsonl": "https://download.test/one",
                        "b/x.jsonl": "https://download.test/two",
                        "0001-x.jsonl": "https://download.test/three",
                    }
                },
            )
        return httpx.Response(200, text=request.url.path)

    fireworks_transport(handler)
    files = fireworks.download_batch_dataset("output", tmp_path, 1)
    assert len(set(files)) == 3
    assert {file.read_text() for file in files} == {"/one", "/two", "/three"}


@pytest.mark.parametrize("upload_succeeds", [True, False])
def test_create_then_upload_outcomes(fireworks_transport, tmp_path, upload_succeeds):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json={"name": "dataset"})
        return (
            httpx.Response(200, json={"uploaded": True})
            if upload_succeeds
            else httpx.Response(503)
        )

    fireworks_transport(handler)
    path = tmp_path / "batch.jsonl"
    path.write_text('{"input":"one"}\n\n')
    if upload_succeeds:
        assert fireworks.upload_batch_dataset(
            path, "dataset", 1
        ) == fireworks.FireworksUploadResult("dataset", 1)
    else:
        with pytest.raises(fireworks.FireworksError, match="503"):
            fireworks.upload_batch_dataset(path, "dataset", 1)
    assert len(requests) == 2
    assert json.loads(requests[0].content)["dataset"]["exampleCount"] == "1"
    assert requests[1].url.path.endswith("/dataset:upload")
    assert b'{"input":"one"}' in requests[1].content
