from pathlib import Path

from cluster.s3_ingest import S3Ingestor
from cluster.store import ClusterStore


class NotFound(Exception):
    response = {"Error": {"Code": "404"}}


class FakePaginator:
    def paginate(self, **_):
        return [{"Contents": [
            {"Key": "incoming/a.mp4", "Size": 123, "ETag": "etag-a"},
            {"Key": "incoming/readme.txt", "Size": 3, "ETag": "etag-text"},
        ]}]


class FakeS3:
    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator()

    def head_object(self, **_):
        raise NotFound()

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        return f"https://signed.example/{operation}/{Params['Bucket']}/{Params['Key']}?ttl={ExpiresIn}"


def test_s3_scan_is_idempotent_and_claim_urls_are_fresh(tmp_path: Path):
    store = ClusterStore(tmp_path / "controller.sqlite3")
    ingestor = S3Ingestor(store, "source-bucket", "incoming/", "output-bucket", "masked/",
                          "us-east-2", presign_seconds=3600, client=FakeS3())

    assert ingestor.scan() == 1
    assert ingestor.scan() == 0
    task = store.list_tasks()[0]
    assert task["source_url"] == "s3://source-bucket/incoming/a.mp4"
    assert task["output_object_key"] == "masked/masked_a.mp4"

    claimed = ingestor.materialize(task)
    assert claimed["source_url"].startswith("https://signed.example/get_object/source-bucket/incoming/a.mp4")
    assert claimed["output_upload_url"].startswith("https://signed.example/put_object/output-bucket/masked/masked_a.mp4")
    assert task["source_url"].startswith("s3://")
    assert task["output_upload_url"] == ""
    store.close()
