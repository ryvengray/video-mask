from pathlib import Path
import asyncio

import pytest

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
    def __init__(self, region="us-east-2"):
        self.region = region
        self.completed_upload = None
        self.aborted_upload = None

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator()

    def head_object(self, **_):
        raise NotFound()

    def generate_presigned_url(self, operation, Params, ExpiresIn):
        return f"https://signed.{self.region}.example/{operation}/{Params['Bucket']}/{Params['Key']}?ttl={ExpiresIn}"

    def get_bucket_location(self, Bucket):
        return {"LocationConstraint": self.region}

    def create_multipart_upload(self, **kwargs):
        self.created_upload = kwargs
        return {"UploadId": "upload-123"}

    def complete_multipart_upload(self, **kwargs):
        self.completed_upload = kwargs

    def abort_multipart_upload(self, **kwargs):
        self.aborted_upload = kwargs


def test_s3_scan_is_idempotent_and_claim_urls_are_fresh(tmp_path: Path):
    store = ClusterStore(tmp_path / "controller.sqlite3")
    ingestor = S3Ingestor(store, "source-bucket", "incoming/", "output-bucket", "masked/",
                          "us-east-2", presign_seconds=3600, client=FakeS3())

    assert ingestor.scan() == 1
    assert ingestor.scan() == 0
    task = store.list_tasks()[0]
    assert task["source_url"] == "s3://source-bucket/incoming/a.mp4"
    assert task["output_object_key"] == "masked/a.mp4"

    claimed = ingestor.materialize(task)
    assert claimed["source_url"].startswith("https://signed.us-east-2.example/get_object/source-bucket/incoming/a.mp4")
    assert claimed["output_upload_url"].startswith("https://signed.us-east-2.example/put_object/output-bucket/masked/a.mp4")
    assert task["source_url"].startswith("s3://")
    assert task["output_upload_url"] == ""
    store.close()


def test_s3_output_key_preserves_source_relative_directories(tmp_path: Path):
    store = ClusterStore(tmp_path / "controller.sqlite3")
    ingestor = S3Ingestor(store, "source-bucket", "source/inbox/", "output-bucket", "outputs/",
                          "us-east-2", client=FakeS3())

    assert ingestor.output_key("source/inbox/s/s/m.mp4") == "outputs/s/s/m.mp4"
    assert ingestor.output_key("source/inbox/other/m.mp4") == "outputs/other/m.mp4"
    store.close()


def test_manual_s3_scan_prefix_is_normalised_and_does_not_change_scheduled_prefix(tmp_path: Path):
    class TrackingPaginator(FakePaginator):
        def __init__(self):
            self.calls = []

        def paginate(self, **kwargs):
            self.calls.append(kwargs)
            return super().paginate(**kwargs)

    class TrackingS3(FakeS3):
        def __init__(self):
            super().__init__()
            self.paginator = TrackingPaginator()

        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return self.paginator

    store = ClusterStore(tmp_path / "controller.sqlite3")
    client = TrackingS3()
    ingestor = S3Ingestor(store, "source-bucket", "incoming/", "output-bucket", "masked/",
                          "us-east-2", client=client)
    assert ingestor.scan_prefix("/test/v3/") == 1
    assert client.paginator.calls[-1] == {"Bucket": "source-bucket", "Prefix": "test/v3/"}
    # The temporary scan target does not replace the scheduled source prefix.
    assert ingestor.source_prefix == "incoming"
    store.close()


def test_s3_separate_output_region_uses_its_own_client(tmp_path: Path):
    store = ClusterStore(tmp_path / "controller.sqlite3")
    ingestor = S3Ingestor(store, "source-bucket", "incoming/", "output-bucket", "masked/",
                          "us-east-2", output_region="ap-southeast-1",
                          client=FakeS3("us-east-2"), output_client=FakeS3("ap-southeast-1"))
    ingestor.validate_bucket_regions()
    assert ingestor.scan() == 1
    claimed = ingestor.materialize(store.list_tasks()[0])
    assert claimed["source_url"].startswith("https://signed.us-east-2.example/")
    assert claimed["output_upload_url"].startswith("https://signed.ap-southeast-1.example/")
    store.close()


def test_s3_upload_url_can_be_refreshed_without_returning_the_source_url(tmp_path: Path):
    store = ClusterStore(tmp_path / "controller.sqlite3")
    ingestor = S3Ingestor(store, "source-bucket", "incoming/", "output-bucket", "masked/",
                          "us-east-2", client=FakeS3("us-east-2"))
    assert ingestor.scan() == 1

    refreshed = ingestor.materialize_upload_url(store.list_tasks()[0])

    assert refreshed["output_object_key"] == "masked/a.mp4"
    assert refreshed["output_upload_url"].startswith("https://signed.us-east-2.example/put_object/")
    assert "source_url" not in refreshed
    store.close()


def test_s3_multipart_upload_is_presigned_and_completed_by_controller(tmp_path: Path):
    store = ClusterStore(tmp_path / "controller.sqlite3")
    output = FakeS3("us-east-2")
    ingestor = S3Ingestor(store, "source-bucket", "incoming/", "output-bucket", "masked/",
                          "us-east-2", presign_seconds=3600, client=FakeS3(), output_client=output)
    assert ingestor.scan() == 1
    task = store.list_tasks()[0]

    started = ingestor.initiate_multipart_upload(task)
    assert started == {"upload_id": "upload-123", "part_size": 64 * 1024 * 1024,
                       "output_object_key": "masked/a.mp4"}
    signed = ingestor.multipart_part_url(task, "upload-123", 1)
    assert signed["upload_part_url"].startswith("https://signed.us-east-2.example/upload_part/")
    assert "upload-123" in signed["upload_part_url"] or "upload_part" in signed["upload_part_url"]

    assert ingestor.complete_multipart_upload(task, "upload-123", [
        {"part_number": 1, "etag": '"etag-1"'},
        {"part_number": 2, "etag": '"etag-2"'},
    ]) == {"output_object_key": "masked/a.mp4"}
    assert output.completed_upload == {
        "Bucket": "output-bucket", "Key": "masked/a.mp4", "UploadId": "upload-123",
        "MultipartUpload": {"Parts": [
            {"PartNumber": 1, "ETag": '"etag-1"'}, {"PartNumber": 2, "ETag": '"etag-2"'},
        ]},
    }
    ingestor.abort_multipart_upload(task, "upload-456")
    assert output.aborted_upload == {"Bucket": "output-bucket", "Key": "masked/a.mp4", "UploadId": "upload-456"}
    store.close()


def test_controller_scans_s3_in_background_without_http_requests(tmp_path: Path):
    try:
        from cluster.controller import create_app
    except TypeError as exc:
        if "eval_type_backport" in str(exc):
            pytest.skip("Controller's Pydantic v2 annotations require Python 3.10+ in this environment")
        raise

    class BackgroundIngestor:
        poll_seconds = 0.01

        def __init__(self):
            self.validated = False
            self.scans = 0

        def validate_bucket_regions(self):
            self.validated = True

        def scan(self):
            self.scans += 1
            return 0

    async def run() -> BackgroundIngestor:
        ingestor = BackgroundIngestor()
        app = create_app(tmp_path / "controller.sqlite3", "a" * 16, s3_ingestor=ingestor)  # type: ignore[arg-type]
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.03)
        return ingestor

    ingestor = asyncio.run(run())
    assert ingestor.validated
    assert ingestor.scans >= 1


def test_controller_manual_s3_scan_runs_while_scheduled_ingestion_is_paused(tmp_path: Path):
    try:
        from cluster.controller import create_app
    except TypeError as exc:
        if "eval_type_backport" in str(exc):
            pytest.skip("Controller's Pydantic v2 annotations require Python 3.10+ in this environment")
        raise

    class ManualIngestor:
        def __init__(self):
            self.scans = 0

        @staticmethod
        def normalise_scan_prefix(value):
            return value.strip().strip("/")

        def scan_prefix(self, source_prefix):
            assert source_prefix == "test/v3"
            self.scans += 1
            return 3

    database = tmp_path / "controller.sqlite3"
    store = ClusterStore(database)
    store.set_boolean_setting("s3_ingest_enabled", False)
    store.close()
    ingestor = ManualIngestor()
    app = create_app(database, "a" * 16, s3_ingestor=ingestor)  # type: ignore[arg-type]
    route = next(route for route in app.routes if route.path == "/api/admin/s3-ingest/scan")

    assert route.endpoint("/test/v3/") == {
        "configured": True, "enabled": False, "source_prefix": "test/v3", "created": 3,
    }
    assert ingestor.scans == 1
