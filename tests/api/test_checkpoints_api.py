import os
import torch
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_list_checkpoints_endpoint():
    res = client.get("/api/checkpoints/list")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_delete_checkpoint_endpoint(tmp_path):
    ckpt_dir = "experiments/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    test_file = os.path.join(ckpt_dir, "test_dummy_ckpt.pt")
    torch.save({"dummy": 1}, test_file)

    # Verify present
    res_list = client.get("/api/checkpoints/list")
    assert any(c["checkpoint_id"] == "test_dummy_ckpt.pt" for c in res_list.json())

    # Delete single
    del_res = client.delete("/api/checkpoints/test_dummy_ckpt.pt")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True
    assert not os.path.exists(test_file)
