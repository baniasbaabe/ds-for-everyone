import pytest

from src.apps import data_upload_app


@pytest.mark.parametrize(
    "string, true_extension",
    [("file.csv", "csv"), ("file.xlsx", "xlsx"), ("file.file.xlsx", "xlsx")],
)
def test_get_file_type(string, true_extension):
    extension = data_upload_app.get_file_type(string)
    assert extension == true_extension
