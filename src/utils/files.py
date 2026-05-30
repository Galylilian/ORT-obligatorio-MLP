import os
from uuid import uuid4


def save_uploaded_file(file, folder="/tmp"):
    """
    Guarda el archivo subido en un path temporal.
    """

    os.makedirs(folder, exist_ok=True)

    filename = f"{uuid4()}.jpg"
    filepath = os.path.join(folder, filename)

    with open(filepath, "wb") as f:
        f.write(file.file.read())

    return filepath


def delete_file(path):
    """
    Borra archivo temporal.
    """
    if os.path.exists(path):
        os.remove(path)